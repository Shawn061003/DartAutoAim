#include "Communication/Communication.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace std::chrono_literals;

// 仅测试目标通过 --wrap=write 接管写调用，模拟串口短写和暂不可写。
// 正常库与 demo 都直接使用系统 write，不包含此故障注入。
std::atomic<int> write_mode{0};
std::atomic<unsigned> write_calls{0};
extern "C" ssize_t __real_write(int, const void*, size_t);
extern "C" ssize_t __wrap_write(int fd, const void* data, size_t size) {
    const int mode = write_mode.load();
    if (mode && fd > STDERR_FILENO && ::isatty(fd)) {
        const auto call = ++write_calls;
        if (mode == 2) { errno = EAGAIN; return -1; }
        if (call == 1) { errno = EINTR; return -1; }
        if (call == 2) { errno = EAGAIN; return -1; }
        size = std::min<size_t>(3, size);
    }
    return __real_write(fd, data, size);
}

namespace {
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <class Function>
void waitFor(Function predicate, const char* message) {
    const auto deadline = std::chrono::steady_clock::now() + 3s;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error(message);
        std::this_thread::sleep_for(1ms);
    }
}

template <class Function>
void mustThrow(Function action, const char* message) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    check(threw, message);
}

// Linux 伪终端：master 模拟 MCU，slave 交给通信库，完全不接触真实串口。
struct PseudoMcu {
    int master = -1;
    std::string port;
    PseudoMcu() {
        master = ::posix_openpt(O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
        check(master >= 0, "创建伪终端失败");
        if (::grantpt(master) < 0 || ::unlockpt(master) < 0) {
            ::close(master);
            throw std::runtime_error("初始化伪终端失败");
        }
        char name[256]{};
        if (::ptsname_r(master, name, sizeof(name)) != 0) {
            ::close(master);
            throw std::runtime_error("获取伪终端名称失败");
        }
        port = name;
    }
    ~PseudoMcu() { if (master >= 0) ::close(master); }
    void disconnect() { ::close(master); master = -1; }
    void transmit(const std::vector<std::uint8_t>& bytes) {
        std::size_t sent = 0;
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        while (sent < bytes.size()) {
            check(std::chrono::steady_clock::now() < deadline, "模拟电控发送超时");
            const auto count = ::write(master, bytes.data() + sent, bytes.size() - sent);
            if (count > 0) sent += static_cast<std::size_t>(count);
            else {
                check(errno == EINTR || errno == EAGAIN, "模拟电控发送失败");
                pollfd wait{master, POLLOUT, 0};
                ::poll(&wait, 1, 20);
            }
        }
    }
    std::vector<std::uint8_t> receive(std::size_t size) {
        std::vector<std::uint8_t> data(size);
        std::size_t received = 0;
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        while (received < size) {
            check(std::chrono::steady_clock::now() < deadline, "模拟电控接收超时");
            pollfd wait{master, POLLIN, 0};
            if (::poll(&wait, 1, 20) <= 0) continue;
            const auto count = ::read(master, data.data() + received, size - received);
            if (count > 0) received += static_cast<std::size_t>(count);
            else check(errno == EINTR || errno == EAGAIN, "模拟电控接收失败");
        }
        return data;
    }
    bool hasData() const {
        pollfd wait{master, POLLIN, 0};
        return ::poll(&wait, 1, 30) > 0 && (wait.revents & POLLIN);
    }
};

const std::vector<std::uint8_t> moving{0x42,0x52,3,4,0,0,0,0,0x12};
const std::vector<std::uint8_t> stationary{0x42,0x52,3,4,1,0,0,0,0x9D};
const std::vector<std::uint8_t> golden_command{
    0x42,0x52,3,0x14,1,0,0,0,0,0,0,0,0,0,0xC0,0x3F,0,0,0,0,0,0,0x3E,0x40,0xED};

void testReceiveAndSend() {
    PseudoMcu mcu;
    dart_mcu::Communication communication;
    check(!communication.latestStatus(1s), "启动前不应有状态");
    mustThrow([&] { communication.send({1,0,0}); }, "未启动时发送未报错");
    mustThrow([&] { communication.start(mcu.port, 12345); }, "非法波特率未报错");
    std::atomic<unsigned> callbacks{0}, bad_frames{0};
    communication.start(mcu.port, 115200,
        [&](const dart_mcu::ReceivedStatus&) {
            // 回调读取状态，验证调用时没有持有状态锁。
            check(communication.latestStatus(1s).has_value(), "回调中无法获取状态");
            ++callbacks;
        },
        [&](const std::string& message) {
            if (message.find("RX 校验失败") != std::string::npos) ++bad_frames;
        });
    check(!communication.latestStatus(1s), "收到首帧前状态应未知");
    check(!mcu.hasData(), "启动后意外自动发送");
    mustThrow([&] { communication.start(mcu.port); }, "重复启动未报错");
    check(communication.isRunning(), "重复启动破坏原连接");

    // 噪声中包含半个帧头，随后逐段补齐。
    mcu.transmit({0x99, 0x42, 0x42, 0x52});
    std::this_thread::sleep_for(20ms);
    check(!communication.latestStatus(1s), "半帧被接受");
    mcu.transmit({3,4,0});
    std::this_thread::sleep_for(20ms);
    check(!communication.latestStatus(1s), "半帧被接受");
    mcu.transmit({0,0,0,0x12});
    waitFor([&] { return callbacks == 1; }, "接收 loop 未解析移动帧");
    auto status = communication.latestStatus(1s);
    check(status && status->status.is_stationary == 0 && status->sequence == 1,
          "移动状态字段错误");

    auto bad = stationary;
    bad.back() ^= 1;
    const std::vector<std::uint8_t> invalid_state{0x42,0x52,3,4,2,0,0,0,0x15};
    bad.insert(bad.end(), invalid_state.begin(), invalid_state.end());
    bad.insert(bad.end(), stationary.begin(), stationary.end());
    bad.insert(bad.end(), moving.begin(), moving.end());
    mcu.transmit(bad);
    waitFor([&] { return callbacks == 3; }, "坏帧恢复或粘包解析失败");
    check(bad_frames == 2, "错误帧未被拒绝");
    status = communication.latestStatus(1s);
    check(status && status->status.is_stationary == 0 && status->sequence == 3,
          "最新状态没有更新");

    std::this_thread::sleep_for(30ms);
    check(!communication.latestStatus(10ms), "过期状态仍有效");
    check(communication.isRunning(), "状态过期不应停止串口接收");
    mustThrow([&] { communication.latestStatus(-1ms); }, "负状态有效期未报错");
    mustThrow([&] { communication.send({2,0,0}); }, "非法命令未拒绝");
    mustThrow([&] { communication.send({1,0,0}, 0ms); }, "非法发送超时未拒绝");
    check(communication.isRunning() && !mcu.hasData(), "无效输入改变连接或发送数据");

    // 注入 EINTR、EAGAIN 和每次最多 3 字节的短写。
    write_calls = 0;
    write_mode = 1;
    communication.send({1,0.125,30});
    write_mode = 0;
    check(write_calls >= 11, "短写分支未执行");
    check(mcu.receive(25) == golden_command, "短写后完整帧与固定样例不一致");

    // 多个调用线程同时发送，接收端应始终得到边界完整的帧。
    std::atomic<unsigned> send_errors{0};
    std::vector<std::thread> senders;
    for (int i = 0; i < 8; ++i) senders.emplace_back([&, i] {
        try { communication.send({0, static_cast<double>(i), 30}); }
        catch (...) { ++send_errors; }
    });
    for (auto& thread : senders) thread.join();
    check(send_errors == 0, "并发发送失败");
    const auto frames = mcu.receive(8 * 25);
    std::vector<int> values;
    for (std::size_t offset = 0; offset < frames.size(); offset += 25) {
        dart_mcu::Command command;
        check(dart_mcu::decode_command(frames.data() + offset, 25, command),
              "并发发送造成帧交错");
        values.push_back(static_cast<int>(command.yaw_error_rad));
    }
    std::sort(values.begin(), values.end());
    check(values == std::vector<int>({0,1,2,3,4,5,6,7}), "并发命令丢失或重复");
    check(!mcu.hasData(), "模块自动重复发送了命令");

    const auto begin = std::chrono::steady_clock::now();
    communication.stop();
    check(std::chrono::steady_clock::now() - begin < 500ms, "停止空闲接收线程超时");
    check(!communication.isRunning() && !communication.latestStatus(1s), "停止后状态仍有效");
    communication.stop(); // 重复停止应可安全执行。
}

void testDisconnectAndRestart() {
    PseudoMcu first, second;
    dart_mcu::Communication communication;
    communication.start(first.port);
    first.transmit(stationary);
    waitFor([&] { return communication.latestStatus(1s).has_value(); }, "未收到初始静止状态");
    first.disconnect();
    waitFor([&] { return !communication.isRunning(); }, "断线后接收循环未退出");
    check(!communication.latestStatus(1s) && !communication.lastError().empty(),
          "断线后状态或错误信息不正确");
    communication.start(second.port);
    check(communication.lastError().empty() && !communication.latestStatus(1s),
          "重新启动残留旧状态");
    second.transmit(moving);
    waitFor([&] { return communication.latestStatus(1s).has_value(); }, "重新启动后无法接收");
    const auto status = communication.latestStatus(1s);
    check(status->sequence == 1 && status->status.is_stationary == 0, "重新启动未重置会话");
}

void testTimeoutAndCancellation() {
    PseudoMcu mcu;
    dart_mcu::Communication communication;
    communication.start(mcu.port);
    write_calls = 0;
    write_mode = 2;
    mustThrow([&] { communication.send({1,0,30}, 30ms); }, "持续不可写未超时");
    write_mode = 0;
    check(!communication.isRunning() && !communication.lastError().empty(),
          "发送超时未使会话失效");
    mustThrow([&] { communication.send({1,0,30}); }, "发送故障后仍允许新命令");

    communication.start(mcu.port);
    write_calls = 0;
    write_mode = 2;
    std::atomic<bool> cancelled{false};
    std::thread sender([&] {
        try { communication.send({1,0,30}, 5s); }
        catch (const std::exception&) { cancelled = true; }
    });
    waitFor([] { return write_calls > 1; }, "发送线程未进入等待");
    const auto begin = std::chrono::steady_clock::now();
    communication.stop();
    sender.join();
    write_mode = 0;
    check(cancelled && std::chrono::steady_clock::now() - begin < 500ms,
          "stop 未及时取消等待中的发送");
}

void testCallbackFailureAndRestart() {
    PseudoMcu mcu;
    dart_mcu::Communication communication;
    communication.start(mcu.port, 115200, [](const dart_mcu::ReceivedStatus&) {
        throw std::runtime_error("callback test failure");
    });
    mcu.transmit(stationary);
    waitFor([&] { return !communication.isRunning(); }, "回调异常未停止接收");
    check(!communication.latestStatus(1s) &&
          communication.lastError().find("callback test failure") != std::string::npos,
          "回调异常没有使旧状态失效");
    communication.start(mcu.port);
    mcu.transmit(moving);
    waitFor([&] { return communication.latestStatus(1s).has_value(); }, "回调故障后无法重启");
}

} // 匿名命名空间

int main() {
    try {
        testReceiveAndSend();
        testDisconnectAndRestart();
        testTimeoutAndCancellation();
        testCallbackFailureAndRestart();
        std::cout << "Communication 测试通过：接收循环、分包/粘包、CRC、状态时效、"
                     "短写、并发发送、断线/重启、取消和异常退出。\n";
        return 0;
    } catch (const std::exception& error) {
        write_mode = 0;
        std::cerr << error.what() << '\n';
        return 1;
    }
}
