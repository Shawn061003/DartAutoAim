#include "Communication/Communication.hpp"

#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <system_error>
#include <termios.h>
#include <unistd.h>
#include <utility>

namespace dart_mcu {
namespace {

speed_t baudConstant(int baud) {
    switch (baud) {
        case 9600: return B9600;
        case 19200: return B19200;
        case 38400: return B38400;
        case 57600: return B57600;
        case 115200: return B115200;
        case 230400: return B230400;
        case 460800: return B460800;
        case 921600: return B921600;
        default: throw std::invalid_argument("不支持的串口波特率");
    }
}

[[noreturn]] void systemError(const char* operation) {
    throw std::system_error(errno, std::generic_category(), operation);
}

void configureSerial(int fd, speed_t baud) {
    termios settings{};
    if (::tcgetattr(fd, &settings) < 0) systemError("读取串口配置失败");
    ::cfmakeraw(&settings); // 二进制传输，关闭回显、字符转换和行缓冲。
    settings.c_cflag &= ~(CSIZE | PARENB | CSTOPB | CRTSCTS);
    settings.c_cflag |= CS8 | CLOCAL | CREAD;
    settings.c_iflag &= ~(IXON | IXOFF | IXANY);
    settings.c_cc[VMIN] = 0;
    settings.c_cc[VTIME] = 0; // 由 poll 等待，避免非阻塞 read 忙等。
    if (::cfsetispeed(&settings, baud) < 0 || ::cfsetospeed(&settings, baud) < 0)
        systemError("设置波特率失败");
    if (::tcsetattr(fd, TCSANOW, &settings) < 0) systemError("应用串口配置失败");
    if (::tcflush(fd, TCIFLUSH) < 0) systemError("清理旧接收数据失败");
}

} // 匿名命名空间

Communication::~Communication() {
    stop();
}

void Communication::start(const std::string& port, int baud,
                          StatusCallback on_status,
                          DiagnosticCallback on_diagnostic) {
    if (port.empty()) throw std::invalid_argument("串口路径不能为空");
    const speed_t speed = baudConstant(baud);
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (running_) throw std::logic_error("通信已启动，请先 stop 再重新 start");
    stopLocked(); // 回收上次故障留下的线程和描述符。

    // 与并发 send 互斥，防止打开到一半就被发送方使用。
    std::lock_guard<std::mutex> send_lock(send_mutex_);
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        latest_status_.reset();
        last_error_.clear();
        receive_sequence_ = 0;
    }
    receive_buffer_.clear();
    on_status_ = std::move(on_status);
    on_diagnostic_ = std::move(on_diagnostic);
    try {
        fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
        if (fd_ < 0) systemError("打开串口失败");
        configureSerial(fd_, speed);
        running_ = true;
        receiver_ = std::thread(&Communication::loop, this);
    } catch (const std::exception& error) {
        fail(error.what());
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        throw;
    }
}

void Communication::stopLocked() {
    running_ = false;
    // 接收线程仍可能处于 poll 中；最多等待一个轮询超时后便退出。
    // 此时不关闭 fd，避免其他线程复用同一编号造成误读误写。
    if (receiver_.joinable()) receiver_.join();
    std::lock_guard<std::mutex> send_lock(send_mutex_);
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    latest_status_.reset();
}

void Communication::stop() {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    stopLocked();
}

bool Communication::isRunning() const noexcept {
    return running_.load();
}

void Communication::fail(const std::string& message) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    // 保留最初的故障原因，避免随后发生的“发送取消”等信息掩盖原因。
    if (last_error_.empty()) last_error_ = message;
    running_ = false;
    latest_status_.reset();
}

std::string Communication::lastError() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return last_error_;
}

std::optional<ReceivedStatus> Communication::latestStatus(
        std::chrono::milliseconds max_age) const {
    if (max_age.count() < 0) throw std::invalid_argument("状态有效期不能为负数");
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!running_ || !latest_status_ ||
        std::chrono::steady_clock::now() - latest_status_->received_at > max_age) {
        return std::nullopt;
    }
    return latest_status_;
}

void Communication::send(const Command& command, std::chrono::milliseconds timeout) {
    Frame frame{};
    if (!encode(command, frame)) throw std::invalid_argument("命令字段无效");
    if (timeout.count() <= 0) throw std::invalid_argument("发送超时必须大于零");

    std::lock_guard<std::mutex> lock(send_mutex_);
    if (!running_ || fd_ < 0) throw std::runtime_error("通信未启动或已停止");
    std::size_t sent = 0;
    const auto started = std::chrono::steady_clock::now();
    try {
        while (sent < frame.size()) {
            if (!running_) throw std::runtime_error("通信停止，发送已取消");
            const auto elapsed = std::chrono::steady_clock::now() - started;
            if (elapsed >= timeout)
                throw std::runtime_error("发送超时，已写入 " + std::to_string(sent) + "/25 字节");

            const ssize_t count = ::write(fd_, frame.data() + sent, frame.size() - sent);
            if (count > 0) {
                sent += static_cast<std::size_t>(count);
                continue;
            }
            if (count < 0 && errno == EINTR) continue;
            if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
                systemError("串口写入失败");

            // 缓冲区暂时满时等待可写，短暂唤醒也用于响应 stop。
            const auto remaining = timeout -
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - started);
            const int wait_ms = static_cast<int>(
                std::max<std::int64_t>(0, std::min<std::int64_t>(50, remaining.count())));
            pollfd wait{fd_, POLLOUT, 0};
            const int ready = ::poll(&wait, 1, wait_ms);
            if (ready < 0 && errno != EINTR) systemError("等待串口可写失败");
            if (ready > 0 && (wait.revents & (POLLERR | POLLHUP | POLLNVAL)))
                throw std::runtime_error("串口发送连接已断开");
        }
    } catch (const std::exception& error) {
        if (running_) fail(error.what());
        throw;
    }
}

// 接收 loop 只在 start 后运行；poll 有数据时立即唤醒，无数据时等待。
// 100 ms 是空闲时检查退出标志的上限，不是固定接收周期或实时调度保证。
void Communication::loop() {
    try {
        if (on_diagnostic_) on_diagnostic_("等待电控回传；尚未收到有效状态。");
        while (running_) {
            pollfd wait{fd_, POLLIN, 0};
            const int ready = ::poll(&wait, 1, 100);
            if (ready < 0) {
                if (errno == EINTR) continue;
                systemError("等待串口数据失败");
            }
            if (!running_) break;
            if (ready == 0) continue;
            if (wait.revents & POLLIN) receiveOnce();
            if (wait.revents & (POLLERR | POLLHUP | POLLNVAL))
                throw std::runtime_error("串口接收连接已断开");
        }
    } catch (const std::exception& error) {
        if (running_) fail(error.what());
    } catch (...) {
        if (running_) fail("接收回调或循环抛出未知异常");
    }

    // 回调异常也由后台线程收拢，避免未捕获异常终止整个进程。
    try {
        if (on_diagnostic_) {
            const auto error = lastError();
            on_diagnostic_(error.empty() ? "接收结束。" : "接收错误：" + error);
        }
    } catch (...) {}
}

void Communication::receiveOnce() {
    std::uint8_t bytes[256];
    const ssize_t count = ::read(fd_, bytes, sizeof(bytes));
    if (count < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) return;
        systemError("串口读取失败");
    }
    if (count == 0) return;
    receive_buffer_.insert(receive_buffer_.end(), bytes, bytes + count);
    parseBuffer();
}

void Communication::parseBuffer() {
    const std::array<std::uint8_t, 4> header{
        0x42, 0x52, kVersion, static_cast<std::uint8_t>(kReceivePayloadSize)};
    while (running_ && receive_buffer_.size() >= header.size()) {
        const auto begin = std::search(receive_buffer_.begin(), receive_buffer_.end(),
                                       header.begin(), header.end());
        if (begin == receive_buffer_.end()) {
            // 留下可能是帧头起始的最后 3 字节，等待下次 read 补齐。
            receive_buffer_.erase(receive_buffer_.begin(), receive_buffer_.end() - 3);
            return;
        }
        receive_buffer_.erase(receive_buffer_.begin(), begin);
        if (receive_buffer_.size() < kReceiveFrameSize) return;

        Status status;
        if (!decode(receive_buffer_.data(), kReceiveFrameSize, status)) {
            if (on_diagnostic_) on_diagnostic_("RX 校验失败：丢弃候选帧并重新同步。");
            receive_buffer_.erase(receive_buffer_.begin());
            continue;
        }

        ReceivedStatus received;
        received.status = status;
        received.received_at = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!running_) return;
            received.sequence = ++receive_sequence_;
            latest_status_ = received;
        }
        receive_buffer_.erase(receive_buffer_.begin(), receive_buffer_.begin() + kReceiveFrameSize);
        // 业务回调在状态锁外调用，允许回调读取最新状态或按需发送。
        if (on_status_) on_status_(received);
    }
}

} // 命名空间 dart_mcu
