// 镖架串口联调程序：主窗口输入命令，独立终端显示电控回传。
// 示例：./build/dart_control_demo --port /dev/ttyACM0 --baud 115200
// 输入：1 2.1 30（允许开火、向左偏差 2.1 rad、绝对伸缩目标 30 mm）。
// 每行只发送一次；输入 q、quit 或按 Ctrl+C 退出，不自动重发旧命令。
#include "Communication/Communication.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <optional>
#include <poll.h>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace {
volatile std::sig_atomic_t interrupted = 0;
void onSignal(int) { interrupted = 1; }

// 待上位机接口确定后实现：自动寻找电控串口。
// 暂时返回空值，由 --port 显式指定路径。
std::optional<std::string> findSerialPort() {
    return std::nullopt;
}

// 待实现：根据上位机设备信息核对串口是否属于电控。
// nullopt 表示尚未验证身份；后续实现可返回 true / false。
// open、termios、read、write 的系统错误仍然会检查。
std::optional<bool> validateSerialPort(const std::string&) {
    return std::nullopt;
}

struct Options {
    std::string port;
    int baud = 115200;
    bool new_window = true;
    std::string rx_log;
};

void usage() {
    std::cout
        << "用法：dart_control_demo --port <串口路径> [--baud 115200]\n"
        << "                      [--rx-log <日志路径>] [--no-window]\n"
        << "每行输入：开火许可(0/1) yaw偏差(rad，左正右负) 绝对伸缩量(mm)\n"
        << "示例：1 2.1 30；输入 q / quit 或 Ctrl+C 退出。\n"
        << "自动查找和设备身份验证尚未实现，暂时必须手动指定 --port。\n";
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-window") options.new_window = false;
        else if (arg == "--port" || arg == "--baud" || arg == "--rx-log") {
            if (++i == argc) throw std::runtime_error(arg + " 缺少参数");
            if (arg == "--port") options.port = argv[i];
            else if (arg == "--rx-log") options.rx_log = argv[i];
            else {
                std::size_t used = 0;
                options.baud = std::stoi(argv[i], &used);
                if (used != std::string(argv[i]).size())
                    throw std::runtime_error("波特率必须为整数");
            }
        } else throw std::runtime_error("未知参数：" + arg);
    }
    return options;
}

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
        default: throw std::runtime_error("暂不支持该波特率");
    }
}

// 文件描述符由此对象持有，作用域结束时自动关闭。
struct FileDescriptor {
    int value = -1;
    ~FileDescriptor() { if (value >= 0) ::close(value); }
    FileDescriptor() = default;
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
};

[[noreturn]] void systemError(const char* operation) {
    throw std::system_error(errno, std::generic_category(), operation);
}

void openSerial(const Options& options, FileDescriptor& serial) {
    const speed_t baud = baudConstant(options.baud);
    serial.value = ::open(options.port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
    if (serial.value < 0) systemError("打开串口失败");

    termios settings{};
    if (::tcgetattr(serial.value, &settings) < 0) systemError("读取串口配置失败");
    ::cfmakeraw(&settings); // 关闭回显、行缓冲和字符转换，原样传输二进制数据。
    settings.c_cflag &= ~(CSIZE | PARENB | CSTOPB | CRTSCTS);
    settings.c_cflag |= CS8 | CLOCAL | CREAD; // 8N1，允许接收，不使用硬件流控。
    settings.c_iflag &= ~(IXON | IXOFF | IXANY);
    settings.c_cc[VMIN] = 0;
    settings.c_cc[VTIME] = 0; // 使用 poll 等待数据，避免忙等。
    if (::cfsetispeed(&settings, baud) < 0 || ::cfsetospeed(&settings, baud) < 0)
        systemError("设置波特率失败");
    if (::tcsetattr(serial.value, TCSANOW, &settings) < 0)
        systemError("应用串口配置失败");
    // 初始化时丢弃此前残留的输入，随后所有回传都交给接收线程。
    if (::tcflush(serial.value, TCIFLUSH) < 0) systemError("清理旧接收数据失败");
}

std::string hexBytes(const std::uint8_t* data, std::size_t size) {
    std::ostringstream text;
    text << std::hex << std::uppercase << std::setfill('0');
    for (std::size_t i = 0; i < size; ++i) {
        if (i) text << ' ';
        text << std::setw(2) << static_cast<unsigned>(data[i]);
    }
    return text.str();
}

// 发送主函数：编码一帧后完整写出。短写时继续发送剩余部分。
// EINTR 重新尝试；EAGAIN 等待串口可写。总超时 1 秒，失败后退出联调，
// 避免在半帧后继续追加新的控制命令。成功仅表示本地写入完成。
void writeCommand(int fd, const dart_mcu::Command& command,
                  const std::atomic<bool>& running) {
    dart_mcu::Frame frame{};
    if (!dart_mcu::encode(command, frame))
        throw std::runtime_error("命令字段无效");
    std::size_t sent = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (sent < frame.size()) {
        if (!running || interrupted) throw std::runtime_error("发送已取消");
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("发送超时，已写入 " + std::to_string(sent) + "/25 字节");
        const ssize_t count = ::write(fd, frame.data() + sent, frame.size() - sent);
        if (count > 0) {
            sent += static_cast<std::size_t>(count);
            continue;
        }
        if (count < 0 && errno == EINTR) continue;
        if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
            systemError("串口写入失败");
        pollfd wait{fd, POLLOUT, 0};
        const int ready = ::poll(&wait, 1, 50);
        if (ready < 0 && errno != EINTR) systemError("等待串口可写失败");
        if (ready > 0 && (wait.revents & (POLLERR | POLLHUP | POLLNVAL)))
            throw std::runtime_error("串口发送连接已断开");
    }
    std::cout << "TX 25 字节已写入：" << hexBytes(frame.data(), frame.size()) << '\n';
}

// 日志仅由接收线程写入；每行带本地单调时钟的相对时间。
// 日志时间代表上位机处理时间，单位 ms；当前协议尚无电控时间戳或任务编号。
class ReceiveLog {
public:
    explicit ReceiveLog(const std::string& path) : output_(path, std::ios::app) {
        if (!output_) throw std::runtime_error("无法打开接收日志：" + path);
    }
    void line(const std::string& text) {
        const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_).count();
        output_ << "[+" << milliseconds << " ms] " << text << std::endl;
        if (!output_) throw std::runtime_error("接收日志写入失败");
    }
private:
    std::ofstream output_;
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
};

// 接收主函数：读字节流、拼接缓存、解析 9 字节状态帧。
// 只在完整帧通过 decode 校验后显示移动/静止；坏帧逐字节重新同步。
// 每次解析后最多保留 8 字节的未完成帧，防止噪声使缓存无限增长。
void readStatus(int fd, std::atomic<bool>& running, std::atomic<bool>& failed,
                ReceiveLog& log) {
    try {
        std::vector<std::uint8_t> buffer;
        const std::array<std::uint8_t, 4> header{
            0x42, 0x52, dart_mcu::kVersion,
            static_cast<std::uint8_t>(dart_mcu::kReceivePayloadSize)};
        log.line("等待电控回传；尚未收到有效状态。");
        while (running) {
            pollfd wait{fd, POLLIN, 0};
            const int ready = ::poll(&wait, 1, 100);
            if (ready < 0) {
                if (errno == EINTR) continue;
                systemError("等待串口数据失败");
            }
            if (ready == 0) continue;
            if (wait.revents & POLLIN) {
                std::uint8_t bytes[256];
                const ssize_t count = ::read(fd, bytes, sizeof(bytes));
                if (count < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
                    systemError("串口读取失败");
                if (count > 0) {
                    buffer.insert(buffer.end(), bytes, bytes + count);
                    while (buffer.size() >= header.size()) {
                        const auto begin = std::search(buffer.begin(), buffer.end(),
                                                       header.begin(), header.end());
                        if (begin == buffer.end()) {
                            buffer.erase(buffer.begin(), buffer.end() - 3);
                            break;
                        }
                        buffer.erase(buffer.begin(), begin);
                        if (buffer.size() < dart_mcu::kReceiveFrameSize) break;
                        dart_mcu::Status status;
                        if (dart_mcu::decode(buffer.data(), dart_mcu::kReceiveFrameSize, status)) {
                            log.line("RX is_stationary=" + std::to_string(status.is_stationary) +
                                     (status.is_stationary ? " 镖架静止" : " 镖架移动中") +
                                     " | " + hexBytes(buffer.data(), dart_mcu::kReceiveFrameSize));
                            buffer.erase(buffer.begin(), buffer.begin() + dart_mcu::kReceiveFrameSize);
                        } else {
                            log.line("RX 校验失败：丢弃候选帧并重新同步。");
                            buffer.erase(buffer.begin());
                        }
                    }
                }
            }
            if (wait.revents & (POLLERR | POLLHUP | POLLNVAL))
                throw std::runtime_error("串口接收连接已断开");
        }
        log.line("接收结束。");
    } catch (const std::exception& error) {
        try { log.line(std::string("接收错误：") + error.what()); } catch (...) {}
        std::cerr << "\n接收错误：" << error.what() << '\n';
        failed = true;
        running = false;
    }
}

std::string makeLogPath() {
    char path[] = "/tmp/dart_control_demo-rx-XXXXXX";
    const int fd = ::mkstemp(path);
    if (fd < 0) systemError("创建接收日志失败");
    ::close(fd);
    return path;
}

std::string shellQuote(const std::string& value) {
    std::string quoted = "'";
    for (const char c : value) quoted += (c == '\'' ? "'\\''" : std::string(1, c));
    return quoted + "'";
}

// 新窗口仅跟随日志，不打开串口。自动尝试系统默认终端、GNOME Terminal、xterm。
// 使用参数数组启动程序，无 shell 拼接执行；主程序退出后 tail 自动结束。
void openReceiveWindow(const std::string& log_path, bool enabled) {
    const std::string pid_arg = "--pid=" + std::to_string(::getpid());
    std::cout << "接收日志：" << log_path << "\n"
              << "也可在另一终端执行：tail " << pid_arg << " -n +1 -f "
              << shellQuote(log_path) << '\n';
    if (!enabled || (!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY"))) {
        std::cout << "自动窗口未启用或没有桌面环境，请用上述命令查看回传。\n";
        return;
    }
    const std::vector<std::vector<std::string>> launchers{
        {"x-terminal-emulator", "-e"},
        {"gnome-terminal", "--"},
        {"xterm", "-T", "Dart MCU RX", "-e"}};
    for (auto args : launchers) {
        args.insert(args.end(), {"tail", pid_arg, "-n", "+1", "-f", log_path});
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        argv.push_back(nullptr);
        posix_spawn_file_actions_t actions;
        if (::posix_spawn_file_actions_init(&actions) != 0) break;
        int setup = ::posix_spawn_file_actions_addopen(
            &actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
        if (setup == 0) setup = ::posix_spawn_file_actions_addopen(
            &actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
        if (setup == 0) setup = ::posix_spawn_file_actions_addopen(
            &actions, STDERR_FILENO, log_path.c_str(), O_WRONLY | O_APPEND, 0600);
        pid_t child = -1;
        const int result = setup == 0 ?
            ::posix_spawnp(&child, argv[0], &actions, nullptr, argv.data(), environ) : setup;
        ::posix_spawn_file_actions_destroy(&actions);
        if (result == 0) {
            std::cout << "已请求打开接收窗口；若未显示，请查看日志或运行上述 tail 命令。\n";
            return;
        }
    }
    std::cout << "未能启动终端窗口，请在另一终端运行上述 tail 命令。\n";
}

// 一行必须恰好包含三个数值；拒绝多余字段、非 0/1 许可和非有限浮点数。
void handleLine(const std::string& line, int fd, std::atomic<bool>& running) {
    std::istringstream input(line);
    input.imbue(std::locale::classic());
    input >> std::ws;
    if (input.eof()) return;
    if (line == "q" || line == "quit" || line == "exit") {
        running = false;
        return;
    }
    dart_mcu::Command command;
    if (!(input >> command.fire_allowed >> command.yaw_error_rad >> command.stretch_target_mm)) {
        std::cout << "输入无效，请输入三个数值，例如：1 2.1 30\n";
        return;
    }
    input >> std::ws;
    if (!input.eof() || !dart_mcu::valid(command)) {
        std::cout << "输入无效：许可只能为 0/1，角度和伸缩量须为有限数值，且不能有多余字段。\n";
        return;
    }
    writeCommand(fd, command, running);
}

// poll + read 同时支持键盘输入和管道输入。无换行的半行不会阻塞接收错误处理。
void inputLoop(int fd, std::atomic<bool>& running) {
    std::string pending;
    bool discard = false;
    std::cout << "输入命令（例如 1 2.1 30），q 退出：\n" << std::flush;
    while (running && !interrupted) {
        pollfd input{STDIN_FILENO, POLLIN, 0};
        const int ready = ::poll(&input, 1, 100);
        if (ready < 0) {
            if (errno == EINTR) continue;
            systemError("等待键盘输入失败");
        }
        if (!ready) continue;
        if (input.revents & (POLLERR | POLLNVAL)) throw std::runtime_error("标准输入不可用");
        if (input.revents & (POLLIN | POLLHUP)) {
            char bytes[256];
            const ssize_t count = ::read(STDIN_FILENO, bytes, sizeof(bytes));
            if (count < 0) {
                if (errno == EINTR || errno == EAGAIN) continue;
                systemError("读取键盘输入失败");
            }
            if (count == 0) {
                if (!discard && !pending.empty()) handleLine(pending, fd, running);
                break;
            }
            for (ssize_t i = 0; i < count && running && !interrupted; ++i) {
                if (bytes[i] == '\n') {
                    if (!discard) {
                        if (!pending.empty() && pending.back() == '\r') pending.pop_back();
                        handleLine(pending, fd, running);
                    }
                    pending.clear();
                    discard = false;
                } else if (!discard) {
                    pending += bytes[i];
                    if (pending.size() > 4096) {
                        std::cout << "输入过长，本行已丢弃。\n";
                        pending.clear();
                        discard = true;
                    }
                }
            }
            std::cout << std::flush;
        }
    }
}

} // 匿名命名空间

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") { usage(); return 0; }
        Options options = parseOptions(argc, argv);
        if (options.port.empty()) options.port = findSerialPort().value_or("");
        if (options.port.empty()) { usage(); return 1; }
        const auto identity = validateSerialPort(options.port);
        if (identity && !*identity) throw std::runtime_error("设备身份验证失败");
        if (!identity) std::cout << "使用手动指定的串口，设备身份验证接口待实现。\n";
        FileDescriptor serial;
        openSerial(options, serial);
        if (options.rx_log.empty()) options.rx_log = makeLogPath();
        ReceiveLog log(options.rx_log);
        log.line("会话开始，串口=" + options.port + "，波特率=" + std::to_string(options.baud));
        std::signal(SIGINT, onSignal);
        std::signal(SIGTERM, onSignal);
        std::signal(SIGCHLD, SIG_IGN); // 自动回收终端启动进程，避免僵尸进程。
        openReceiveWindow(options.rx_log, options.new_window);

        std::atomic<bool> running{true};
        std::atomic<bool> receive_failed{false};
        std::thread receiver(readStatus, serial.value, std::ref(running),
                             std::ref(receive_failed), std::ref(log));
        int result = 0;
        try { inputLoop(serial.value, running); }
        catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            result = 1;
        }
        running = false;
        receiver.join(); // 接收线程退出后才关闭串口。
        if (receive_failed) result = 1;
        std::cout << "程序结束。接收记录保存在：" << options.rx_log << '\n';
        return result;
    } catch (const std::exception& error) {
        std::cerr << "启动失败：" << error.what() << '\n';
        return 1;
    }
}
