// 镖架串口联调程序：主窗口输入命令，独立终端显示电控回传。
// 示例：./build/dart_control_demo --port /dev/ttyACM0 --baud 115200
// 输入：1 2.1 30（允许开火、向右偏差 2.1 rad、绝对伸缩目标 30 mm）。
// 每行只发送一次；输入 q、quit 或按 Ctrl+C 退出，不自动重发旧命令。
#include "Communication/Communication.hpp"

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
        << "每行输入：开火许可(0/1) yaw偏差(rad，右正左负) 绝对伸缩量(mm)\n"
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

[[noreturn]] void systemError(const char* operation) {
    throw std::system_error(errno, std::generic_category(), operation);
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
void handleLine(const std::string& line, dart_mcu::Communication& communication, std::atomic<bool>& running) {
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
    // 实际串口写入由公共模块完成，demo 只负责输入和显示。
    communication.send(command);
    dart_mcu::Frame frame{};
    dart_mcu::encode(command, frame);
    std::cout << "TX 25 字节已写入：" << hexBytes(frame.data(), frame.size()) << '\n';
}

// poll + read 同时支持键盘输入和管道输入。无换行的半行不会阻塞接收错误处理。
void inputLoop(dart_mcu::Communication& communication, std::atomic<bool>& running) {
    std::string pending;
    bool discard = false;
    std::cout << "输入命令（例如 1 2.1 30），q 退出：\n" << std::flush;
    while (running && communication.isRunning() && !interrupted) {
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
                if (!discard && !pending.empty()) handleLine(pending, communication, running);
                break;
            }
            for (ssize_t i = 0; i < count && running && communication.isRunning() && !interrupted; ++i) {
                if (bytes[i] == '\n') {
                    if (!discard) {
                        if (!pending.empty() && pending.back() == '\r') pending.pop_back();
                        handleLine(pending, communication, running);
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

        if (options.rx_log.empty()) options.rx_log = makeLogPath();
        ReceiveLog log(options.rx_log);
        log.line("会话开始，串口=" + options.port + "，波特率=" + std::to_string(options.baud));
        std::signal(SIGINT, onSignal);
        std::signal(SIGTERM, onSignal);
        std::signal(SIGCHLD, SIG_IGN);

        // start 内部打开串口并运行接收 loop，调用方无需另建串口接收线程。
        // log 先构造、communication 后构造，确保析构时线程先退出，日志仍有效。
        dart_mcu::Communication communication;
        communication.start(options.port, options.baud,
            [&log](const dart_mcu::ReceivedStatus& received) {
                const auto state = received.status.is_stationary;
                log.line("RX is_stationary=" + std::to_string(state) +
                         (state ? " 镖架静止" : " 镖架移动中"));
            },
            [&log](const std::string& message) { log.line(message); });
        openReceiveWindow(options.rx_log, options.new_window);

        std::atomic<bool> running{true};
        int result = 0;
        try { inputLoop(communication, running); }
        catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            result = 1;
        }
        communication.stop();
        const auto error = communication.lastError();
        if (!error.empty()) {
            std::cerr << "接收/发送错误：" << error << '\n';
            result = 1;
        }
        std::cout << "程序结束。接收记录保存在：" << options.rx_log << '\n';
        return result;
    } catch (const std::exception& error) {
        std::cerr << "启动失败：" << error.what() << '\n';
        return 1;
    }
}
