#include "Communication/Communication.hpp"
#include <iostream>
#include <stdexcept>

// 无硬件测试，在项目根目录编译：
// g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I include tests/test_mcu_protocol.cpp -o build/test_mcu_protocol
namespace {
void check(bool result, const char* message) {
    // Release 模式也执行检查。
    if (!result) throw std::runtime_error(message);
}
void rejected(const std::uint8_t* data, std::size_t size) {
    dart_mcu::Status status{1};
    check(!dart_mcu::decode(data, size, status), "无效帧被接受");
    check(status.is_stationary == 0, "失败后残留静止状态");
}
}
int main() {
    try {
        using namespace dart_mcu;
        // 固定字节样例供电控核对，避免仅用同一套编解码互相验证。
        const ReceiveFrame moving = {0x42,0x52,3,4,0,0,0,0,0x12};
        const ReceiveFrame stationary = {0x42,0x52,3,4,1,0,0,0,0x9D};
        Status status{1};
        check(decode(moving.data(), moving.size(), status), "移动帧解码失败");
        check(status.is_stationary == 0, "移动状态错误");
        check(decode(stationary.data(), stationary.size(), status), "静止帧解码失败");
        check(status.is_stationary == 1, "静止状态错误");
        check(decode(moving.data(), moving.size(), status), "状态切换失败");
        check(status.is_stationary == 0, "移动帧未覆盖静止值");
        const std::uint8_t sample[] = {'1','2','3','4','5','6','7','8','9'};
        check(crc8(sample, sizeof(sample)) == 0x0B, "CRC 样例错误");
        rejected(nullptr, kReceiveFrameSize);
        for (std::size_t size = 0; size < moving.size(); ++size)
            rejected(moving.data(), size);
        const std::array<std::uint8_t, 10> extra = {0x42,0x52,3,4,1,0,0,0,0x9D,0};
        rejected(extra.data(), extra.size());
        // 即使 CRC 正确，也拒绝错误帧头、版本、载荷长度。
        for (std::size_t i = 0; i < 4; ++i) {
            auto bad = stationary;
            bad[i] ^= 1;
            bad.back() = crc8(bad.data(), bad.size() - 1);
            rejected(bad.data(), bad.size());
        }
        // 检查完整四字节状态，拒绝 2、高字节非零及负数编码。
        for (auto value : {2u, 256u, 0xFFFFFFFFu}) {
            auto bad = stationary;
            put_le(bad.data() + 4, value, 4);
            bad.back() = crc8(bad.data(), bad.size() - 1);
            rejected(bad.data(), bad.size());
        }
        // 两个合法帧所有单比特翻转，共 144 种破坏。
        for (const auto& frame : {moving, stationary})
            for (std::size_t i = 0; i < frame.size(); ++i)
                for (unsigned bit = 0; bit < 8; ++bit) {
                    auto bad = frame;
                    bad[i] ^= (1u << bit);
                    rejected(bad.data(), bad.size());
                }
        // 校验原发送帧布局，并拒绝将发送帧当作接收帧。
        Frame tx{};
        check(encode(Command{1, 0.125, 30.0}, tx), "命令编码失败");
        const Frame expected = {0x42,0x52,3,0x14,1,0,0,0,0,0,0,0,0,0,0xC0,0x3F,0,0,0,0,0,0,0x3E,0x40,0xED};
        check(tx == expected, "发送格式发生变化");
        rejected(tx.data(), tx.size());
        Command command;
        check(decode_command(tx.data(), tx.size(), command), "命令解码失败");
        check(command.fire_allowed == 1 && command.yaw_error_rad == 0.125 &&
              command.stretch_target_mm == 30.0, "命令字段错误");
        std::cout << "协议测试通过：状态解码、非法帧、144 种单比特错误、发送格式回归。\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}