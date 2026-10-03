#pragma once

#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace dart_mcu {

// 镖架发送协议：沿用 TorqueController 的帧结构和 CRC8 算法。
// 本模块将版本号定义为 0x03，电控接收端需要同步使用该版本及字段布局。
// 所有多字节字段采用小端序；浮点字段采用 IEEE 754 binary64。
// 完整帧共 25 字节，偏移从 0 开始：
//   0       帧头第 1 字节，固定 0x42
//   1       帧头第 2 字节，固定 0x52
//   2       协议版本，固定 0x03
//   3       有效载荷长度，固定 20（十六进制 0x14）
//   4..7    开火许可 int32_t：0 禁止，1 允许
//   8..15   yaw 角度偏差 double：单位 rad，左正右负
//   16..23  目标伸缩量 double：单位 mm，相对标定零点的绝对目标，拉紧为正
//   24      CRC8：校验偏移 0..23 的全部 24 字节
// 帧尾仅包含 CRC8，没有额外的结束标志。
constexpr std::size_t kPayloadSize = 20;
constexpr std::size_t kFrameSize = 25;
constexpr std::uint8_t kVersion = 0x03;
using Frame = std::array<std::uint8_t, kFrameSize>;

static_assert(CHAR_BIT == 8, "Protocol requires 8-bit bytes");
static_assert(sizeof(double) == 8 &&
              std::numeric_limits<double>::is_iec559 &&
              std::numeric_limits<double>::digits == 53 &&
              std::numeric_limits<double>::max_exponent == 1024,
              "Protocol requires IEEE 754 binary64");

struct Command {
    std::int32_t fire_allowed = 0; // 开火许可：0 禁止，1 允许；实际开火指令由人工下达
    double yaw_error_rad = 0.0;    // 目标相对当前的角度偏差，单位 rad，左正右负
    double stretch_target_mm = 0.0; // 相对标定零点的绝对目标伸缩量，单位 mm，拉紧为正
};

// 与 TorqueController/src/communication/CRC.cpp 的查表算法等价。
// 多项式 0x31，右移实现使用反射形式 0x8C；初值 0xFF。
// 输入和输出采用反射约定，最终异或值为 0；结果直接放在帧末尾。
// data 必须指向至少 size 字节的有效内存；仅 size 为 0 时可传空指针。
// 校验向量：ASCII 字符串 123456789 的 CRC8 为 0x0B。
inline std::uint8_t crc8(const std::uint8_t* data, std::size_t size) {
    std::uint8_t crc = 0xFF;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (unsigned bit = 0; bit < 8; ++bit) {
            crc = static_cast<std::uint8_t>(
                (crc >> 1) ^ ((crc & 1u) ? 0x8Cu : 0u));
        }
    }
    return crc;
}

// 将整数的低 size 字节依次写入缓冲区，低字节在前。
// 内部辅助函数：size 范围为 1..8，调用方保证缓冲区容量。
inline void put_le(std::uint8_t* dst, std::uint64_t value,
                   std::size_t size) {
    for (std::size_t i = 0; i < size; ++i)
        dst[i] = static_cast<std::uint8_t>(value >> (8 * i));
}

// 按小端序恢复整数；size 范围为 1..8，调用方保证输入长度。
inline std::uint64_t get_le(const std::uint8_t* src, std::size_t size) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < size; ++i)
        value |= static_cast<std::uint64_t>(src[i]) << (8 * i);
    return value;
}

// 复制浮点数的原始位模式，再按小端序写入 8 字节；避免结构体填充。
inline void put_double(std::uint8_t* dst, double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    put_le(dst, bits, 8);
}

// 从 8 字节小端序数据恢复浮点数，避免未对齐地址上的 double 访问。
inline double get_double(const std::uint8_t* src) {
    const std::uint64_t bits = get_le(src, 8);
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// 检查许可值以及 NaN、正负无穷；机械行程范围由应用层标定参数限定。
inline bool valid(const Command& command) {
    return (command.fire_allowed == 0 || command.fire_allowed == 1) &&
           std::isfinite(command.yaw_error_rad) &&
           std::isfinite(command.stretch_target_mm);
}

// 将业务命令编码为完整的 25 字节帧，并自动填写 CRC8。
// 返回 true 才允许发送 output；失败时 output 全部清零。
// 调用方应在编码前检查实际机构允许的角度和伸缩行程。
// 本函数只负责封包；串口层负责打开设备及完整写出所有字节。
inline bool encode(const Command& command, Frame& output) {
    output.fill(0);
    if (!valid(command)) return false;
    output[0] = 0x42;
    output[1] = 0x52;
    output[2] = kVersion;
    output[3] = static_cast<std::uint8_t>(kPayloadSize);
    put_le(output.data() + 4,
           static_cast<std::uint32_t>(command.fire_allowed), 4);
    put_double(output.data() + 8, command.yaw_error_rad);
    put_double(output.data() + 16, command.stretch_target_mm);
    output[24] = crc8(output.data(), 24);
    return true;
}

// 校验并解码恰好一帧：检查长度、帧头、版本、载荷长度、CRC 和字段值。
// 串口接收层负责处理分包、粘包，并将完整帧交给本函数。
// 失败时返回 false，同时将 output 重置为默认值（禁止开火）。
// 使用本地对齐变量恢复数值，避免直接将缓冲区转换为结构体指针。
inline bool decode(const std::uint8_t* data, std::size_t size,
                   Command& output) {
    output = Command{};
    if (!data || size != kFrameSize ||
        data[0] != 0x42 || data[1] != 0x52 ||
        data[2] != kVersion || data[3] != kPayloadSize ||
        crc8(data, 24) != data[24]) return false;
    const auto permission = get_le(data + 4, 4);
    if (permission > 1) return false;
    Command candidate;
    candidate.fire_allowed = static_cast<std::int32_t>(permission);
    candidate.yaw_error_rad = get_double(data + 8);
    candidate.stretch_target_mm = get_double(data + 16);
    if (!valid(candidate)) return false;
    output = candidate;
    return true;
}

} // 命名空间 dart_mcu
