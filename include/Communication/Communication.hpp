#pragma once

#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace dart_mcu {

// 镖架双向协议：沿用 TorqueController 的帧结构和 CRC8 算法。
// 本模块将版本号定义为 0x03，电控接收端需要同步使用该版本及字段布局。
// 所有多字节字段采用小端序；浮点字段采用 IEEE 754 binary64。
// 视觉发送到电控的命令帧共 25 字节，偏移从 0 开始：
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

// 电控发送到视觉的状态帧共 9 字节；载荷仅包含一个 int32_t。
//   0..1    帧头：0x42 0x52
//   2       协议版本：0x03
//   3       有效载荷长度：4（0x04）
//   4..7    is_stationary：0 表示移动中，1 表示静止，小端序
//   8       CRC8：校验偏移 0..7 的全部 8 字节
// 收发共用帧头和版本，通过方向、载荷长度与解码接口区分。
constexpr std::size_t kReceivePayloadSize = 4;
constexpr std::size_t kReceiveFrameSize = 9;
using ReceiveFrame = std::array<std::uint8_t, kReceiveFrameSize>;

struct Status {
    // 只允许 0 或 1；表示电控报告的镖架运动状态。
    // 后续测定与弹道解算需同时满足：decode 返回 true 且该字段为 1。
    // 默认值 0 不放行后续步骤；解码失败时应视为状态未知。
    std::int32_t is_stationary = 0;
};

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

// 计算 CRC8（8 位循环冗余校验），返回一个字节的校验值。
// 发送端先对约定范围内的字节计算 CRC，再把结果追加到帧末尾；
// 接收端对收到的同一范围重新计算，将结果与帧末尾的 CRC 字节比较。
// 结果不同则丢弃该帧；结果相同后还需检查字段取值和业务条件。
// CRC 用于检出数据错误，存在漏检可能，也不会自动修复损坏的内容。
// 本函数只计算校验值；追加校验字节、比较校验值由调用方完成。
//
// 【算法参数】与 TorqueController/src/communication/CRC.cpp 的查表算法等价。
// 1. 宽度为 8 位，初值为 0xFF：每计算一帧都从 0xFF 开始，帧间不累计。
// 2. 多项式为 x^8 + x^5 + x^4 + 1；省略最高次项后记为 0x31。
//    本实现逐位向右移，使用其位反转形式 0x8C（0x31 的 8 位倒序）。
// 3. 输入、输出采用反射约定（RefIn=true、RefOut=true）。
//    下方的右移算法已经实现该约定，返回前无需额外翻转位序。
// 4. 最终异或值为 0x00（XorOut=0x00），计算结束后直接返回 crc。
//
// 【逐字节计算流程】
// 按缓冲区中的先后顺序处理字节；反射约定不会颠倒缓冲区的字节顺序。
// 先将当前 crc 与一个输入字节异或，然后重复 8 次以下操作：
//   - 检查移位前 crc 的最低位；
//   - crc 向右移动 1 位；
//   - 若刚才的最低位为 1，再与 0x8C 异或；为 0 则保留移位结果。
// 完成后继续处理下一个字节。查表版本把这 8 次操作预先算进表中。
//
// 【参数与校验范围】
// data：参与计算的首字节地址；size：参与计算的字节数。
// data 必须指向至少 size 字节的有效内存；仅 size 为 0 时可传空指针。
// size 为 0 时不读取内存，返回初值 0xFF。
// 本协议的计算范围包含帧头、版本、载荷长度和业务数据，排除末尾 CRC：
//   - 25 字节发送命令帧：计算偏移 0..23，共 24 字节。
//   -  9 字节接收状态帧：计算偏移 0..7，共 8 字节。
//
// 【示例：电控回传“镖架静止”】下列字节均使用十六进制表示。
// 待校验内容：42 52 03 04 01 00 00 00
//             帧头  版本 长度  状态 int32_t=1（小端序）
// crc 初值为 FF，依次处理上述 8 字节后的结果为：
//             CF EC A8 0C FD 89 10 9D
// 因此 crc8(data, 8) 返回 0x9D，完整帧为：
//             42 52 03 04 01 00 00 00 9D
// 接收端同样计算前 8 字节，与第 9 字节 0x9D 比较。
//
// 【独立校验向量】ASCII 字符串 "123456789" 的结果为 0x0B。
// 输入字节为 31 32 33 34 35 36 37 38 39，共 9 字节，不包含结尾的 '\0'。
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

// 解码视觉发给电控的 25 字节命令帧，供电控端适配或协议测试使用。
// 串口接收层负责处理分包、粘包，并将完整帧交给本函数。
// 失败时返回 false，同时将 output 重置为默认值（禁止开火）。
// 使用本地对齐变量恢复数值，避免直接将缓冲区转换为结构体指针。
inline bool decode_command(const std::uint8_t* data, std::size_t size,
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

// 解码电控回传的 9 字节状态帧，串口层需先完成分包与粘包处理。
// 返回 true 表示本帧格式、CRC 以及状态值均有效；移动和静止都属于有效状态。
// 返回 false 表示本帧无效，同时清除输出，避免残留的静止值被误用。
// 接收层还需记录有效状态的接收时间，并在通信超时时使状态失效。
// 此函数只解析当前帧，不保存历史状态，也不直接启动测定或弹道解算。
inline bool decode(const std::uint8_t* data, std::size_t size,
                   Status& output) {
    output = Status{};
    if (!data || size != kReceiveFrameSize ||
        data[0] != 0x42 || data[1] != 0x52 ||
        data[2] != kVersion || data[3] != kReceivePayloadSize ||
        crc8(data, kReceiveFrameSize - 1) != data[kReceiveFrameSize - 1]) {
        return false;
    }

    // 先用无符号值读取 4 字节；2、负数编码及其他未约定值全部拒绝。
    const auto state = get_le(data + 4, kReceivePayloadSize);
    if (state > 1) return false;
    output.is_stationary = static_cast<std::int32_t>(state);
    return true;
}
} // 命名空间 dart_mcu
