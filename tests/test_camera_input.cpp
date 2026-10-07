#include "Acquisition/CameraInput.h"
#include "Acquisition/AcquireFrame.h"
#include <MvCameraControl.h>
#include <MvErrorDefine.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

// SDK 替身只存在于本测试可执行文件；生产目标链接真实 MVS 库。
namespace fake {
struct Device {
    unsigned int ip;
    bool opened = false;
    bool grabbing = false;
    bool leased = false;
    unsigned int sequence = 0;
    std::array<unsigned char, 12> pixels{};
};
std::atomic<int> handles{0}, leases{0}, gets{0}, frees{0}, activeGets{0};
std::atomic<int> initializations{0}, finalizations{0}, violations{0};
bool failRightOpen = false;
bool failConvert = false;
bool timeout = false;
bool connected = true;
bool wrongSize = false;
bool truncated = false;
bool lostPacket = false;
bool rgb = false;

void reset()
{
    if (handles != 0 || leases != 0 || activeGets != 0 || violations != 0 ||
        initializations != finalizations)
        throw std::runtime_error("SDK resources were not released correctly");
    gets = frees = 0;
    failRightOpen = failConvert = timeout = wrongSize = truncated = lostPacket = rgb = false;
    connected = true;
}
} // namespace fake

extern "C" {
int MV_CC_Initialize() { ++fake::initializations; return MV_OK; }
int MV_CC_Finalize()
{
    if (fake::handles != 0 || fake::leases != 0) ++fake::violations;
    ++fake::finalizations;
    return MV_OK;
}
int MV_CC_CreateHandle(void** handle, const MV_CC_DEVICE_INFO* info)
{
    if (info->nTLayerType != MV_GIGE_DEVICE || info->SpecialInfo.stGigEInfo.nNetExport != 0xc0a80101)
        return static_cast<int>(MV_E_PARAMETER);
    *handle = new fake::Device{info->SpecialInfo.stGigEInfo.nCurrentIp};
    ++fake::handles;
    return MV_OK;
}
int MV_CC_OpenDevice(void* handle, unsigned int, unsigned short)
{
    auto& d = *static_cast<fake::Device*>(handle);
    if (fake::failRightOpen && d.ip == 0xc0a80103) return static_cast<int>(MV_E_ACCESS_DENIED);
    d.opened = true;
    return MV_OK;
}
int MV_CC_CloseDevice(void* handle)
{
    auto& d = *static_cast<fake::Device*>(handle);
    if (d.grabbing || d.leased) ++fake::violations;
    d.opened = false;
    return MV_OK;
}
int MV_CC_DestroyHandle(void* handle)
{
    auto* d = static_cast<fake::Device*>(handle);
    if (d->opened || d->grabbing || d->leased) ++fake::violations;
    delete d;
    --fake::handles;
    return MV_OK;
}
int MV_CC_GetOptimalPacketSize(void*) { return 1500; }
int MV_CC_SetIntValueEx(void*, const char*, int64_t) { return MV_OK; }
int MV_CC_SetEnumValueByString(void*, const char*, const char*) { return MV_OK; }
int MV_CC_GetFloatValue(void*, const char*, MVCC_FLOATVALUE* value)
{
    value->fMin = 0;
    value->fMax = 10000;
    return MV_OK;
}
int MV_CC_SetFloatValue(void*, const char*, float) { return MV_OK; }
int MV_CC_GetIntValueEx(void*, const char* name, MVCC_INTVALUE_EX* value)
{
    value->nCurValue = std::strncmp(name, "Offset", 6) == 0 ? 0 : (fake::wrongSize ? 3 : 2);
    return MV_OK;
}
int MV_CC_SetImageNodeNum(void*, unsigned int) { return MV_OK; }
int MV_CC_SetGrabStrategy(void*, MV_GRAB_STRATEGY) { return MV_OK; }
int MV_CC_StartGrabbing(void* handle)
{
    auto& d = *static_cast<fake::Device*>(handle);
    if (!d.opened || d.grabbing) ++fake::violations;
    d.grabbing = true;
    return MV_OK;
}
int MV_CC_StopGrabbing(void* handle)
{
    auto& d = *static_cast<fake::Device*>(handle);
    if (d.leased || fake::activeGets != 0) ++fake::violations;
    d.grabbing = false;
    return MV_OK;
}
bool MV_CC_IsDeviceConnected(void*) { return fake::connected; }
int MV_CC_GetImageBuffer(void* handle, MV_FRAME_OUT* frame, unsigned int timeoutMs)
{
    auto& d = *static_cast<fake::Device*>(handle);
    ++fake::activeGets;
    // 模拟 SDK 的有限等待，检查停止是否等待取图结束后才销毁设备。
    std::this_thread::sleep_for(std::chrono::milliseconds(std::min(timeoutMs, 2u)));
    --fake::activeGets;
    if (!d.grabbing || d.leased) ++fake::violations;
    if (fake::timeout) return static_cast<int>(MV_E_NODATA);
    for (std::size_t i = 0; i < d.pixels.size(); i += 3) {
        d.pixels[i] = 11; d.pixels[i + 1] = 22; d.pixels[i + 2] = 33;
    }
    d.leased = true;
    frame->pBufAddr = d.pixels.data();
    frame->stFrameInfo.nWidth = 2;
    frame->stFrameInfo.nHeight = 2;
    frame->stFrameInfo.nFrameLen = fake::truncated ? 2 : 12;
    frame->stFrameInfo.nFrameNum = ++d.sequence;
    frame->stFrameInfo.enPixelType = fake::rgb ? PixelType_Gvsp_RGB8_Packed : PixelType_Gvsp_BGR8_Packed;
    frame->stFrameInfo.nLostPacket = fake::lostPacket ? 1 : 0;
    ++fake::leases;
    ++fake::gets;
    return MV_OK;
}
int MV_CC_FreeImageBuffer(void* handle, MV_FRAME_OUT*)
{
    auto& d = *static_cast<fake::Device*>(handle);
    if (!d.leased) ++fake::violations;
    d.leased = false;
    // 释放即覆盖原始像素，输出帧必须仍保留完整图像。
    d.pixels.fill(0);
    --fake::leases;
    ++fake::frees;
    return MV_OK;
}
int MV_CC_ConvertPixelTypeEx(void*, MV_CC_PIXEL_CONVERT_PARAM_EX* p)
{
    if (fake::failConvert) return static_cast<int>(MV_E_PARAMETER);
    if (p->enDstPixelType != PixelType_Gvsp_BGR8_Packed || p->nDstBufferSize < 12)
        return static_cast<int>(MV_E_PARAMETER);
    for (unsigned int i = 0; i < 12; i += 3) {
        p->pDstBuffer[i] = p->pSrcData[i + 2];
        p->pDstBuffer[i + 1] = p->pSrcData[i + 1];
        p->pDstBuffer[i + 2] = p->pSrcData[i];
    }
    p->nDstLen = 12;
    return MV_OK;
}
} // extern "C"

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
template<class F> void requireThrow(F action, const char* message)
{
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}
CameraParams params(bool right = false)
{
    CameraParams p;
    p.deviceIp = right ? "192.168.1.3" : "192.168.1.2";
    p.netIp = "192.168.1.1";
    p.exposure = 1500.5F;
    p.gain = 0.5F;
    p.width = p.height = 2;
    return p;
}
std::optional<StereoFrame> waitPair(CameraInput& input)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        auto pair = input.frameQueue().GetFrame(50);
        if (pair) return pair;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return std::nullopt;
}
} // namespace

int main()
{
    try {
        // 参数错误在任何设备调用前被拒绝。
        fake::reset();
        requireThrow([] { CameraDevice d(params(), CameraSide::Unknown); }, "Unknown side accepted");
        auto invalid = params(); invalid.deviceIp = "bad-ip";
        requireThrow([&] { CameraDevice d(invalid, CameraSide::Left); }, "Invalid IP accepted");
        invalid = params(); invalid.exposure = 0;
        requireThrow([&] { CameraDevice d(invalid, CameraSide::Left); }, "Placeholder exposure accepted");
        requireThrow([] { CameraInput i(params(), params(), 4); }, "Duplicate devices accepted");
        requireThrow([] { CameraInput i(params(), params(true), 4, 0); }, "Zero timeout accepted");
        requireThrow([] { CameraInput i(params(), params(true), 4, UINT32_MAX); }, "Infinite timeout accepted");
        require(fake::initializations == 0, "Validation opened SDK");

        // 直接 BGR 路径：输出不依赖已归还的 SDK 缓存；超时返回空结果。
        {
            CameraDevice d(params(), CameraSide::Left);
            requireThrow([&] { d.GrabImage(10); }, "Grab before start succeeded");
            d.StartGrabbing();
            requireThrow([&] { d.StartGrabbing(); }, "Duplicate start accepted");
            auto f = d.GrabImage(10);
            require(f && f->camera_side == CameraSide::Left && f->frame_id == 1, "Frame metadata lost");
            require(f->image.at<cv::Vec3b>(0, 0) == cv::Vec3b(11, 22, 33), "SDK buffer alias escaped");
            require(fake::leases == 0 && fake::gets == fake::frees, "Frame was not freed");
            auto next = d.GrabImage(10);
            require(next->timestamp_ms >= f->timestamp_ms, "Nonmonotonic receive time");
            fake::timeout = true;
            require(!d.GrabImage(10), "Timeout returned stale image");
            fake::connected = false;
            requireThrow([&] { d.GrabImage(10); }, "Disconnected camera returned ordinary timeout");
            d.StopGrabbing(); d.StopGrabbing();
        }
        fake::reset();

        // 转换成功及失败、截断、丢包路径均须准确归还缓存。
        {
            CameraDevice d(params(), CameraSide::Right);
            d.StartGrabbing();
            fake::rgb = true;
            auto f = d.GrabImage(10);
            require(f->image.at<cv::Vec3b>(0, 0) == cv::Vec3b(33, 22, 11), "RGB conversion failed");
            fake::failConvert = true;
            requireThrow([&] { d.GrabImage(10); }, "Conversion failure ignored");
            fake::failConvert = fake::rgb = false;
            fake::truncated = true;
            requireThrow([&] { d.GrabImage(10); }, "Truncated BGR frame accepted");
            fake::truncated = false; fake::lostPacket = true;
            requireThrow([&] { d.GrabImage(10); }, "Incomplete frame accepted");
            require(fake::gets == fake::frees && fake::leases == 0, "Error path leaked image buffer");
        }
        fake::reset();

        // 右侧打开失败时必须回滚左侧，参数设置失败也不能遗留句柄。
        {
            CameraInput input(params(), params(true), 2, 10);
            fake::failRightOpen = true;
            requireThrow([&] { input.StartCamera(); }, "Right-open failure ignored");
            require(fake::handles == 0, "Partial startup leaked device");
            fake::failRightOpen = false;
            fake::wrongSize = true;
            requireThrow([&] { input.StartCamera(); }, "Calibration size mismatch accepted");
            require(fake::handles == 0, "Configuration failure leaked device");
        }
        fake::reset();

        // 双线程可以产出左右帧；停止可重复，重启保留同一个队列且没有旧帧。
        {
            CameraInput input(params(), params(true), 2, 10);
            auto* queue = &input.frameQueue();
            input.StartCamera();
            requireThrow([&] { input.StartCamera(); }, "Duplicate input start accepted");
            auto pair = waitPair(input);
            require(pair && pair->left.camera_side == CameraSide::Left &&
                    pair->right.camera_side == CameraSide::Right, "Stereo capture failed");
            input.StopCamera(); input.StopCamera();
            require(queue->leftSize() == 0 && queue->rightSize() == 0, "Stop retained old frames");
            require(fake::handles == 0 && fake::leases == 0, "Stop leaked SDK resources");
            fake::timeout = true;
            input.StartCamera();
            require(&input.frameQueue() == queue, "Restart invalidated queue reference");
            require(!input.frameQueue().GetFrame(), "Restart returned stale frame");
            // 析构期间有正在等待的取图调用，仍须先回收线程再销毁句柄。
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        fake::reset();

        // 工作线程中的转换错误不能触发 std::terminate，停止后仍无资源泄漏。
        {
            fake::rgb = fake::failConvert = true;
            CameraInput input(params(), params(true), 2, 10);
            input.StartCamera();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (fake::frees == 0 && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            input.StopCamera();
            require(fake::frees > 0 && fake::gets == fake::frees, "Worker error was not exercised/cleaned");
        }
        fake::reset();
        std::cout << "Camera lifecycle, buffers, failure rollback and threading: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
