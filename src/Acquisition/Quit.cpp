#include "Acquisition/Quit.h"

#include <stdexcept>

namespace {
// 信号处理函数只修改退出标记，避免在信号上下文中调用 SDK 或等待线程。
volatile std::sig_atomic_t cameraStopRequested = 0;

void requestCameraStop(int) { cameraStopRequested = 1; }
} // namespace

CameraStopSignals::CameraStopSignals()
{
    cameraStopRequested = 0;
    previousInt_ = std::signal(SIGINT, requestCameraStop);
    if (previousInt_ == SIG_ERR)
        throw std::runtime_error("Could not install SIGINT handler");

    previousTerm_ = std::signal(SIGTERM, requestCameraStop);
    if (previousTerm_ == SIG_ERR) {
        // 第二个信号注册失败时，先恢复已接管的 SIGINT，再报告错误。
        std::signal(SIGINT, previousInt_);
        throw std::runtime_error("Could not install SIGTERM handler");
    }
}

CameraStopSignals::~CameraStopSignals() noexcept
{
    // 正常返回和异常退出都会经过此处，避免影响后续代码的信号处理。
    std::signal(SIGINT, previousInt_);
    std::signal(SIGTERM, previousTerm_);
}

bool CameraStopSignals::stopRequested() const noexcept
{
    return cameraStopRequested != 0;
}
