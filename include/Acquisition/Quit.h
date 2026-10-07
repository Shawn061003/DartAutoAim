#pragma once

#include <csignal>

// 在相机运行期间接管 SIGINT/SIGTERM；作用域结束时恢复原有信号处理方式。
// 同一时刻只使用一个实例，并在 CameraInput 之前构造，使相机先完成析构清理。
class CameraStopSignals {
public:
    // 清空退出标记并注册信号处理；注册失败抛异常，恢复已修改的处理方式。
    CameraStopSignals();
    ~CameraStopSignals() noexcept;

    CameraStopSignals(const CameraStopSignals&) = delete;
    CameraStopSignals& operator=(const CameraStopSignals&) = delete;

    // 主循环查询退出请求；线程回收和相机关闭仍在正常控制流程中执行。
    bool stopRequested() const noexcept;

private:
    using Handler = void (*)(int);
    Handler previousInt_ = SIG_DFL;
    Handler previousTerm_ = SIG_DFL;
};
