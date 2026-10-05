#include "common/DartConfig.h"

#include <exception>
#include <iostream>

// 当前入口完成配置加载并输出图像尺寸，后续在此接入自瞄各模块。
// 用法：dart_vision [配置路径]；省略参数时读取工作目录下的 config/dart.yaml。
int main(int argc, char* argv[])
{
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [config/dart.yaml]\n";
        return 1;
    }

    try {
        const DartCongfig config(argc == 2 ? argv[1] : "config/dart.yaml");
        std::cout << "Camera configuration loaded.\n"
                  << "Left: " << config.leftCamera.width << " x "
                  << config.leftCamera.height << '\n'
                  << "Right: " << config.rightCamera.width << " x "
                  << config.rightCamera.height << '\n';
        // TODO: 补齐 YAML 中的设备参数后，连接并启动左右相机。
        // TODO: 获取时间戳相同（相近）的两帧，接入感知、解算与准入准出门控。
        std::cout << "TODO: camera acquisition and aiming pipeline are not connected yet.\n";
        return 0;
    } catch (const std::exception& error) {
        // 配置加载失败时输出诊断并返回非零退出码，便于启动脚本检查结果。
        std::cerr << error.what() << '\n';
        return 1;
    }
}
