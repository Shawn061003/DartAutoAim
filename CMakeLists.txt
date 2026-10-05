cmake_minimum_required(VERSION 3.16)

# 可执行程序编译开关（ON 编译，OFF 跳过），也可通过 cmake -D<开关名>=ON/OFF 设置。
# 两个开关独立生效；仅构建串口程序时，设置 DART_BUILD_MAIN=OFF。
option(DART_BUILD_MAIN "Build the dart vision main program" ON)
option(DART_BUILD_CONTROL_DEMO "Build the manual MCU serial demo" ON)

project(dart_vision LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 公共模块：镖架 MCU 帧编解码与 CRC8 校验（仅头文件）。
# INTERFACE 向链接该目标的调用方传递头文件目录和 C++ 标准要求。
add_library(dart_mcu_protocol INTERFACE)
add_library(dart::mcu_protocol ALIAS dart_mcu_protocol)
target_include_directories(dart_mcu_protocol INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(dart_mcu_protocol INTERFACE cxx_std_17)

# 主程序：相机配置加载；采集、感知和解算模块实现后在此显式添加源文件。
if(DART_BUILD_MAIN)
    # FileStorage 与 Mat 均由 OpenCV core 提供，依赖随主程序开关启用。
    find_package(OpenCV REQUIRED COMPONENTS core)

    # 配置模块提供 dart::config 别名，供调用方通过 target_link_libraries 引用。
    # PUBLIC 使调用方同时获得配置头文件所需的 OpenCV 头文件路径及链接依赖。
    add_library(dart_config src/common/DartConfig.cpp)
    add_library(dart::config ALIAS dart_config)
    target_include_directories(dart_config PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(dart_config PUBLIC ${OpenCV_LIBS})
    target_compile_features(dart_config PUBLIC cxx_std_17)

    add_executable(dart_vision src/main.cpp)
    target_link_libraries(dart_vision PRIVATE dart::config dart::mcu_protocol)

    # 支持以构建目录为工作目录运行；源配置变化后 CMake 会同步副本。
    configure_file(config/dart.yaml config/dart.yaml COPYONLY)
endif()

# 串口通信程序：手动指定设备，与电控联调。
if(DART_BUILD_CONTROL_DEMO)
    find_package(Threads REQUIRED)
    add_executable(dart_control_demo tests/dart_control_demo.cpp)
    target_link_libraries(dart_control_demo PRIVATE dart::mcu_protocol Threads::Threads)
    target_compile_options(dart_control_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()
