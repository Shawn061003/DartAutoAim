cmake_minimum_required(VERSION 3.16)
project(dart_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(DART_BUILD_DEMARCATE "Build the standalone camera calibration tool" ON)

if(DART_BUILD_DEMARCATE)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d)
    add_executable(demarcate
        sub_module/demarcate.cpp
    )
    target_include_directories(demarcate PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(demarcate PRIVATE ${OpenCV_LIBS})
endif()

# 镖架 MCU 通信协议：提供带中文注释的帧编解码与 CRC8 校验。
# 使用方通过 target_link_libraries(<目标> PRIVATE dart::mcu_protocol) 引入。
# 头文件路径：Communication/Communication.hpp。
add_library(dart_mcu_protocol INTERFACE)
add_library(dart::mcu_protocol ALIAS dart_mcu_protocol)
target_include_directories(dart_mcu_protocol INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(dart_mcu_protocol INTERFACE cxx_std_17)

# 手动串口收发联调；自动设备发现和身份验证保留在 demo 中待实现。
option(DART_BUILD_CONTROL_DEMO "Build the manual MCU serial demo" ON)
if(DART_BUILD_CONTROL_DEMO)
    find_package(Threads REQUIRED)
    add_executable(dart_control_demo tests/dart_control_demo.cpp)
    target_link_libraries(dart_control_demo PRIVATE dart::mcu_protocol Threads::Threads)
    target_compile_options(dart_control_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(test_mcu_protocol tests/test_mcu_protocol.cpp)
    target_link_libraries(test_mcu_protocol PRIVATE dart::mcu_protocol)
    add_test(NAME dart_mcu_protocol COMMAND test_mcu_protocol)
endif()

# 伪终端模拟电控；不连接真实设备。Python 不存在时仍可构建 demo。
if(BUILD_TESTING AND DART_BUILD_CONTROL_DEMO)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME dart_control_demo_pty
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dart_control_demo.py
                $<TARGET_FILE:dart_control_demo>
        )
        set_tests_properties(dart_control_demo_pty PROPERTIES TIMEOUT 30)
    endif()
endif()
