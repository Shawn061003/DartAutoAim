cmake_minimum_required(VERSION 3.16)
project(dart_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d)

add_executable(demarcate demarcate.cpp)
target_include_directories(demarcate PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(demarcate PRIVATE ${OpenCV_LIBS})

# 镖架 MCU 通信协议：提供带中文注释的帧编解码与 CRC8 校验。
# 使用方通过 target_link_libraries(<目标> PRIVATE dart::mcu_protocol) 引入。
# 头文件路径：dart/communication/DartMcuProtocol.hpp。
add_library(dart_mcu_protocol INTERFACE)
add_library(dart::mcu_protocol ALIAS dart_mcu_protocol)
target_include_directories(dart_mcu_protocol INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(dart_mcu_protocol INTERFACE cxx_std_17)
