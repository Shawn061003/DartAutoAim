cmake_minimum_required(VERSION 3.16)

# 可执行程序编译开关（ON 编译，OFF 跳过），也可通过 cmake -D<开关名>=ON/OFF 设置。
# 开关独立生效；仅构建串口程序时，设置 DART_BUILD_MAIN=OFF、DART_BUILD_YOLO=OFF。
option(DART_BUILD_MAIN "Build the dart vision main program" ON)
option(DART_BUILD_CONTROL_DEMO "Build the manual MCU serial demo" ON)
option(DART_BUILD_YOLO "Build OpenVINO YOLO inference and tests" ON)
option(DART_BUILD_HSV_TUNER "Build YOLO ROI HSV tuner (requires DART_BUILD_YOLO)" ON)

# 主程序的图片检测链调用 YOLO；关闭该模块时须同时关闭主程序。
if(DART_BUILD_MAIN AND NOT DART_BUILD_YOLO)
    message(FATAL_ERROR "DART_BUILD_MAIN requires DART_BUILD_YOLO=ON")
endif()

project(dart_vision LANGUAGES CXX)
include(CTest)
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

# 主程序：从 YAML 选择输入模式，图片输入依次完成采集、YOLO 与引导灯检测。
if(DART_BUILD_MAIN)
    find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui)
    find_package(Threads REQUIRED)

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

    add_library(dart_acquisition
        src/Acquisition/AcquireFrame.cpp
        src/Acquisition/VideoInput.cpp
    )
    add_library(dart::acquisition ALIAS dart_acquisition)
    target_include_directories(dart_acquisition PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(dart_acquisition PUBLIC opencv_core Threads::Threads PRIVATE opencv_imgcodecs)
    target_compile_options(dart_acquisition PRIVATE -Wall -Wextra -Wpedantic)

    # 可视化独立于采集与感知模块，仅依赖 OpenCV。
    add_library(dart_visualization src/Visualization/Visualize.cpp)
    add_library(dart::visualization ALIAS dart_visualization)
    target_include_directories(dart_visualization PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(dart_visualization PUBLIC opencv_core PRIVATE opencv_imgproc opencv_highgui)
    target_compile_options(dart_visualization PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(dart_vision src/main.cpp)
    # 默认配置定位到源码目录；模型和媒体路径由 YAML 所在目录解析。
    target_compile_definitions(dart_vision PRIVATE DART_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(dart_vision PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(dart_vision PRIVATE dart::config dart::mcu_protocol dart::acquisition dart::visualization)

    if(BUILD_TESTING)
        add_executable(dart_config_test tests/test_dart_config.cpp)
        target_link_libraries(dart_config_test PRIVATE dart::config)
        add_test(NAME dart_config COMMAND dart_config_test
            ${CMAKE_CURRENT_SOURCE_DIR}/config/dart.yaml
            ${CMAKE_CURRENT_BINARY_DIR}/config_fixtures)

        add_executable(dart_guide_light_test
            tests/test_guide_light_detect.cpp src/Perception/GuideLightDetect.cpp)
        target_include_directories(dart_guide_light_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(dart_guide_light_test PRIVATE opencv_core opencv_imgproc)
        add_test(NAME guide_light COMMAND dart_guide_light_test)

        add_executable(dart_visualization_test tests/test_visualization.cpp)
        target_link_libraries(dart_visualization_test PRIVATE dart::visualization opencv_imgcodecs)
        target_compile_options(dart_visualization_test PRIVATE -Wall -Wextra -Wpedantic)
        add_test(NAME visualization COMMAND dart_visualization_test)

        add_executable(dart_video_input_test tests/test_video_input.cpp)
        target_link_libraries(dart_video_input_test PRIVATE dart::acquisition opencv_imgcodecs)
        target_compile_options(dart_video_input_test PRIVATE -Wall -Wextra -Wpedantic)
        add_test(NAME video_input COMMAND dart_video_input_test
            ${CMAKE_CURRENT_BINARY_DIR}/video_input_fixtures)
    endif()
    # 默认直接读取源目录 YAML，保持媒体与模型相对路径的含义。
endif()

# 串口通信程序：手动指定设备，与电控联调。
if(DART_BUILD_CONTROL_DEMO)
    find_package(Threads REQUIRED)
    add_executable(dart_control_demo tests/dart_control_demo.cpp)
    target_link_libraries(dart_control_demo PRIVATE dart::mcu_protocol Threads::Threads)
    target_compile_options(dart_control_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

# YOLO模块独立构建，模型路径和设备由调用方传入。
if(DART_BUILD_YOLO)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)

    add_library(dart_yolo src/Perception/YOLOinfer.cpp)
    add_library(dart::yolo ALIAS dart_yolo)
    target_include_directories(dart_yolo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(dart_yolo PUBLIC opencv_core PRIVATE opencv_imgproc opencv_dnn openvino::runtime)
    target_compile_options(dart_yolo PRIVATE -Wall -Wextra -Wpedantic)

    find_package(OpenCV REQUIRED COMPONENTS imgcodecs videoio)
    add_executable(dart_yolo_video_test tests/test_yolo_video.cpp)
    target_link_libraries(dart_yolo_video_test PRIVATE dart::yolo opencv_imgproc opencv_imgcodecs opencv_videoio)
    target_compile_options(dart_yolo_video_test PRIVATE -Wall -Wextra -Wpedantic)

    if(BUILD_TESTING)
        add_executable(dart_yolo_test tests/test_yolo_inference.cpp)
        target_link_libraries(dart_yolo_test PRIVATE dart::yolo openvino::runtime)
        target_compile_options(dart_yolo_test PRIVATE -Wall -Wextra -Wpedantic)
        add_test(NAME yolo_inference COMMAND dart_yolo_test ${CMAKE_CURRENT_BINARY_DIR}/yolo_fixtures)
    endif()

    if(DART_BUILD_MAIN)
        target_sources(dart_vision PRIVATE src/Perception/GuideLightDetect.cpp)
        target_link_libraries(dart_vision PRIVATE dart::yolo opencv_imgproc)
    endif()
endif()

# YOLO ROI HSV调参工具（随YOLO开关启用）：源码在tests，程序由构建目录输出，保存的数据在outputs/hsv_tuner。
if(DART_BUILD_HSV_TUNER AND DART_BUILD_YOLO)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
    add_executable(dart_hsv_tuner tests/test_hsv_tuner.cpp src/Perception/GuideLightDetect.cpp)
    target_include_directories(dart_hsv_tuner PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(dart_hsv_tuner PRIVATE dart::yolo opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui)
    target_compile_definitions(dart_hsv_tuner PRIVATE DART_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(dart_hsv_tuner PRIVATE -Wall -Wextra -Wpedantic)
endif()
