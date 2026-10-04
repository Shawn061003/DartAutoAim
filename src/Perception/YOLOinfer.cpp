// 视觉YOLO粗检测，主要用于确定引导灯位置以及
// 为后续OpenCV传统视觉提供ROI
// 载入weight中的.pt 或 .onnx文件进行推理

/*
输入：AcquireFrame.cpp中输出的两张图片
输出：两张图片上YOLO模型检测结果
*/