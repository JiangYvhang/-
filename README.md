# 用途：统一借口封装的图像操作
- 图像操作类统一接口：operate()方法：输入：初始化参数：img(图像，目前必须为二维图),par(字典，包含经常需调节的参数)
- 示例：RANSAC圆拟合，`[imagedR, resultsR] = RansacCircleDetector(image_biankuang, par).operate()`

# 结构：
- 图像操作抽象工厂
  - 图像检测工厂
    - RANSAC圆拟合
    - Hough圆检测
  - 画家类
  # 使用方法：
  main.py里面修改par字典调节参数
  
