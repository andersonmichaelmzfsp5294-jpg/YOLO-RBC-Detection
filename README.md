# 红细胞目标检测 (RBC Detection)

这是一个基于 YOLO11 的红细胞检测项目。

## 环境要求
- Python 3.x
- ultralytics

## 使用方法
1. **准备数据**: 运行 `prepare_data.py` 下载并转换 BCCD 数据集。
2. **训练模型**: 运行 `train_yolo.py` 开始训练。
3. **预测结果**: 运行 `predict_result.py` 生成检测结果和坐标文件。

## 结果
在测试集上达到了 mAP@0.5: 0.92+ 的效果。
计数。

 ##  
本项目已采用 Streamlit 框架完成云端部署。
访问链接：https://yolo-rbc-detection-xxxx.streamlit.app
系统状态：已上线 (Online)
功能说明：支持上传 JPG/PNG 图片，实时返回检测框及细胞计数。
运行截图：
(在这里贴一张你刚才截的全屏大图，证明你真的做出来了)
