from ultralytics import YOLO
import torch


def main():
    # ----------------- 配置区域 -----------------
    # 自动判断：如果有显卡且装好了驱动就用显卡(0)，否则用 CPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"当前使用的训练设备: {device} (0=GPU, cpu=CPU)")

    # 作业要求 batch_size=4 进行测试，训练时也可以用 4
    batch_size = 4
    # -------------------------------------------

    # 1. 加载模型
    # 首次运行会自动下载 yolo11n.pt，请保持网络通畅
    print("正在加载模型...")
    model = YOLO('yolo11n.pt')

    # 2. 开始训练
    print("开始训练...")
    results = model.train(
        data='bccd.yaml',  # 指向刚才生成的配置文件
        epochs=30,  # 训练30轮（作业演示足够了，想要更好效果可改为50-100）
        imgsz=640,  # 图片大小
        batch=batch_size,  # 批次大小
        device=device,  # 设备
        workers=0,  # Windows下设为0可以避免多线程报错
        project='BCCD_Project',  # 结果保存目录
        name='yolo11n_run',  # 实验名称
        plots=True  # 自动画图
    )

    # 3. 验证一下
    print("训练结束，正在验证...")
    metrics = model.val()
    print(f"mAP@50: {metrics.box.map50}")


if __name__ == '__main__':
    # Windows 下运行多进程必须加这一行保护
    import multiprocessing

    multiprocessing.freeze_support()

    main()