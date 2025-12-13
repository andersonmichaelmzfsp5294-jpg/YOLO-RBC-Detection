import os
import time
from ultralytics import YOLO


def main():
    # ================= 配置路径 =================
    # 1. 模型路径 (指向你刚才训练生成的 best.pt)
    model_path = 'BCCD_Project/yolo11n_run/weights/best.pt'

    # 2. 测试集图片路径
    source_dir = 'datasets/BCCD_YOLO/images/test'

    # 3. 结果保存目录
    save_dir = 'Submission_Result'
    # ===========================================

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}，请检查路径！")
        return

    # 加载模型
    print(f"正在加载模型: {model_path} ...")
    model = YOLO(model_path)

    # 创建保存目录
    os.makedirs(f"{save_dir}/images", exist_ok=True)
    os.makedirs(f"{save_dir}/labels_center", exist_ok=True)

    print("开始进行推理和生成坐标文件...")

    # ---------------------------------------------------------
    # 任务 1 & 2: 生成可视化图 + 导出红细胞中心点坐标 (.txt)
    # ---------------------------------------------------------
    # 这里的 stream=True 是为了防止内存溢出，save=True 会自动保存可视化图
    # classes=[1] 表示只检测红细胞 (RBC)，因为在 prepare_data 中 RBC 是第2个，索引为1
    # 如果你想检测所有细胞，去掉 classes=[1] 即可
    results = model.predict(source=source_dir, save=True, project=save_dir, name='images', classes=[1], conf=0.25)

    for result in results:
        # 获取文件名
        img_name = os.path.basename(result.path)
        txt_name = img_name.replace('.jpg', '.txt')

        # 准备写入坐标
        centers = []

        # result.boxes.xywh 包含: [中心x, 中心y, 宽, 高] (绝对像素坐标)
        # 变成 CPU numpy 格式方便处理
        boxes = result.boxes.xywh.cpu().numpy()

        for box in boxes:
            cx, cy, w, h = box
            # 保存格式：中心点X, 中心点Y
            # (你可以根据作业要求调整，例如保留几位小数)
            centers.append(f"{cx:.2f}, {cy:.2f}")

        # 写入 txt 文件
        with open(os.path.join(save_dir, 'labels_center', txt_name), 'w') as f:
            f.write("Center_X, Center_Y\n")  # 表头
            f.write('\n'.join(centers))

    print(f"\n[完成] 可视化结果已保存在: {save_dir}/images")
    print(f"[完成] 中心点坐标已保存在: {save_dir}/labels_center")

    # ---------------------------------------------------------
    # 任务 3: 测算 Batch_size=4 时的 FPS
    # ---------------------------------------------------------
    print("\n开始测试 FPS (Batch Size = 4)...")

    # 模拟一个 batch=4 的输入（随机拿4张测试图，如果不够就重复）
    import glob
    test_imgs = glob.glob(os.path.join(source_dir, '*.jpg'))
    if len(test_imgs) < 4:
        batch_imgs = test_imgs * 4
    else:
        batch_imgs = test_imgs[:4]

    batch_imgs = batch_imgs[:4]  # 确保是4张

    # 预热一次 (让 GPU/CPU 准备好，不计入时间)
    model(batch_imgs, verbose=False)

    # 开始计时
    start_time = time.time()
    # 运行 10 次取平均
    for _ in range(10):
        model(batch_imgs, verbose=False)
    end_time = time.time()

    total_time = end_time - start_time
    # 总帧数 = 10次 * 4张/次 = 40张
    fps = 40 / total_time

    print(f"设备信息: {model.device}")
    print(f"Batch_size=4 时的 FPS: {fps:.2f}")
    print("------------------------------------------------")


if __name__ == '__main__':
    main()