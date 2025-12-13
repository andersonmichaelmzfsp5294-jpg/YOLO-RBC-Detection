import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET
import yaml
from git import Repo  # 需要 pip install gitpython

# ================= 配置参数 =================
DATASET_URL = "https://github.com/experiencor/BCCD_Dataset.git"
RAW_DATA_DIR = "./BCCD_Dataset_Raw"
OUTPUT_DIR = "./datasets/BCCD_YOLO"
CLASSES = ["WBC", "RBC", "Platelets"]  # 类别顺序很重要，ID分别为 0, 1, 2


# ===========================================

def convert_box(size, box):
    """将 VOC XML 坐标转换为 YOLO 归一化坐标"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def prepare_dataset():
    # 1. 下载数据集
    if not os.path.exists(RAW_DATA_DIR):
        print(f"正在克隆数据集从 {DATASET_URL} ...")
        Repo.clone_from(DATASET_URL, RAW_DATA_DIR)
    else:
        print("原始数据集已存在，跳过下载。")

    # 2. 创建输出目录结构
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    # 3. 读取所有 XML 文件
    xml_files = glob.glob(os.path.join(RAW_DATA_DIR, 'BCCD', 'Annotations', '*.xml'))
    random.seed(42)
    random.shuffle(xml_files)

    # 划分数据集 (80% 训练, 10% 验证, 10% 测试)
    train_split = int(len(xml_files) * 0.8)
    val_split = int(len(xml_files) * 0.9)

    train_files = xml_files[:train_split]
    val_files = xml_files[train_split:val_split]
    test_files = xml_files[val_split:]

    splits = {'train': train_files, 'val': val_files, 'test': test_files}

    print("开始格式转换...")
    for split, files in splits.items():
        for xml_file in files:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # 获取图片尺寸
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            # 对应的图片路径
            file_id = os.path.basename(xml_file).replace('.xml', '')
            img_path = os.path.join(RAW_DATA_DIR, 'BCCD', 'JPEGImages', file_id + '.jpg')

            if not os.path.exists(img_path):
                continue  # 如果图片不存在则跳过

            # 转换标签
            label_lines = []
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in CLASSES or int(difficult) == 1:
                    continue

                cls_id = CLASSES.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_box((w, h), b)
                label_lines.append(f"{cls_id} {bb[0]} {bb[1]} {bb[2]} {bb[3]}")

            # 只有当图片有对应的标签时才保存
            if label_lines:
                # 复制图片
                shutil.copy(img_path, os.path.join(OUTPUT_DIR, 'images', split, file_id + '.jpg'))
                # 保存 TXT
                with open(os.path.join(OUTPUT_DIR, 'labels', split, file_id + '.txt'), 'w') as f:
                    f.write('\n'.join(label_lines))

    # 4. 生成 data.yaml
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(CLASSES)}
    }

    with open('bccd.yaml', 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"数据处理完成！数据集位于: {OUTPUT_DIR}")
    print("配置文件已生成: bccd.yaml")


if __name__ == '__main__':
    prepare_dataset()