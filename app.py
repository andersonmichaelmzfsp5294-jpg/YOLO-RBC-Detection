import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# ================= 页面基础设置 =================
st.set_page_config(page_title="红细胞智能检测系统", page_icon="🩸")
st.title("🩸 红细胞(RBC) 智能计数与检测系统")
st.markdown("### 基于 YOLO11 深度学习模型")

# ================= 侧边栏：设置 =================
st.sidebar.header("配置面板")

# 1. 设置置信度阈值 (Conf)
conf_threshold = st.sidebar.slider("置信度阈值 (Confidence)", 0.0, 1.0, 0.25, 0.05)
st.sidebar.info(f"当前阈值: {conf_threshold} (低于此分数的框会被过滤)")

# 2. 模型路径 (请根据你的实际情况修改路径)
MODEL_PATH = 'best.pt'


# ================= 加载模型 =================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"找不到模型文件！请检查路径: {MODEL_PATH}")
        return None
    return YOLO(MODEL_PATH)


model = load_model()

# ================= 主界面：上传与检测 =================
uploaded_file = st.file_uploader("请上传一张显微镜图片 (.jpg, .png)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None and model is not None:
    # 1. 打开图片
    image = Image.open(uploaded_file)

    # 显示原图
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="原始图片", use_container_width=True)

    # 2. 开始检测
    with st.spinner('正在分析细胞...'):
        # 运行推理
        # classes=[1] 代表只检测红细胞(RBC)，如果你想检测所有，去掉这个参数
        results = model.predict(image, conf=conf_threshold, classes=[1])

        # 获取画了框的图
        # YOLO plot() 返回的是 BGR 格式的 numpy 数组
        res_plotted = results[0].plot()

        # 获取红细胞数量
        rbc_count = len(results[0].boxes)

    # 3. 显示结果
    with col2:
        # channels="BGR" 很重要，否则图片颜色会发蓝
        st.image(res_plotted, caption="检测结果", channels="BGR", use_container_width=True)

    # 4. 显示统计数据
    st.success("检测完成！")
    st.metric(label="红细胞 (RBC) 计数", value=f"{rbc_count} 个")

    # 5. 导出结果 (可选展示)
    with st.expander("查看详细坐标数据"):
        boxes = results[0].boxes.xywh.cpu().numpy()
        for i, box in enumerate(boxes):
            st.text(f"细胞 #{i + 1}: 中心X={box[0]:.1f}, 中心Y={box[1]:.1f}, 宽={box[2]:.1f}, 高={box[3]:.1f}")

else:
    st.info("👈 请在上方上传图片开始检测")