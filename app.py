import cv2
import numpy as np
import streamlit as st
from PIL import Image
from paddleocr import PaddleOCR

# 页面配置
st.set_page_config(page_title="汽车车牌识别系统", layout="centered")
st.title("🚗 汽车车牌识别系统")


# 缓存 OCR 模型，避免重复加载
@st.cache_resource
def load_ocr():
    return PaddleOCR(lang="ch")


with st.spinner("正在加载 OCR 模型，请稍候..."):
    ocr = load_ocr()

# 文件上传组件（支持手机拍照或相册选图）
uploaded_file = st.file_uploader("请选择或拍摄汽车图片", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # 读取图片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 左右分栏显示
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始图片")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    # 简单的颜色车牌定位逻辑（复用你代码中的核心）
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 60, 60]), np.array([140, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 60, 60]), np.array([40, 255, 255]))
    mask = cv2.bitwise_or(blue_mask, yellow_mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plate = None
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if 2.0 <= w / float(h) <= 6.0:
                plate = image[y:y + h, x:x + w]
                break

    if plate is not None:
        with col2:
            st.subheader("车牌区域")
            st.image(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), use_column_width=True)

        # 执行 OCR
        result = ocr.ocr(plate)
        text = ""
        if result:
            for res in result:
                if isinstance(res, dict) and 'rec_texts' in res:
                    text = "".join(res['rec_texts'])
                elif res:
                    for line in res:
                        try:
                            text += line[1][0]
                        except:
                            pass

        # 清理文本
        text = text.replace(" ", "").replace("\n", "").replace("-", "")

        st.markdown("---")
        st.subheader("📌 识别结果")
        if text:
            st.success(f"车牌号: **{text}**")
        else:
            st.warning("未能识别出车牌字符")
    else:
        st.error("没有检测到车牌，请尝试上传更清晰的正面汽车照片。")