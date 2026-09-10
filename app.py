import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr

st.set_page_config(page_title="汽车车牌识别系统", layout="centered")
st.title("🚗 汽车车牌识别系统")

@st.cache_resource
def load_reader():
    return easyocr.Reader(['ch_sim', 'en'], gpu=False)

with st.spinner("正在加载识别模型，请稍候..."):
    reader = load_reader()

uploaded_file = st.file_uploader("请拍照或上传车牌图片", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始图像")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    with st.spinner("正在定位并识别车牌..."):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb_image)

        plate_texts = []
        for bbox, text, score in results:
            cleaned_text = text.replace(" ", "").upper()
            if len(cleaned_text) >= 6:
                plate_texts.append(f"{cleaned_text} (置信度: {score:.2f})")

    with col2:
        st.subheader("识别结果")
        if plate_texts:
            for pt in plate_texts:
                st.success(f"识别车牌: **{pt}**")
        else:
            st.warning("未检测到有效车牌字符，请尝试调整拍摄角度。")
