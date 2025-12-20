"""
demo_backend/streamlit_app.py

Fastest all-in-one demo (UI + inference).
Run:
  pip install -r demo_backend/requirements_demo.txt
  streamlit run demo_backend/streamlit_app.py
"""

import os

import streamlit as st
from PIL import Image

from demo_backend.predictor import Predictor, PredictorConfig


def list_models(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        return []
    ms = [fn for fn in os.listdir(ckpt_dir) if fn.endswith(".pth") or fn.endswith(".pt")]
    ms.sort()
    return ms


st.set_page_config(page_title="CycleGAN Style Transfer Demo", layout="wide")

st.title("CycleGAN 风格迁移 Demo")

ckpt_dir = st.sidebar.text_input("Checkpoint 目录", value="demo_backend/checkpoints")
models = list_models(ckpt_dir)

if not models:
    st.warning("未在 checkpoint 目录中找到 .pth/.pt 文件。请把权重放到 demo_backend/checkpoints/ 下。")
    st.stop()

model_name = st.sidebar.selectbox("选择模型", options=models, index=0)
direction = st.sidebar.selectbox("转换方向", options=["AtoB", "BtoA"], index=0)
size = st.sidebar.slider("输入尺寸 (Resize to square)", min_value=128, max_value=1024, value=256, step=32)
strength = st.sidebar.slider("风格强度 (线性混合)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
device = st.sidebar.selectbox("Device", options=["cuda", "cpu"], index=0)
amp = st.sidebar.checkbox("AMP (CUDA)", value=True)
channels_last = st.sidebar.checkbox("channels_last (CUDA)", value=True)

uploaded = st.file_uploader("上传图片 (jpg/png)", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info("请先上传一张图片。")
    st.stop()

img = Image.open(uploaded)
st.image(img, caption="Input", use_container_width=True)

ckpt_path = os.path.join(ckpt_dir, model_name)
pred = Predictor(PredictorConfig(
    checkpoint=ckpt_path,
    device=device,
    amp=amp,
    amp_dtype="auto",
    channels_last=channels_last,
    default_size=size,
))

if st.button("开始转换", type="primary"):
    with st.spinner("推理中..."):
        out = pred.predict(img, direction=direction, size=size, strength=strength)

    c1, c2 = st.columns(2)
    with c1:
        st.image(img, caption="原图", use_container_width=True)
    with c2:
        st.image(out, caption="风格迁移结果", use_container_width=True)

    # Download
    import io
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    st.download_button("下载结果 PNG", data=buf.getvalue(), file_name="output.png", mime="image/png")
