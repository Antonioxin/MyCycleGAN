# Demo Backend（网页/推理对接包）

这个文件夹用于**网页 Demo 对接**：把「训练代码」和「推理接口」隔离开，让负责网页的同学不需要理解 CBAM / CycleGAN 训练细节，也能快速做上传与展示页面。

---

## 1. 目录结构（交付包）

```
demo_backend/
  checkpoints/                 # 放 .pth 权重（建议用 Git LFS 或 Release 下载）
  assets/example_inputs/        # 可选：放几张示例输入图
  outputs/                      # 本地输出（建议 gitignore）
  predictor.py                  # ✅ 核心：Predictor 类（PIL in -> PIL out）
  infer_one.py                  # ✅ 单图推理 CLI
  server_fastapi.py             # ✅ FastAPI 服务（前后端分离）
  streamlit_app.py              # ✅ Streamlit 一体化 demo（最快出效果）
  requirements_demo.txt         # demo 依赖
```

---

## 2. 快速开始（先跑通单图推理）

### 2.1 准备权重
把权重放到：
- `demo_backend/checkpoints/epoch_200_cbam.pth`（示例）

> 注意：`.pth` 可能 > 100MB，**不建议直接 git push**。推荐用 Git LFS（见第 6 节）。

### 2.2 安装 demo 依赖
在项目根目录执行：

```bash
pip install -r demo_backend/requirements_demo.txt
```

（你们的 torch/torchvision 一般由 `environment.yml` 负责安装。）

### 2.3 单图推理（CLI）
```bash
python demo_backend/infer_one.py \
  --checkpoint demo_backend/checkpoints/epoch_200_cbam.pth \
  --input demo_backend/assets/example_inputs/1.jpg \
  --output demo_backend/outputs/out.png \
  --direction AtoB \
  --size 256 \
  --strength 1.0
```

---

## 3. 方案 A：Streamlit 一体化 Demo（最快）

```bash
streamlit run demo_backend/streamlit_app.py
```

特点：
- ✅ UI + 推理一套 Python 代码，最快演示、录视频
- ✅ 适合答辩现场离线演示（本地/服务器都行）

---

## 4. 方案 B：FastAPI 推理服务（前后端分离）

### 4.1 启动服务
```bash
python demo_backend/server_fastapi.py \
  --checkpoint-dir demo_backend/checkpoints \
  --device cuda
```

默认端口 `8000`。

### 4.2 联调测试（curl）
```bash
curl -F "image=@demo_backend/assets/example_inputs/1.jpg" \
     -F "direction=AtoB" \
     -F "size=256" \
     -F "strength=1.0" \
     http://127.0.0.1:8000/predict \
     --output out.png
```

### 4.3 关键接口
- `GET /health`：健康检查
- `GET /models`：列出 `checkpoints/` 下的权重文件
- `POST /predict`：multipart 上传图片并返回 png

---

## 5. 接口契约（网页同学只需要看这部分）

### 5.1 POST /predict 参数
- `image`：文件（jpg/png/webp）
- `direction`：`AtoB` 或 `BtoA`
- `size`：int，默认 256（会 resize 成正方形）
- `strength`：0~1（风格强度，当前用线性混合实现）
- `model`：可选，选择某个 checkpoint 文件名；不传用默认模型

### 5.2 返回
- 成功：直接返回 `image/png` 二进制
- 失败：返回 JSON `{ok:false, error:"...", detail:"..."}`

---

## 6. ⚠️ 权重文件如何同步到 GitHub（强烈建议 Git LFS）

如果 `.pth` > 100MB，GitHub 会拒绝推送。推荐：

```bash
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes
git add demo_backend/checkpoints/epoch_200_cbam.pth
git commit -m "Add demo backend and checkpoint (LFS)"
git push
```

组员 clone 后，需要：

```bash
git lfs install
git lfs pull
```

---

## 7. 常见问题

### 7.1 输出颜色/对比度不对？
原因通常是归一化不一致。这里固定使用：
- 输入：Normalize(mean=0.5,std=0.5) -> [-1,1]
- 输出：(x+1)/2 -> [0,1]

### 7.2 图片变成正方形了？
当前为了与训练/测试一致，使用 resize 到 `(size,size)`。
如果要保留长宽比，需要额外做 padding 或 center-crop（后续再加）。

