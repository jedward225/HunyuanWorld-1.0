# 使用已缓存的模型

## 模型位置

HunyuanWorld 使用的模型会自动从 HuggingFace 下载并缓存到：
`~/.cache/huggingface/hub/`

当前已缓存的模型：
- `models--tencent--HunyuanWorld-1` (LoRA weights)

需要下载的模型（首次运行时会自动下载）：
- `black-forest-labs/FLUX.1-dev` (文本生成全景图)
- `black-forest-labs/FLUX.1-Fill-dev` (图像生成全景图)

## 使用方法

模型会自动从缓存加载，无需修改代码。直接运行：

```bash
# 激活环境
conda activate HunyuanWorld

# 文本生成全景图
python3 demo_panogen.py --prompt "你的提示词" --output_path test_results/case1

# 图像生成全景图  
python3 demo_panogen.py --image_path examples/case2/input.png --output_path test_results/case2
```

## 避免重复下载

1. **保持缓存目录**：不要删除 `~/.cache/huggingface/` 目录
2. **首次运行**：第一次运行时会下载 FLUX 模型（约 23GB），请确保网络稳定
3. **离线使用**：下载完成后，可以设置环境变量离线使用：
   ```bash
   export TRANSFORMERS_OFFLINE=1
   export HF_DATASETS_OFFLINE=1
   ```

## 如果有本地模型

如果你在其他位置有 FLUX 模型（如 /mnt/pretrained），可以：

1. 创建符号链接到 HuggingFace 缓存：
```bash
# 例如，如果 FLUX.1-dev 在 /mnt/pretrained/FLUX.1-dev
ln -s /mnt/pretrained/FLUX.1-dev ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev
```

2. 或者修改代码中的 model_path：
```python
# 在 demo_panogen.py 中修改
self.model_path = "/mnt/pretrained/FLUX.1-dev"  # 改为本地路径
```

## ZIM 模型

ZIM 模型需要单独下载：
```bash
cd ZIM/zim_vit_l_2092
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx
```

## 验证模型

在 conda 环境中运行测试脚本：
```bash
conda activate HunyuanWorld
python3 test_model_loading.py
```