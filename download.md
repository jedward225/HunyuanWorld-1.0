# HunyuanWorld Setup Commands

## 1. Create Conda Environment
```bash
conda create -n HunyuanWorld python=3.10 -y
conda activate HunyuanWorld
```

## 2. Install PyTorch with CUDA
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

## 3. Install Core Dependencies
```bash
pip install diffusers==0.32.0 transformers==4.51.0 accelerate==1.6.0 huggingface-hub==0.30.2
pip install numpy==1.24.1 opencv-python==4.11.0.86 pillow matplotlib scipy scikit-image
pip install einops omegaconf hydra-core pyyaml tqdm easydict
pip install xformers==0.0.28.post2
pip install peft  # Required for LoRA loading
pip install flash-attn==2.7.4.post1
pip install open3d trimesh plyfile
pip install onnxruntime-gpu==1.21.1
pip install kornia albumentations
pip install segment-anything timm
pip install open-clip-torch sentencepiece
```

## 4. Install PyTorch3D (try pre-built wheel first)
```bash
# Try pre-built wheel (faster)
# pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu124_pyt250/download.html

# If above fails, install from source (slower)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## 5. Install MoGe
```bash
pip install git+https://github.com/microsoft/MoGe.git
```

## 6. Install Real-ESRGAN
```bash
git clone https://githubfast.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install basicsr
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop
cd ..
```

## 7. Install ZIM and Download Models
```bash
git clone https://github.com/naver-ai/ZIM.git
cd ZIM
pip install -e .
mkdir zim_vit_l_2092
cd zim_vit_l_2092
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx
wget https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx
cd ../..
```

## 8. Install Draco (Optional - for mesh export)
```bash
git clone https://github.com/google/draco.git
cd draco
mkdir build
cd build
cmake ..
make
sudo make install
cd ../..
```

## 9. Login to Hugging Face
```bash
huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN
```

## 10. Use Local Models (推荐)
本地模型已经存在，无需下载：
- FLUX 模型位置: `/mnt/pretrained/`
- HunyuanWorld LoRA 模型位置: `/mnt/zhouzihan/HunyuanWorld-1.0/models/`

使用本地模型运行：
```bash
# 使用本地模型脚本
./run_with_local_models.sh

# 或直接运行
python3 run_local.py --prompt "A beautiful landscape" --output_path test_results/test

# 从图片生成
python3 run_local.py --image_path examples/case2/input.png --output_path test_results/test
```

## 10a. (可选) 如果需要下载模型
如果本地模型不可用，模型会自动从 HuggingFace 下载：
- https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Text
- https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoDiT-Image
- https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Scene
- https://huggingface.co/tencent/HunyuanWorld-1/tree/main/HunyuanWorld-PanoInpaint-Sky

## Test Installation
```bash
# 测试配置
python3 config.py

# 使用本地模型测试
python3 run_local.py --prompt "A beautiful landscape" --output_path test_results/test
```