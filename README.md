# mmsec-eval-platform

`mmsec-eval-platform` 是一个面向多模态模型的对抗样本安全测评平台。系统包含评测引擎、FastAPI 后端、React 前端、任务队列、运行记录、案例复盘和报告展示能力。

本仓库包含可迁移的工程代码、配置、脚本、必要测试和小型演示数据。

## 主要功能

- 创建图文检索、视觉问答、图像描述等多模态安全测评任务
- 运行 clean / adversarial / defended 评测流程
- 支持 CLIP、BLIP、ViLT、本地 OpenAI-compatible VLM 服务等模型适配
- 支持 AdvCLIP、TMM、AdvEDM-inspired、视觉扰动等攻击配置
- 聚合运行结果、风险评分、证据案例和报告数据
- 提供中文化 Web 页面：总览、新建测评、任务监控、跨运行分析、案例库和报告页

## 技术栈

- 后端：Python、FastAPI、SQLite、Uvicorn
- 评测引擎：PyTorch、Transformers、Pillow、NumPy
- 前端：React、TypeScript、Vite、TanStack Query、ECharts
- 部署：Linux shell、systemd、Nginx

## 目录结构

```text
src/mmsec_api/      FastAPI 后端、接口路由、任务队列和数据存储
src/mmsec_eval/     评测引擎、攻击、模型适配器、指标和报告渲染
frontend/           React + Vite 前端源码
configs/            测评、benchmark、风险权重和 profile 配置
scripts/            启动、数据准备、模型服务、验证和实验脚本
seed/               小型演示数据，用于基础页面和流程验证
tests/              必要测试
assets/templates/   HTML 报告模板
deploy/             systemd、Nginx 部署模板
docs/               部署补充说明
```

## 环境要求

推荐服务器环境：

- Ubuntu 22.04 / 24.04
- Python 3.11 或 3.12，最低 3.10
- Node.js 20 或 22
- pnpm 9.x
- NVIDIA GPU 可选；真实大模型评测建议 24GB 显存级别

只启动页面和 API 不强制要求 GPU；运行真实本地多模态模型时需要 GPU、CUDA 版 PyTorch、模型权重和数据集。

## 快速启动

以下命令假设项目放在 `/opt/mmsec-eval-platform`。

### 1. 安装系统依赖

```bash
sudo apt-get update
sudo apt-get install -y \
  python3 \
  python3-venv \
  python3-dev \
  build-essential \
  curl \
  git \
  rsync \
  ca-certificates
```

确认 Node.js 和 pnpm：

```bash
node --version
corepack enable
corepack prepare pnpm@9.12.3 --activate
pnpm --version
```

### 2. 安装 Python 依赖

```bash
cd /opt/mmsec-eval-platform
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

如果服务器有 NVIDIA GPU，建议安装 CUDA 版 PyTorch：

```bash
export PYTHON_BIN="$PWD/.venv/bin/python"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
bash scripts/install_torch_cuda.sh
pip install -r requirements-gpu.txt
```

验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY
```

### 3. 安装并构建前端

```bash
cd /opt/mmsec-eval-platform
pnpm -C frontend install --frozen-lockfile
pnpm -C frontend build
```

构建成功后会生成：

```text
frontend/dist/index.html
frontend/dist/assets/
```

生产模式下后端会直接托管 `frontend/dist`。

### 4. 启动后端和页面

```bash
cd /opt/mmsec-eval-platform
export PYTHONPATH="$PWD/src"
export PYTHON_BIN="$PWD/.venv/bin/python"
export USE_EXISTING_PYTHON=1
export LISTEN_HOST=127.0.0.1
export PORT=8000
export MMSEC_BOOTSTRAP_ENABLED=1
export MMSEC_ARTIFACTS_DIR="$PWD/artifacts"
export MMSEC_APP_DB="$PWD/artifacts/app.db"
export HF_HOME="$PWD/.hf-cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_XET=1

bash scripts/run_backend.sh
```

检查 API：

```bash
curl -fsS http://127.0.0.1:8000/api/v1/health
curl -fsS http://127.0.0.1:8000/api/v1/system/overview
```

浏览器访问：

```text
http://127.0.0.1:8000/
```

主要页面：

- `/`：系统总览
- `/testing`：新建测评
- `/jobs`：任务监控
- `/analysis`：跨运行分析
- `/cases`：案例库
- `/reports`：报告列表

## 开发模式

开发时可以分开启动后端和前端。

后端：

```bash
cd /opt/mmsec-eval-platform
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
uvicorn mmsec_api.main:app --host 127.0.0.1 --port 8000 --reload
```

前端：

```bash
pnpm -C frontend dev
```

开发前端地址：

```text
http://127.0.0.1:5173/
```

Vite 会把 `/api/v1` 请求代理到 `127.0.0.1:8000`。

## 模型和数据

模型权重和正式数据集按各自许可证单独下载到目标服务器，仓库只记录下载方式和目录约定，不提交大型二进制文件。以下路径都以 `/opt/mmsec-eval-platform` 为项目根目录。

建议先创建目录：

```bash
cd /opt/mmsec-eval-platform
mkdir -p artifacts/local_vlm artifacts/hf_models artifacts/hf-cache data/coco data/flickr30k data/flickr1k data/coco2014/generation
```

### 模型权重下载

传统图文检索模型放在 `artifacts/hf_models/`：

| 用途 | 模型仓库 | 放置位置 |
| --- | --- | --- |
| CLIP | `openai/clip-vit-base-patch32` | `artifacts/hf_models/clip` |
| BLIP-ITM | `Salesforce/blip-itm-base-coco` | `artifacts/hf_models/blip_itm` |
| ViLT-ITM | `dandelin/vilt-b32-finetuned-coco` | `artifacts/hf_models/vilt_itm` |
| BERT MLM | `bert-base-uncased` | `artifacts/hf_models/bert_mlm` |

国际源使用 Hugging Face 官方 CLI：

```bash
source .venv/bin/activate
pip install -U huggingface_hub
hf download openai/clip-vit-base-patch32 --local-dir artifacts/hf_models/clip
hf download Salesforce/blip-itm-base-coco --local-dir artifacts/hf_models/blip_itm
hf download dandelin/vilt-b32-finetuned-coco --local-dir artifacts/hf_models/vilt_itm
hf download bert-base-uncased --local-dir artifacts/hf_models/bert_mlm
```

国内网络可临时使用 HF-Mirror：

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download openai/clip-vit-base-patch32 --local-dir artifacts/hf_models/clip
```

本地 OpenAI-compatible VLM 放在 `artifacts/local_vlm/`。目录名必须和代码中的 key 一致：

| key | 默认模型仓库 | 放置位置 |
| --- | --- | --- |
| `qwen35_9b` | `Qwen/Qwen3.5-9B` | `artifacts/local_vlm/qwen35_9b` |
| `qwen3_vl` | `Qwen/Qwen3-VL-8B-Instruct` | `artifacts/local_vlm/qwen3_vl` |
| `qwen25_vl` | `Qwen/Qwen2.5-VL-7B-Instruct` | `artifacts/local_vlm/qwen25_vl` |
| `internvl35` | `OpenGVLab/InternVL3_5-8B-HF` | `artifacts/local_vlm/internvl35` |
| `minicpm_v` | `openbmb/MiniCPM-V-4_5` | `artifacts/local_vlm/minicpm_v` |
| `ovis25` | `AIDC-AI/Ovis2.5-9B` | `artifacts/local_vlm/ovis25` |
| `gemma3_12b` | `google/gemma-3-12b-it` | `artifacts/local_vlm/gemma3_12b` |

国际源批量下载：

```bash
source .venv/bin/activate
pip install -U huggingface_hub
python scripts/prefetch_local_vlm_assets.py \
  --out-root artifacts/local_vlm \
  --models qwen25_vl,qwen3_vl,internvl35
```

国内源优先使用 ModelScope；脚本会读取 `src/mmsec_eval/model_adapters/local_vlm_catalog.py` 并写入同样的本地目录：

```bash
source .venv/bin/activate
pip install -U modelscope
OUT_ROOT="$PWD/artifacts/local_vlm" bash scripts/download_target_vlms_modelscope.sh
```

如某个模型在 ModelScope 上的仓库名不同，可手动下载到对应目录，例如：

```bash
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir artifacts/local_vlm/qwen25_vl
printf '%s\n' Qwen/Qwen2.5-VL-7B-Instruct > artifacts/local_vlm/qwen25_vl/.source_model
```

部分模型可能需要登录、申请许可或接受模型协议。下载完成后，每个本地 VLM 目录至少应包含 `config.json`、权重文件和 `.source_model`，本地启动脚本会用它判断模型是否准备完成。

### 正式数据集下载

图文检索 COCO 子集使用 COCO 2017 validation split，放在 `data/coco/`：

```bash
python scripts/prepare_coco_subset.py \
  -Root data/coco \
  -Split val2017 \
  -MaxItems 5000 \
  -DownloadAnnotations \
  -DownloadImages
```

下载完成后的关键文件位置：

```text
data/coco/val2017/*.jpg
data/coco/annotations/captions_val2017.json
data/coco/annotations/captions_val2017_subset.json
data/coco/annotations/captions_val2017_subset.jsonl
```

图文检索 Flickr30k 放在 `data/flickr30k/`：

```bash
python scripts/prepare_flickr30k.py \
  -Root data/flickr30k \
  -ImageDir images \
  -OutputFile captions_index.jsonl \
  -MaxItems 1000
```

如果服务器无法访问默认下载源，可从 Hugging Face、HF-Mirror、Gitee AI、ATYUN 或 OpenDataLab/OpenXLab 下载 Flickr30k，并把图片和标注整理为：

```text
data/flickr30k/images/*.jpg
data/flickr30k/captions_index.jsonl
```

`captions_index.jsonl` 每行至少包含：

```json
{"id":"flickr30k-000001","image":"images/1000092795.jpg","caption":"A caption for this image.","split":"test"}
```

Flickr1k 可以从已经准备好的 Flickr30k 截取：

```bash
python scripts/prepare_flickr1k.py \
  -Root data/flickr1k \
  -SourceRoot data/flickr30k \
  -SourceImageDir images \
  -MaxItems 1000
```

VQA 和图像描述任务使用 COCO2014/VQA v2，生成本项目需要的 JSONL 后放在 `data/coco2014/generation/`：

```bash
python scripts/prepare_coco2014_generation.py \
  --root data/coco2014 \
  --download \
  --download-images \
  --max-vqa 1000 \
  --max-caption 1000
```

下载完成后的关键文件位置：

```text
data/coco2014/val2014/COCO_val2014_*.jpg
data/coco2014/annotations/v2_OpenEnded_mscoco_val2014_questions.json
data/coco2014/annotations/v2_mscoco_val2014_annotations.json
data/coco2014/annotations/captions_val2014.json
data/coco2014/annotations/instances_val2014.json
data/coco2014/generation/vqa_v2_coco_val.jsonl
data/coco2014/generation/coco_caption_object_val.jsonl
```

常用下载入口：

| 资源 | 国际源 | 国内/镜像入口 |
| --- | --- | --- |
| Hugging Face 模型 | <https://huggingface.co/models> | <https://hf-mirror.com/> |
| ModelScope 模型 | <https://www.modelscope.cn/models> | ModelScope 本身为国内源 |
| COCO 2017 / 2014 | <https://cocodataset.org/#download> | <https://opendatalab.com/OpenDataLab/COCO_2017/cli/main>、<https://opendatalab.org.cn/OpenDataLab/COCO_2014> |
| Flickr30k | <https://huggingface.co/datasets/nlphuji/flickr30k> | <https://ai.gitee.com/hf-datasets/HuggingFaceM4/flickr30k>、<https://www.atyun.com/datasets/files/nlphuji/flickr30k.html> |
| VQA v2 | <https://visualqa.org/download.html> | <https://hyper.ai/cn/datasets/15514>；也可先下载 COCO2014 国内镜像，再从 VQA 官方源下载问题/答案标注 |

正式评测前应确认：

```bash
test -f artifacts/hf_models/clip/config.json
test -f data/coco/annotations/captions_val2017_subset.json
test -f data/flickr30k/captions_index.jsonl
test -f data/coco2014/generation/vqa_v2_coco_val.jsonl
test -f data/coco2014/generation/coco_caption_object_val.jsonl
```

模型、数据和运行产物建议放在大容量磁盘下；如果项目根目录磁盘较小，可以把 `artifacts/` 和 `data/` 放到大盘后用软链接接回项目根目录。

## systemd 部署

复制环境变量模板：

```bash
cd /opt/mmsec-eval-platform
cp deploy/env.example .env
vim .env
```

安装服务：

```bash
sudo cp deploy/mmsec-eval-platform.service.example /etc/systemd/system/mmsec-eval-platform.service
sudo systemctl daemon-reload
sudo systemctl enable mmsec-eval-platform
sudo systemctl start mmsec-eval-platform
sudo systemctl status mmsec-eval-platform --no-pager
```

查看日志：

```bash
journalctl -u mmsec-eval-platform -f
```

## Nginx 反向代理

建议后端只监听 `127.0.0.1:8000`，由 Nginx 对外提供访问：

```bash
sudo cp deploy/nginx-mmsec-eval-platform.conf.example /etc/nginx/sites-available/mmsec-eval-platform
sudo ln -s /etc/nginx/sites-available/mmsec-eval-platform /etc/nginx/sites-enabled/mmsec-eval-platform
sudo nginx -t
sudo systemctl reload nginx
```

## 验证

```bash
python -m compileall -q src mmsec_api mmsec_eval scripts app
pnpm -C frontend build
curl -fsS http://127.0.0.1:8000/api/v1/health
curl -fsS http://127.0.0.1:8000/api/v1/system/overview
```

真实攻击测评需要在模型和数据准备完成后再验收。页面能启动不等于所有大模型实验已经完成验证。

## 重要边界

本项目是工程化多模态对抗样本安全测评平台。仓库提供代码、配置和部署入口；模型权重、正式数据集和新的实验结果需要在目标服务器上单独准备和重新运行。
