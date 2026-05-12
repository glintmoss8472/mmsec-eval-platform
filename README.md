# mmsec-eval-platform

`mmsec-eval-platform` 是一个面向多模态模型的对抗样本安全测评平台。系统包含评测引擎、FastAPI 后端、React 前端、任务队列、运行记录、案例复盘和报告展示能力。

本仓库只包含可迁移的工程代码、配置、脚本、必要测试和小型演示数据；不包含论文稿、答辩材料、历史实验产物、大模型权重或正式数据集。

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

仓库不包含大模型权重和正式数据集。首次下载模型时可临时允许联网：

```bash
export MMSEC_HF_LOCAL_ONLY=0
export HF_ENDPOINT="https://hf-mirror.com"
```

预取部分本地 VLM：

```bash
source .venv/bin/activate
python scripts/prefetch_local_vlm_assets.py \
  --out-root artifacts/local_vlm \
  --models qwen25_vl,qwen3_vl
```

可选模型 key 由 `src/mmsec_eval/model_adapters/local_vlm_catalog.py` 定义。

数据准备脚本：

```bash
python scripts/prepare_flickr30k.py --help
python scripts/prepare_coco_subset.py --help
python scripts/prepare_flickr1k.py --help
```

模型、数据和运行产物建议放在大容量磁盘下，并通过环境变量或软链接指向项目目录。

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
