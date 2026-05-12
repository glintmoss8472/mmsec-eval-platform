# 部署补充说明

本文档补充 README 中的启动步骤，重点说明常驻运行、反向代理和运行目录管理。

## 推荐运行目录

```text
/opt/mmsec-eval-platform
```

以下目录会在运行时产生或增长，建议放在大容量磁盘：

```text
artifacts/
data/
.hf-cache/
logs/
tmp/
```

可以用软链接把这些目录指向数据盘。

## 环境变量

常用变量：

```bash
PYTHON_BIN=/opt/mmsec-eval-platform/.venv/bin/python
PYTHONPATH=/opt/mmsec-eval-platform/src
LISTEN_HOST=127.0.0.1
PORT=8000
MMSEC_BOOTSTRAP_ENABLED=1
MMSEC_ARTIFACTS_DIR=/opt/mmsec-eval-platform/artifacts
MMSEC_APP_DB=/opt/mmsec-eval-platform/artifacts/app.db
MMSEC_HF_LOCAL_ONLY=1
HF_HOME=/opt/mmsec-eval-platform/.hf-cache
TRANSFORMERS_CACHE=/opt/mmsec-eval-platform/.hf-cache/transformers
HF_DATASETS_CACHE=/opt/mmsec-eval-platform/.hf-cache/datasets
HF_HUB_DISABLE_XET=1
```

首次下载模型时可以临时设置：

```bash
MMSEC_HF_LOCAL_ONLY=0
HF_ENDPOINT=https://hf-mirror.com
```

## systemd

1. 复制模板：

```bash
cp deploy/env.example .env
sudo cp deploy/mmsec-eval-platform.service.example /etc/systemd/system/mmsec-eval-platform.service
```

2. 修改 `.env` 中的路径和端口。

3. 启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable mmsec-eval-platform
sudo systemctl start mmsec-eval-platform
```

4. 查看：

```bash
sudo systemctl status mmsec-eval-platform --no-pager
journalctl -u mmsec-eval-platform -f
```

## Nginx

如果需要公网或内网域名访问，推荐通过 Nginx 反代：

```bash
sudo cp deploy/nginx-mmsec-eval-platform.conf.example /etc/nginx/sites-available/mmsec-eval-platform
sudo ln -s /etc/nginx/sites-available/mmsec-eval-platform /etc/nginx/sites-enabled/mmsec-eval-platform
sudo nginx -t
sudo systemctl reload nginx
```

生产环境不建议直接把 Uvicorn 端口暴露到公网。

## 常见问题

### 页面显示 frontend_not_built

执行：

```bash
pnpm -C frontend install --frozen-lockfile
pnpm -C frontend build
```

### CUDA 不可用

检查：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

重新安装 CUDA 版 PyTorch：

```bash
export PYTHON_BIN="$PWD/.venv/bin/python"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
bash scripts/install_torch_cuda.sh
```

### 端口无法访问

```bash
ss -lntp | grep 8000
curl -fsS http://127.0.0.1:8000/api/v1/health
```

如果使用 Nginx：

```bash
sudo nginx -t
sudo journalctl -u nginx --no-pager -n 100
```
