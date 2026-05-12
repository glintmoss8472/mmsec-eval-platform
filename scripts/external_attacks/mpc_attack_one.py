# 文件说明：该文件属于外部攻击脚本，集中实现 mpc attack one 相关逻辑。
from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms


# 执行 `install compat stubs` 辅助逻辑，保持外部攻击脚本中的输入处理和结果输出一致。
def _install_compat_stubs() -> None:
    if "config_schema" not in sys.modules:
        module = ModuleType("config_schema")
        module.MainConfig = SimpleNamespace
        sys.modules["config_schema"] = module
    try:
        import transformers
        if not hasattr(transformers, "DINOv3ViTModel"):
            transformers.DINOv3ViTModel = transformers.AutoModel
    except (ImportError, AttributeError) as exc:
        print(f"transformers DINOv3 compatibility patch skipped: {exc}", file=sys.stderr)


# 加载 `module`，把外部文件、配置或运行产物转换为内存结构。
def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# 准备 `official modules` 数据，补齐后续运行、报告或测试需要的字段。
def _prepare_official_modules(repo: Path):
    _install_compat_stubs()
    sys.path.insert(0, str(repo))
    surrogates = sys.modules.setdefault("surrogates", ModuleType("surrogates"))
    surrogates.__path__ = [str(repo / "surrogates")]
    features = sys.modules.setdefault("surrogates.FeatureExtractors", ModuleType("surrogates.FeatureExtractors"))
    features.__path__ = [str(repo / "surrogates" / "FeatureExtractors")]
    base = _load_module("surrogates.FeatureExtractors.Base", repo / "surrogates" / "FeatureExtractors" / "Base.py")
    clip_b16 = _load_module("surrogates.FeatureExtractors.ClipB16", repo / "surrogates" / "FeatureExtractors" / "ClipB16.py")
    clip_b32 = _load_module("surrogates.FeatureExtractors.ClipB32", repo / "surrogates" / "FeatureExtractors" / "ClipB32.py")
    clip_l336 = _load_module("surrogates.FeatureExtractors.ClipL336", repo / "surrogates" / "FeatureExtractors" / "ClipL336.py")
    clip_laion = _load_module("surrogates.FeatureExtractors.ClipLaion", repo / "surrogates" / "FeatureExtractors" / "ClipLaion.py")
    dino = _load_module("surrogates.FeatureExtractors.DINOv2_Base", repo / "surrogates" / "FeatureExtractors" / "DINOv2_Base.py")
    internvl = _load_module("surrogates.FeatureExtractors.InternVL3_1B", repo / "surrogates" / "FeatureExtractors" / "InternVL3_1B.py")
    utils = _load_module("mpc_attack_official_utils", repo / "utils.py")
    return SimpleNamespace(base=base, clip_b16=clip_b16, clip_b32=clip_b32, clip_l336=clip_l336, clip_laion=clip_laion, dino=dino, internvl=internvl, utils=utils)



# 定位 `manual 模型 路径`，把配置值或请求上下文转换成实际文件系统路径。
def _manual_model_path(repo_id: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", "/root/autodl-tmp/hf-cache"))
    return hf_home / "manual" / repo_id


# 执行 `set Hugging Face cache 环境` 辅助逻辑，保持外部攻击脚本中的输入处理和结果输出一致。
def _set_hf_cache_env() -> None:
    hf_home = Path(os.environ.get("HF_HOME", "/root/autodl-tmp/hf-cache")).expanduser().resolve()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# 定位 `cached 模型 路径`，把配置值或请求上下文转换成实际文件系统路径。
def _cached_model_path(repo_id: str) -> Path:
    manual = _manual_model_path(repo_id)
    if manual.exists():
        return manual
    hf_home = Path(os.environ.get("HF_HOME", "/root/autodl-tmp/hf-cache")).expanduser().resolve()
    hub = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", hf_home / "hub")).expanduser().resolve()
    snapshots = hub / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if snapshots.exists():
        candidates = [path for path in snapshots.iterdir() if path.is_dir()]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
    return manual


# 执行 `补丁 来源 pretrained` 辅助逻辑，保持外部攻击脚本中的输入处理和结果输出一致。
def _patch_from_pretrained(module, class_names: tuple[str, ...], repo_id: str, local_path: Path) -> None:
    if not local_path.exists():
        return
    for class_name in class_names:
        cls = getattr(module, class_name, None)
        if cls is None or not hasattr(cls, "from_pretrained"):
            continue
        original = cls.from_pretrained

        # 执行 `来源 pretrained` 辅助逻辑，保持外部攻击脚本中的输入处理和结果输出一致。
        def _from_pretrained(path_or_repo_id, *args, _original=original, **kwargs):
            if str(path_or_repo_id) == repo_id:
                kwargs.setdefault("local_files_only", True)
                return _original(str(local_path), *args, **kwargs)
            return _original(path_or_repo_id, *args, **kwargs)

        cls.from_pretrained = _from_pretrained


# 执行 `补丁 CLIP repos` 辅助逻辑，保持外部攻击脚本中的输入处理和结果输出一致。
def _patch_clip_repos(mods) -> None:
    for module, repo_id in (
        (mods.clip_b16, "openai/clip-vit-base-patch16"),
        (mods.clip_b32, "openai/clip-vit-base-patch32"),
        (mods.clip_laion, "laion/CLIP-ViT-G-14-laion2B-s12B-b42K"),
    ):
        _patch_from_pretrained(module, ("CLIPModel", "CLIPProcessor", "AutoTokenizer"), repo_id, _cached_model_path(repo_id))


# 执行 `forward` 辅助逻辑，保持外部攻击脚本中的输入处理和结果输出一致。
def _patch_clip_forward(cls) -> None:
    # 封装 _forward 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
    def _forward(self, x):
        inputs = dict(pixel_values=self.normalizer(x))
        out = self.model.get_image_features(**inputs)
        if not torch.is_tensor(out):
            outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
            out = outputs.pooler_output
            projection = getattr(self.model, "visual_projection", None)
            if projection is not None:
                out = projection(out)
        return out / out.norm(dim=1, keepdim=True)

    # 规范化 `文本 features get` 字段，把空值和非字符串输入转换为稳定文本。
    def _text_features_get(self, text):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt").to(self.model.device)
        out = self.model.get_text_features(**inputs)
        if not torch.is_tensor(out):
            outputs = self.model.text_model(**inputs)
            out = outputs.pooler_output
            projection = getattr(self.model, "text_projection", None)
            if projection is not None:
                out = projection(out)
        return out / out.norm(dim=1, keepdim=True)

    cls.forward = _forward
    cls.text_features_get = _text_features_get



# 执行 `forward` 辅助逻辑，保持外部攻击脚本中的输入处理和结果输出一致。
def _patch_internvl_forward(cls) -> None:
    # 封装 _forward 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
    def _forward(self, x):
        pixel_values = self.get_input_pixel_values(x)
        pixel_values = pixel_values.to(self.model.device, dtype=torch.bfloat16)
        outputs = self.model.get_image_features(pixel_values=pixel_values)
        if torch.is_tensor(outputs):
            image_features = outputs
        else:
            image_features = getattr(outputs, "pooler_output", None)
            if image_features is None:
                image_features = getattr(outputs, "last_hidden_state", None)
            if image_features is None:
                raise RuntimeError("InternVL get_image_features returned no tensor features")
        return image_features / image_features.norm(dim=-1, keepdim=True)

    cls.forward = _forward

# 执行 `set seed` 辅助逻辑，保持外部攻击脚本中的输入处理和结果输出一致。
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 转换 `tensor` 输入，在类型不匹配时回退到安全默认值。
def _to_tensor(pic: Image.Image) -> torch.Tensor:
    arr = torch.from_numpy(np.array(pic, np.uint8, copy=True))
    arr = arr.view(pic.size[1], pic.size[0], len(pic.getbands()))
    return arr.permute((2, 0, 1)).contiguous().float()


# 加载 `图像`，把外部文件、配置或运行产物转换为内存结构。
def _load_image(path: Path, input_res: int, device: torch.device) -> torch.Tensor:
    transform_fn = transforms.Compose([
        transforms.Resize((input_res, input_res), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Lambda(_to_tensor),
    ])
    return transform_fn(Image.open(path)).unsqueeze(0).to(device)


# 构建 `CLIP 模型` 数据，集中整理外部攻击脚本需要的输出结构。
def _build_clip_models(mods, names: list[str], device: torch.device):
    mapping = {
        "B16": mods.clip_b16.ClipB16FeatureExtractor,
        "B32": mods.clip_b32.ClipB32FeatureExtractor,
        "L336": mods.clip_l336.ClipL336FeatureExtractor,
        "Laion": mods.clip_laion.ClipLaionFeatureExtractor,
    }
    models = []
    for name in names:
        if name not in mapping:
            raise ValueError(f"unknown MPCAttack clip backbone: {name}")
        cls = mapping[name]
        _patch_clip_forward(cls)
        model = cls().eval().to(device)
        model.requires_grad_(False)
        models.append(model)
    return mods.base.EnsembleFeatureExtractor_our(models), models


# 执行 `cat CLIP features` 辅助逻辑，保持外部攻击脚本中的输入处理和结果输出一致。
def _cat_clip_features(feat_dict: dict) -> torch.Tensor:
    return torch.cat([feat_dict[idx] for idx in sorted(feat_dict.keys())], dim=1).float()


# 执行 `join features` 辅助逻辑，保持外部攻击脚本中的输入处理和结果输出一致。
def _join_features(parts: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.float() for p in parts], dim=1)


# 规范化 `文本 features` 字段，把空值和非字符串输入转换为稳定文本。
def _text_features(clip_models, text: str) -> dict:
    return clip_models.get_text_features(text or "a photo")


# 解析 `device` 的真实位置或配置值，减少调用方重复分支。
def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        device_name = "cuda:0"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    return torch.device(device_name)


# 构建 `official 模型` 数据，集中整理外部攻击脚本需要的输出结构。
def _build_official_models(args, mods, device: torch.device):
    internvl_path = Path(args.internvl_model_path).expanduser().resolve() if args.internvl_model_path else _cached_model_path("OpenGVLab/InternVL3-1B-hf")
    dino_path = Path(args.dino_model_path).expanduser().resolve() if args.dino_model_path else _cached_model_path("facebook/dinov2-base")
    _patch_from_pretrained(mods.internvl, ("AutoProcessor", "AutoModelForImageTextToText", "AutoTokenizer"), "OpenGVLab/InternVL3-1B-hf", internvl_path)
    _patch_from_pretrained(mods.dino, ("AutoModel", "AutoImageProcessor"), "facebook/dinov2-base", dino_path)
    _patch_clip_repos(mods)
    clip_names = [item for item in args.clip_backbones.replace(";", ",").split(",") if item] or ["B32"]
    clip_models, _ = _build_clip_models(mods, clip_names, device)
    _patch_internvl_forward(mods.internvl.InternVL3_1B_FeatureExtractor)
    internvl_model = None if args.disable_internvl else mods.internvl.InternVL3_1B_FeatureExtractor().eval().to(device)
    dino_model = None if args.disable_dino else mods.dino.DINOv2FeatureExtractor().eval().to(device)
    if internvl_model is not None:
        internvl_model.requires_grad_(False)
    if dino_model is not None:
        dino_model.requires_grad_(False)
    return clip_models, internvl_model, dino_model


# 执行 `reference features` 辅助逻辑，保持外部攻击脚本中的输入处理和结果输出一致。
def _reference_features(args, clip_models, internvl_model, dino_model, img_src: torch.Tensor, img_tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lam = float(args.lam)
    with torch.no_grad():
        src_text_feats = _text_features(clip_models, args.source_text)
        tgt_text_feats = _text_features(clip_models, args.target_text)
        src_img_feats = clip_models(img_src)
        tgt_img_feats = clip_models(img_tgt)
        clip_src_parts = []
        clip_tgt_parts = []
        for idx in sorted(src_img_feats.keys()):
            clip_src_parts.append(lam * src_img_feats[idx].float() + (1.0 - lam) * src_text_feats[idx].float())
            clip_tgt_parts.append(lam * tgt_img_feats[idx].float() + (1.0 - lam) * tgt_text_feats[idx].float())
        z_src_parts = [torch.cat(clip_src_parts, dim=1)]
        z_tgt_parts = [torch.cat(clip_tgt_parts, dim=1)]
        if internvl_model is not None:
            z_src_parts.append(internvl_model(img_src).mean(dim=1).float())
            z_tgt_parts.append(internvl_model(img_tgt).mean(dim=1).float())
        if dino_model is not None:
            z_src_parts.append(dino_model(img_src).float())
            z_tgt_parts.append(dino_model(img_tgt).float())
        return _join_features(z_src_parts), _join_features(z_tgt_parts)


# 执行 `optimize delta` 辅助逻辑，保持外部攻击脚本中的输入处理和结果输出一致。
def _optimize_delta(args, mods, clip_models, internvl_model, dino_model, source_crop, img_src: torch.Tensor, z_src: torch.Tensor, z_tgt: torch.Tensor, epsilon: float, alpha: float) -> torch.Tensor:
    device = img_src.device
    delta = (min(epsilon, 16.0) / 255.0) * torch.randn_like(img_src, device=device)
    delta.requires_grad_(True)
    momentum = torch.zeros_like(delta, requires_grad=False)
    for step in range(max(1, int(args.steps))):
        adv_img = torch.clamp(img_src + delta, 0.0, 255.0)
        local_cropped = source_crop(adv_img)
        adv_parts = [_cat_clip_features(clip_models(local_cropped))]
        if internvl_model is not None:
            adv_parts.append(internvl_model(local_cropped).mean(dim=1).float())
        if dino_model is not None:
            adv_parts.append(dino_model(local_cropped).float())
        z_adv = _join_features(adv_parts)
        loss = mods.utils.info_nce_loss(z_adv, z_tgt.detach(), z_src.detach(), temperature=float(args.tau), omega=float(args.omega))
        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
        momentum = momentum * 0.9 + grad
        delta.data = torch.clamp(delta + alpha * torch.sign(momentum), min=-epsilon, max=epsilon)
        if step == 0 or step == int(args.steps) - 1:
            sim_src = torch.mean(F.cosine_similarity(z_adv.detach(), z_src.detach(), dim=-1)).item()
            sim_tgt = torch.mean(F.cosine_similarity(z_adv.detach(), z_tgt.detach(), dim=-1)).item()
            print(f"step={step} loss={float(loss.detach().cpu()):.6f} sim_src={sim_src:.6f} sim_tgt={sim_tgt:.6f}", flush=True)
    return delta.detach()


# 写出 `adv`，保证后续报告、页面或复现实验能读取。
def _save_adv(output_image: str, input_image: str, img_src: torch.Tensor, delta: torch.Tensor) -> None:
    adv = torch.clamp((img_src + delta) / 255.0, 0.0, 1.0)
    output = Path(output_image).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    orig = _to_tensor(Image.open(input_image).convert("RGB")).unsqueeze(0) / 255.0
    clean_small = torch.clamp(img_src.detach().cpu() / 255.0, 0.0, 1.0)
    delta_small = adv.detach().cpu() - clean_small
    delta_big = F.interpolate(delta_small, size=orig.shape[-2:], mode="bilinear", align_corners=False)
    out = torch.clamp(orig + delta_big, 0.0, 1.0)
    torchvision.utils.save_image(out.squeeze(0), str(output))


# 作为 `mpc_attack_one.py` 的执行入口，串联参数读取、核心处理和退出状态。
def run(args) -> None:
    repo = Path(args.repo_dir).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"MPCAttack repo_dir not found: {repo}")
    device = _resolve_device(args.device)
    _set_seed(int(args.seed))
    mods = _prepare_official_modules(repo)
    clip_models, internvl_model, dino_model = _build_official_models(args, mods, device)
    img_src = _load_image(Path(args.input_image), int(args.input_res), device)
    img_tgt = _load_image(Path(args.target_image), int(args.input_res), device)
    crop_scale = tuple(float(x) for x in args.crop_scale.split(",") if x)
    if len(crop_scale) == 1:
        crop_scale = (crop_scale[0], crop_scale[0])
    source_crop = transforms.RandomResizedCrop(int(args.input_res), scale=crop_scale)
    epsilon = float(args.epsilon)
    epsilon = epsilon * 255.0 if epsilon <= 1.0 else epsilon
    alpha = float(args.alpha)
    alpha = alpha * 255.0 if alpha <= 1.0 else alpha
    z_src, z_tgt = _reference_features(args, clip_models, internvl_model, dino_model, img_src, img_tgt)
    delta = _optimize_delta(args, mods, clip_models, internvl_model, dino_model, source_crop, img_src, z_src, z_tgt, epsilon, alpha)
    _save_adv(args.output_image, args.input_image, img_src, delta)


# 作为 `mpc_attack_one.py` 的执行入口，串联参数读取、核心处理和退出状态。
def main() -> int:
    _set_hf_cache_env()
    parser = argparse.ArgumentParser(description="Run official MPCAttack feature extractors for one target-transfer image attack.")
    parser.add_argument("--repo_dir", required=True)
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--target_image", required=True)
    parser.add_argument("--output_image", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epsilon", type=float, default=0.0627451)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--crop_scale", default="0.5,0.9")
    parser.add_argument("--clip_backbones", default="B16,B32,Laion")
    parser.add_argument("--source_text", default="a source image")
    parser.add_argument("--target_text", default="a target image")
    parser.add_argument("--lam", type=float, default=0.6)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--omega", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--internvl_model_path", default="")
    parser.add_argument("--dino_model_path", default="")
    parser.add_argument("--disable_internvl", action="store_true")
    parser.add_argument("--disable_dino", action="store_true")
    args = parser.parse_args()
    run(args)
    if not Path(args.output_image).exists():
        raise FileNotFoundError(f"MPCAttack did not create expected output: {args.output_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
