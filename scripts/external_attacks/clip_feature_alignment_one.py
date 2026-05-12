# 文件说明：该文件属于外部攻击脚本，集中实现 clip feature alignment one 相关逻辑。
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import torchvision
from PIL import Image
from torchvision import transforms


# 中文注释：封装 _install_compat_stubs 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
def _install_compat_stubs() -> None:
    if "config_schema" not in sys.modules:
        module = ModuleType("config_schema")
        module.MainConfig = SimpleNamespace
        sys.modules["config_schema"] = module


# 中文注释：封装 _to_module 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
def _to_module(repo: Path, module_name: str):
    _install_compat_stubs()
    repo_text = str(repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    if module_name == "FOAttack":
        # The upstream FOA file hard-codes CUDA_VISIBLE_DEVICES="2" for the
        # authors' multi-GPU host. On a single 4090 this hides the real GPU
        # after torch is imported, so load the official module with only that
        # host-specific line disabled.
        source_path = repo / "FOAttack.py"
        source = source_path.read_text(encoding="utf-8").replace('os.environ["CUDA_VISIBLE_DEVICES"] = "2"', 'os.environ.get("CUDA_VISIBLE_DEVICES", "")')
        spec = importlib.util.spec_from_loader(module_name, loader=None, origin=str(source_path))
        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(source_path)
        sys.modules[module_name] = module
        exec(compile(source, str(source_path), "exec"), module.__dict__)
        return module
    return importlib.import_module(module_name)




# 中文注释：封装 _to_tensor 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
def _to_tensor(pic: Image.Image) -> torch.Tensor:
    img = torch.from_numpy(__import__("numpy").array(pic, __import__("numpy").uint8, copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    return img.permute((2, 0, 1)).contiguous().float()

# 中文注释：封装 _patch_transformers_clip_outputs 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
def _patch_transformers_clip_outputs(module) -> None:
    # 中文注释：封装 _forward 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
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

    for class_name in ("ClipB16FeatureExtractor", "ClipB32FeatureExtractor", "ClipL336FeatureExtractor", "ClipLaionFeatureExtractor"):
        cls = getattr(module, class_name, None)
        if cls is not None:
            cls.forward = _forward


# 中文注释：封装 _patch_kmeans_compat 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
def _patch_kmeans_compat() -> None:
    try:
        import kmeans_pytorch
    except ImportError:
        return
    original = kmeans_pytorch.kmeans

    # 中文注释：封装 _kmeans_compat 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
    def _kmeans_compat(*args, **kwargs):
        kwargs.pop("iter_limit", None)
        return original(*args, **kwargs)

    kmeans_pytorch.kmeans = _kmeans_compat
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").endswith("FeatureExtractors.Base") and hasattr(module, "kmeans"):
            module.kmeans = _kmeans_compat

# 中文注释：封装 _patch_logging 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
def _patch_logging(module) -> None:
    # 中文注释：封装 _no_log 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
    def _no_log(*args, **kwargs):
        return None
    module.log_metrics = _no_log
    if hasattr(module, "wandb"):
        module.wandb.log = _no_log
        module.wandb.finish = _no_log
        module.wandb.define_metric = _no_log


# 中文注释：封装 _image_folder 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
def _image_folder(path: Path, image: Path) -> Path:
    cls = path / "class0"
    cls.mkdir(parents=True, exist_ok=True)
    suffix = image.suffix if image.suffix.lower() in {".png", ".jpg", ".jpeg"} else ".png"
    dst = cls / f"0{suffix}"
    Image.open(image).convert("RGB").save(dst)
    return path


# 中文注释：封装 _cfg 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
def _cfg(args, *, backbone: list[str], ensemble: bool):
    epsilon_255 = float(args.epsilon)
    if epsilon_255 <= 1.0:
        epsilon_255 *= 255.0
    return SimpleNamespace(
        data=SimpleNamespace(batch_size=1, num_samples=1, output=str(args.work_dir), cle_data_path=str(args.clean_root), tgt_data_path=str(args.target_root)),
        optim=SimpleNamespace(alpha=float(args.alpha), epsilon=float(epsilon_255), steps=int(args.steps)),
        model=SimpleNamespace(
            input_res=int(args.input_res),
            use_source_crop=bool(args.use_source_crop),
            use_target_crop=bool(args.use_target_crop),
            crop_scale=tuple(float(x) for x in str(args.crop_scale).split(",") if x),
            ensemble=ensemble,
            device=args.device,
            backbone=backbone,
        ),
        attack=args.attack,
    )


# 中文注释：封装 _load_pair 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
def _load_pair(module, cfg, clean_image: Path, target_image: Path):
    transform_fn = transforms.Compose([
        transforms.Resize((cfg.model.input_res, cfg.model.input_res), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Lambda(lambda img: module.to_tensor(img)),
    ])
    image_org = transform_fn(Image.open(clean_image)).unsqueeze(0).to(cfg.model.device)
    image_tgt = transform_fn(Image.open(target_image)).unsqueeze(0).to(cfg.model.device)
    return image_org, image_tgt


# 中文注释：封装 _save 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
def _save(output: Path, adv: torch.Tensor, clean_small_255: torch.Tensor, clean_image: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    orig = _to_tensor(Image.open(clean_image).convert("RGB")).unsqueeze(0) / 255.0
    clean_small = torch.clamp(clean_small_255.detach().cpu() / 255.0, 0.0, 1.0)
    delta = adv.detach().cpu() - clean_small
    delta = torch.nn.functional.interpolate(delta, size=orig.shape[-2:], mode="bilinear", align_corners=False)
    out = torch.clamp(orig + delta, 0.0, 1.0)
    torchvision.utils.save_image(out.squeeze(0), str(output))


# 中文注释：实现 run_m_attack 的核心流程，支撑外部攻击脚本中的业务语义和异常边界。
def run_m_attack(args) -> None:
    repo = Path(args.repo_dir).expanduser().resolve()
    module = _to_module(repo, "generate_adversarial_samples")
    _patch_logging(module)
    backbones = [x for x in args.backbones.split(",") if x] or ["B32"]
    _patch_transformers_clip_outputs(module)
    _patch_kmeans_compat()
    cfg = _cfg(args, backbone=backbones, ensemble=True)
    image_org, image_tgt = _load_pair(module, cfg, Path(args.input_image), Path(args.target_image))
    models = []
    for name in cfg.model.backbone:
        models.append(module.BACKBONE_MAP[name]().eval().to(cfg.model.device).requires_grad_(False))
    ensemble_extractor = module.EnsembleFeatureExtractor(models)
    ensemble_loss = module.EnsembleFeatureLoss(models)
    source_crop = transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale) if cfg.model.use_source_crop else torch.nn.Identity()
    target_crop = transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale) if cfg.model.use_target_crop else torch.nn.Identity()
    attack_fn = {"fgsm": module.fgsm_attack, "mifgsm": module.mifgsm_attack, "pgd": module.pgd_attack}.get(cfg.attack, module.fgsm_attack)
    adv = attack_fn(cfg, ensemble_extractor, ensemble_loss, source_crop, target_crop, 0, image_org, image_tgt)
    _save(Path(args.output_image), adv, image_org, Path(args.input_image))


# 中文注释：实现 run_foa 的核心流程，支撑外部攻击脚本中的业务语义和异常边界。
def run_foa(args) -> None:
    repo = Path(args.repo_dir).expanduser().resolve()
    module = _to_module(repo, "FOAttack")
    _patch_logging(module)
    backbones = [x for x in args.backbones.split(",") if x] or ["B32"]
    _patch_transformers_clip_outputs(module)
    _patch_kmeans_compat()
    cfg = _cfg(args, backbone=backbones, ensemble=True)
    image_org, image_tgt = _load_pair(module, cfg, Path(args.input_image), Path(args.target_image))
    ensemble_extractor, _models, ensemble_loss = module.get_models_ot_with_cluster(cfg, int(args.cluster_num))
    source_crop = transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale) if cfg.model.use_source_crop else torch.nn.Identity()
    target_crop = transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale) if cfg.model.use_target_crop else torch.nn.Identity()
    attack_fn = {"fgsm": module.fgsm_attack, "mifgsm": module.mifgsm_attack, "pgd": module.pgd_attack}.get(cfg.attack, module.fgsm_attack)
    adv = attack_fn(cfg, ensemble_extractor, ensemble_loss, source_crop, target_crop, 0, image_org, image_tgt)
    _save(Path(args.output_image), adv, image_org, Path(args.input_image))


# 中文注释：串联 main 的主流程，集中处理外部攻击脚本的初始化、执行和退出条件。
def main() -> int:
    parser = argparse.ArgumentParser(description="Run official FOA/M-Attack feature-alignment attack functions for one image pair.")
    parser.add_argument("--method", choices=["foa_attack", "m_attack"], required=True)
    parser.add_argument("--repo_dir", required=True)
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--target_image", required=True)
    parser.add_argument("--output_image", required=True)
    parser.add_argument("--work_dir", default="/tmp/att_external_alignment")
    parser.add_argument("--clean_root", default="")
    parser.add_argument("--target_root", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epsilon", type=float, default=0.0627451)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--crop_scale", default="0.5,0.9")
    parser.add_argument("--use_source_crop", action="store_true")
    parser.add_argument("--use_target_crop", action="store_true")
    parser.add_argument("--backbones", default="B16,B32,Laion")
    parser.add_argument("--cluster_num", type=int, default=3)
    parser.add_argument("--attack", choices=["fgsm", "mifgsm", "pgd"], default="fgsm")
    args = parser.parse_args()
    args.work_dir = Path(args.work_dir).expanduser().resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.clean_root = Path(args.clean_root).expanduser().resolve() if args.clean_root else _image_folder(args.work_dir / "clean", Path(args.input_image))
    args.target_root = Path(args.target_root).expanduser().resolve() if args.target_root else _image_folder(args.work_dir / "target", Path(args.target_image))
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    if args.method == "foa_attack":
        run_foa(args)
    else:
        run_m_attack(args)
    if not Path(args.output_image).exists():
        raise FileNotFoundError(f"attack did not create expected output: {args.output_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
