# 文件说明：该文件属于运维与实验脚本，集中实现 audit strict paper protocol 相关逻辑。
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UTC = timezone.utc

ADVCLIP_VICTIMS = ["RN50", "RN101", "ViT-B/16", "ViT-B/32", "ViT-L/14"]
ADVCLIP_VICTIM_DIR = {
    "RN50": "RN50",
    "RN101": "RN101",
    "ViT-B/16": "ViT-B16",
    "ViT-B/32": "ViT-B32",
    "ViT-L/14": "ViT-L14",
}
ADVCLIP_RETRIEVAL_DATASETS = ["nus-wide", "pascal", "wikipedia", "xmedianet"]
ADVCLIP_CLASSIFICATION_DATASETS = ["stl10", "gtsrb", "cifar10", "imagenet"]

TMM_VLR_DATASETS = {
    "flickr30k": {
        "config": "configs/Retrieval_flickr.yaml",
        "annotation": "datasets/flickr30k_test.json",
        "image_root": "datasets/flickr",
        "expected_images": 1000,
        "expected_captions": 5000,
    },
    "mscoco": {
        "config": "configs/Retrieval_coco.yaml",
        "annotation": "datasets/coco_test.json",
        "image_root": "datasets/mscoco",
        "expected_images": 5000,
        "expected_captions": 25000,
    },
}
TMM_TARGET_MODELS = ["ALBEF", "TCL", "X-VLM", "ViLT", "METER", "BLIP", "CLIP_ViT-B16", "CLIP_RN101"]
TMM_TASKS = ["VLR:flickr30k", "VLR:mscoco", "VG:refcoco+", "VE:snli-ve"]


# 执行 `now tag` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


# 写出 `JSON`，保证后续报告、页面或复现实验能读取。
def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 执行 `rel` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        return str(path)


# 执行 `exists` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _exists(path: Path) -> bool:
    return path.exists()


# 判断或归一 `file 状态` 状态，让调用方可以稳定渲染能力和可用性。
def _file_status(root: Path, relative_path: str, *, required: bool = True) -> dict[str, Any]:
    path = root / relative_path
    return {
        "path": str(path),
        "relative_path": relative_path,
        "present": _exists(path),
        "required": bool(required),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


# 定位 `default advclip root`，把配置值或请求上下文转换成实际文件系统路径。
def _default_advclip_root(project_root: Path) -> Path:
    candidates = [
        project_root / "third_party" / "papers" / "AdvCLIP",
        project_root / "third_party" / "AdvCLIP_official",
        project_root / "AdvCLIP",
        project_root.parent / "new-ATT" / "AdvCLIP",
    ]
    for candidate in candidates:
        if (candidate / "advclip.py").exists():
            return candidate
    return candidates[0]


# 定位 `default tmm root`，把配置值或请求上下文转换成实际文件系统路径。
def _default_tmm_root(project_root: Path) -> Path:
    candidates = [
        project_root / "third_party" / "papers" / "TMM",
        project_root / "TMM-main" / "TMM-main",
        project_root / "TMM-main",
        project_root.parent / "new-ATT" / "TMM-main" / "TMM-main",
    ]
    for candidate in candidates:
        if (candidate / "EvalTransferAttack.py").exists():
            return candidate
    return candidates[0]


# 执行 `git commit` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _git_commit(path: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# 执行 `python compile 探测` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _python_compile_probe(files: list[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for file in files:
        if not file.exists():
            results.append({"path": str(file), "present": False, "compiled": False, "error": "missing"})
            continue
        proc = subprocess.run(
            ["python", "-m", "py_compile", str(file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        results.append(
            {
                "path": str(file),
                "present": True,
                "compiled": proc.returncode == 0,
                "error": proc.stderr.strip(),
            }
        )
    return results


# 读取 `文本 if exists`，并对缺失或异常输入做边界处理。
def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


# 执行 `contains` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _contains(path: Path, needle: str) -> bool:
    return needle in _read_text_if_exists(path)


# 判断或归一 `advclip code and 数据集 状态` 状态，让调用方可以稳定渲染能力和可用性。
def _advclip_code_and_dataset_status(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    code_files = [
        "advclip.py",
        "train_downstream_cross.py",
        "train_downstream_solo.py",
        "test_downstream_task.py",
        "utils/load_data.py",
        "utils/model.py",
        "utils/evaluate.py",
        "utils/nce.py",
        "utils/patch_utils.py",
    ]
    code_status = [_file_status(root, item) for item in code_files]

    dataset_status = []
    for dataset in ADVCLIP_RETRIEVAL_DATASETS + ADVCLIP_CLASSIFICATION_DATASETS:
        dataset_status.append(_file_status(root, f"data/{dataset}/train.pkl"))
        dataset_status.append(_file_status(root, f"data/{dataset}/test.pkl"))
    return code_status, dataset_status


# 执行 `advclip trained 产物` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _advclip_trained_artifacts(root: Path) -> list[dict[str, Any]]:
    trained_artifacts: list[dict[str, Any]] = []
    for victim in ADVCLIP_VICTIMS:
        victim_dir = ADVCLIP_VICTIM_DIR[victim]
        for surrogate in ADVCLIP_RETRIEVAL_DATASETS:
            trained_artifacts.append(
                {
                    "kind": "topology_deviation_gan_uap",
                    "victim": victim,
                    "surrogate_dataset": surrogate,
                    "expected_dir": str(root / "output" / "uap" / "gan_patch" / victim_dir / surrogate / "0.03"),
                    "present": (root / "output" / "uap" / "gan_patch" / victim_dir / surrogate / "0.03").exists(),
                }
            )
        for dataset in ADVCLIP_RETRIEVAL_DATASETS:
            trained_artifacts.append(
                {
                    "kind": "cross_modal_downstream_model",
                    "victim": victim,
                    "dataset": dataset,
                    "expected_dir": str(root / "output" / "model" / victim / dataset),
                    "present": (root / "output" / "model" / victim / dataset).exists(),
                }
            )
        for dataset in ADVCLIP_CLASSIFICATION_DATASETS:
            trained_artifacts.append(
                {
                    "kind": "classification_downstream_model",
                    "victim": victim,
                    "dataset": dataset,
                    "expected_dir": str(root / "output" / "solo_model" / victim / dataset),
                    "present": (root / "output" / "solo_model" / victim / dataset).exists(),
                }
            )
    return trained_artifacts


# 执行 `advclip source findings` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _advclip_source_findings(root: Path) -> list[str]:
    source_findings = []
    if _contains(root / "advclip.py", 'default="cuda:1"'):
        source_findings.append("advclip.py defaults to cuda:1; AutoDL server has one RTX 4090, so strict runs must pass --device cuda:0 or apply a compatibility patch.")
    if _contains(root / "train_downstream_cross.py", 'default="cuda:1"'):
        source_findings.append("train_downstream_cross.py defaults to cuda:1; strict runs must pass --device cuda:0.")
    solo_source = _read_text_if_exists(root / "train_downstream_solo.py")
    if "str(victim_name)" in solo_source and "victim_name =" not in solo_source:
        source_findings.append("train_downstream_solo.py saves with undefined victim_name; the official file needs a compatibility patch before solo classification training.")
    if _contains(root / "train_downstream_solo.py", "choices=['nus-wide', 'pascal', 'wikipedia', 'xmedianet']"):
        source_findings.append("train_downstream_solo.py argparse choices omit STL10/GTSRB/CIFAR10/ImageNet although the paper/test script requires them.")
    if _contains(root / "utils" / "load_data.py", "class CustomDataSet(Dataset)") and not _contains(
        root / "utils" / "load_data.py", "from torch.utils.data import Dataset"
    ):
        source_findings.append("utils/load_data.py subclasses Dataset without importing Dataset; py_compile succeeds but runtime class definition fails.")
    return source_findings


# 执行 `audit advclip` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def audit_advclip(root: Path) -> dict[str, Any]:
    code_status, dataset_status = _advclip_code_and_dataset_status(root)
    trained_artifacts = _advclip_trained_artifacts(root)
    source_findings = _advclip_source_findings(root)
    missing_required = [item for item in code_status + dataset_status if item["required"] and not item["present"]]
    missing_artifacts = [item for item in trained_artifacts if not item["present"]]
    status = "ready" if not missing_required and not missing_artifacts and not source_findings else "blocked"

    return {
        "paper": "AdvCLIP",
        "status": status,
        "root": str(root),
        "git_commit": _git_commit(root),
        "required_protocol": {
            "method": "topology-deviation GAN universal adversarial patch",
            "victim_backbones": ADVCLIP_VICTIMS,
            "surrogate_cross_modal_datasets": ADVCLIP_RETRIEVAL_DATASETS,
            "downstream_retrieval_datasets": ADVCLIP_RETRIEVAL_DATASETS,
            "downstream_classification_datasets": ADVCLIP_CLASSIFICATION_DATASETS,
            "training_hyperparameters": {
                "epochs": 20,
                "batch_size": 16,
                "optimizer": "Adam",
                "learning_rate": 0.0002,
                "noise_percentage": 0.03,
                "alpha": 10,
                "beta": 5,
                "delta": 1,
            },
        },
        "code_files": code_status,
        "datasets": dataset_status,
        "trained_artifacts": trained_artifacts,
        "source_findings": source_findings,
        "blockers": {
            "missing_required_files": missing_required,
            "missing_trained_artifacts_count": len(missing_artifacts),
            "source_findings_count": len(source_findings),
        },
    }


# 判断或归一 `tmm code 数据集 checkpoint 状态` 状态，让调用方可以稳定渲染能力和可用性。
def _tmm_code_dataset_checkpoint_status(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    code_files = [
        "EvalTransferAttack.py",
        "attack/__init__.py",
        "attack/imageAttack.py",
        "attack/multimodalAttack.py",
        "attack/textAttack.py",
        "configs/Retrieval_flickr.yaml",
        "configs/Retrieval_coco.yaml",
        "configs/config_bert.json",
        "dataset/caption_dataset.py",
        "models/model_retrieval.py",
    ]
    code_status = [_file_status(root, item) for item in code_files]
    dataset_status = []
    for spec in TMM_VLR_DATASETS.values():
        dataset_status.append(_file_status(root, spec["annotation"]))
        dataset_status.append(_file_status(root, spec["image_root"]))
    dataset_status.extend(
        [
            _file_status(root, "datasets/refcoco+_test.json"),
            _file_status(root, "datasets/refcoco+_val.json"),
            _file_status(root, "datasets/snli_ve_test.json"),
        ]
    )

    checkpoint_status = []
    for dataset in ("flickr30k", "mscoco"):
        checkpoint_status.extend(
            [
                _file_status(root, f"checkpoints/ALBEF/{dataset}.pth"),
                _file_status(root, f"checkpoints/TCL/{dataset}.pth"),
                _file_status(root, f"checkpoints/X-VLM/{dataset}.pth"),
                _file_status(root, f"checkpoints/ViLT/{dataset}.ckpt"),
                _file_status(root, f"checkpoints/METER/{dataset}.ckpt"),
                _file_status(root, f"checkpoints/BLIP/{dataset}.pth"),
            ]
        )
    return code_status, dataset_status, checkpoint_status


# 执行 `tmm source findings` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _tmm_source_findings(root: Path) -> list[str]:
    source_findings = []
    eval_file = root / "EvalTransferAttack.py"
    multimodal_file = root / "attack" / "multimodalAttack.py"
    dataset_file = root / "dataset" / "caption_dataset.py"
    if _contains(eval_file, "eval_handler.itm_eval") and not _contains(eval_file, "def itm_eval"):
        source_findings.append("EvalTransferAttack.py calls eval_handler.itm_eval, but the Evaluation class does not define itm_eval.")
    if _contains(eval_file, "self.retrieval_score(self.model, image_feats"):
        source_findings.append("EvalTransferAttack.py passes an extra self.model argument into retrieval_score.")
    if _contains(eval_file, "for images, texts, texts_ids in"):
        source_findings.append("EvalTransferAttack.py expects 3 dataset fields, while pair_dataset_attack returns 4 fields.")
    if _contains(eval_file, "TextAttacker(self.ref_model, self.tokenizer, cls=args.cls)"):
        source_findings.append("EvalTransferAttack.py omits the required device argument for TextAttacker.")
    if _contains(multimodal_file, "self.image_attacker") and not _contains(multimodal_file, "self.image_attacker ="):
        source_findings.append("multimodalAttack.py uses self.image_attacker without initializing it.")
    if _contains(multimodal_file, "self.image_normalize") and not _contains(multimodal_file, "self.image_normalize ="):
        source_findings.append("multimodalAttack.py uses self.image_normalize without initializing it.")
    if _contains(multimodal_file, "self.ssim") and not _contains(multimodal_file, "self.ssim ="):
        source_findings.append("multimodalAttack.py uses self.ssim without initializing it.")
    if not any((root / item).exists() for item in ("EvalGrounding.py", "EvalVE.py", "Grounding.py", "VE.py")):
        source_findings.append("Downloaded TMM-main repo only exposes a retrieval runner; no VG/VE runner is present.")
    if not any(_contains(eval_file, model) for model in ("X-VLM", "METER", "ViLT", "BLIP")):
        source_findings.append("Downloaded TMM-main repo README links target checkpoints but code does not instantiate X-VLM/ViLT/METER/BLIP target models.")
    return source_findings


# 执行 `audit tmm` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def audit_tmm(root: Path) -> dict[str, Any]:
    code_status, dataset_status, checkpoint_status = _tmm_code_dataset_checkpoint_status(root)
    source_findings = _tmm_source_findings(root)
    missing_required = [item for item in code_status + dataset_status + checkpoint_status if item["required"] and not item["present"]]
    status = "ready" if not missing_required and not source_findings else "blocked"

    return {
        "paper": "TMM",
        "status": status,
        "root": str(root),
        "git_commit": _git_commit(root),
        "required_protocol": {
            "method": "attention-directed feature perturbation plus orthogonal-guided feature heterogenization",
            "tasks": TMM_TASKS,
            "surrogate_model": "ALBEF",
            "black_box_targets": TMM_TARGET_MODELS,
            "vlr_splits": TMM_VLR_DATASETS,
            "metrics": [
                "attack success rate over first/fifth/tenth-rank recall for text-to-image and image-to-text retrieval",
                "visual grounding attack success rate",
                "visual entailment attack success rate",
                "transfer matrix",
            ],
        },
        "code_files": code_status,
        "datasets": dataset_status,
        "checkpoints": checkpoint_status,
        "source_findings": source_findings,
        "blockers": {
            "missing_required_files": missing_required,
            "source_findings_count": len(source_findings),
        },
    }


# 执行 `advclip runbook` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _advclip_runbook(root: Path) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {root.as_posix()!r}",
        "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}",
        "",
        "# 1. Train downstream cross-modal retrieval heads for all paper datasets/backbones.",
        "for victim in 'RN50' 'RN101' 'ViT-B/16' 'ViT-B/32' 'ViT-L/14'; do",
        "  for dataset in 'nus-wide' 'pascal' 'wikipedia' 'xmedianet'; do",
        "    python train_downstream_cross.py --dataset \"$dataset\" --victim \"$victim\" --device cuda:0 --save True --num_epochs 500",
        "  done",
        "done",
        "",
        "# 2. Train topology-deviation GAN UAPs on all surrogate cross-modal datasets.",
        "for victim in 'RN50' 'RN101' 'ViT-B/16' 'ViT-B/32' 'ViT-L/14'; do",
        "  for dataset in 'nus-wide' 'pascal' 'wikipedia' 'xmedianet'; do",
        "    python advclip.py --dataset \"$dataset\" --victim \"$victim\" --device cuda:0 --num_epochs 20 --batch_size 16 --alpha 10 --beta 5 --delta 1 --noise_percentage 0.03 --save True",
        "  done",
        "done",
        "",
        "# 3. Train solo classifiers for STL10/GTSRB/CIFAR10/ImageNet after applying the compatibility patch.",
        "for victim in 'RN50' 'RN101' 'ViT-B/16' 'ViT-B/32' 'ViT-L/14'; do",
        "  for dataset in 'stl10' 'gtsrb' 'cifar10' 'imagenet'; do",
        "    python train_downstream_solo.py --dataset \"$dataset\" --victim \"$victim\" --device cuda:0 --save True --epochs 20",
        "  done",
        "done",
        "",
        "# 4. Evaluate all retrieval and classification transfer combinations.",
        "for victim in 'RN50' 'RN101' 'ViT-B/16' 'ViT-B/32' 'ViT-L/14'; do",
        "  for sup in 'nus-wide' 'pascal' 'wikipedia' 'xmedianet'; do",
        "    for dataset in 'nus-wide' 'pascal' 'wikipedia' 'xmedianet'; do",
        "      python test_downstream_task.py --down_type cross --dataset \"$dataset\" --sup_dataset \"$sup\" --victim \"$victim\" --device cuda:0 --save True",
        "    done",
        "    for dataset in 'stl10' 'gtsrb' 'cifar10' 'imagenet'; do",
        "      python test_downstream_task.py --down_type solo --dataset \"$dataset\" --sup_dataset \"$sup\" --victim \"$victim\" --device cuda:0 --save True",
        "    done",
        "  done",
        "done",
        "",
    ]
    return "\n".join(lines)


# 执行 `tmm runbook` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _tmm_runbook(root: Path) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {root.as_posix()!r}",
        "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}",
        "",
        "# This TMM-main checkout contains only an ALBEF retrieval entrypoint.",
        "# VG/VE and X-VLM/ViLT/METER/BLIP target-model runners must be added before this can pass the strict paper gate.",
        "python EvalTransferAttack.py --adv 1 --gpu 0 --dataset flickr \\",
        "  --config ./configs/Retrieval_flickr.yaml \\",
        "  --output_dir ./output/Retrieval_flickr \\",
        "  --checkpoint ./checkpoints/ALBEF/flickr30k.pth \\",
        "  --log_name tmm_flickr_albef.jsonl \\",
        "  --save_json_name ./output/Retrieval_flickr/adv_examples.json \\",
        "  --config_name config.yaml \\",
        "  --save_dir ./output/Retrieval_flickr",
        "",
        "python EvalTransferAttack.py --adv 1 --gpu 0 --dataset mscoco \\",
        "  --config ./configs/Retrieval_coco.yaml \\",
        "  --output_dir ./output/Retrieval_coco \\",
        "  --checkpoint ./checkpoints/ALBEF/mscoco.pth \\",
        "  --log_name tmm_coco_albef.jsonl \\",
        "  --save_json_name ./output/Retrieval_coco/adv_examples.json \\",
        "  --config_name config.yaml \\",
        "  --save_dir ./output/Retrieval_coco",
        "",
    ]
    return "\n".join(lines)


# 写出 `报告`，保证后续报告、页面或复现实验能读取。
def _write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Strict Paper Reproduction Audit",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- overall_status: `{payload['status']}`",
        f"- host: `{payload['environment']['host']}`",
        f"- python: `{payload['environment']['python']}`",
        "",
        "## AdvCLIP",
        "",
        f"- status: `{payload['advclip']['status']}`",
        f"- root: `{payload['advclip']['root']}`",
        f"- missing required files: `{len(payload['advclip']['blockers']['missing_required_files'])}`",
        f"- missing trained artifact groups: `{payload['advclip']['blockers']['missing_trained_artifacts_count']}`",
        f"- source findings: `{payload['advclip']['blockers']['source_findings_count']}`",
        "",
    ]
    for finding in payload["advclip"]["source_findings"]:
        lines.append(f"- {finding}")
    lines.extend(
        [
            "",
            "## TMM",
            "",
            f"- status: `{payload['tmm']['status']}`",
            f"- root: `{payload['tmm']['root']}`",
            f"- missing required files: `{len(payload['tmm']['blockers']['missing_required_files'])}`",
            f"- source findings: `{payload['tmm']['blockers']['source_findings_count']}`",
            "",
        ]
    )
    for finding in payload["tmm"]["source_findings"]:
        lines.append(f"- {finding}")
    lines.extend(
        [
            "",
            "## Required Position",
            "",
            "This audit is intentionally stricter than the engineering MVP. It only reports `ready` when the original-paper assets, official-code entrypoints, downstream checkpoints, and full transfer matrix prerequisites are present. Until then, the correct thesis wording is `paper-inspired engineering evaluation`, not `complete original-paper reproduction`.",
            "",
            "## Generated Runbooks",
            "",
            "- `advclip_official_matrix.sh`",
            "- `tmm_official_matrix.sh`",
            "",
        ]
    )
    (out_dir / "strict_paper_reproduction_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# 作为 `audit_strict_paper_protocol.py` 的执行入口，串联参数读取、核心处理和退出状态。
def main() -> int:
    parser = argparse.ArgumentParser(description="Audit strict original-paper reproduction prerequisites for AdvCLIP and TMM.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--advclip-root", default="")
    parser.add_argument("--tmm-root", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--compile-probe", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    advclip_root = Path(args.advclip_root).resolve() if args.advclip_root else _default_advclip_root(project_root).resolve()
    tmm_root = Path(args.tmm_root).resolve() if args.tmm_root else _default_tmm_root(project_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else project_root / "artifacts" / f"strict_paper_protocol_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    advclip = audit_advclip(advclip_root)
    tmm = audit_tmm(tmm_root)
    compile_probe = []
    if args.compile_probe:
        compile_probe = _python_compile_probe(
            [
                advclip_root / "advclip.py",
                advclip_root / "train_downstream_cross.py",
                advclip_root / "train_downstream_solo.py",
                advclip_root / "test_downstream_task.py",
                tmm_root / "EvalTransferAttack.py",
                tmm_root / "attack" / "multimodalAttack.py",
            ]
        )

    status = "ready" if advclip["status"] == "ready" and tmm["status"] == "ready" else "blocked"
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "project_root": str(project_root),
        "environment": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cwd": os.getcwd(),
        },
        "advclip": advclip,
        "tmm": tmm,
        "compile_probe": compile_probe,
    }
    _write_json(out_dir / "strict_paper_reproduction_audit.json", payload)
    (out_dir / "advclip_official_matrix.sh").write_text(_advclip_runbook(advclip_root), encoding="utf-8")
    (out_dir / "tmm_official_matrix.sh").write_text(_tmm_runbook(tmm_root), encoding="utf-8")
    _write_report(out_dir, payload)

    print(
        json.dumps(
            {
                "status": status,
                "out_dir": str(out_dir),
                "advclip_status": advclip["status"],
                "tmm_status": tmm["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if status == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
