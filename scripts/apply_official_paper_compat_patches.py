# 文件说明：该文件属于运维与实验脚本，集中实现 apply official paper compat patches 相关逻辑。
from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# 中文注释：封装 _default_advclip_root 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _default_advclip_root(project_root: Path) -> Path:
    for candidate in (
        project_root / "third_party" / "papers" / "AdvCLIP",
        project_root / "third_party" / "AdvCLIP_official",
        project_root / "AdvCLIP",
        project_root.parent / "new-ATT" / "AdvCLIP",
    ):
        if (candidate / "advclip.py").exists():
            return candidate
    return project_root / "third_party" / "papers" / "AdvCLIP"


# 中文注释：封装 _default_tmm_root 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _default_tmm_root(project_root: Path) -> Path:
    for candidate in (
        project_root / "third_party" / "papers" / "TMM",
        project_root / "TMM-main" / "TMM-main",
        project_root / "TMM-main",
        project_root.parent / "new-ATT" / "TMM-main" / "TMM-main",
    ):
        if (candidate / "EvalTransferAttack.py").exists():
            return candidate
    return project_root / "third_party" / "papers" / "TMM"


# 中文注释：封装 _replace 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _replace(path: Path, old: str, new: str, changes: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        return
    path.write_text(text.replace(old, new), encoding="utf-8")
    changes.append(str(path))


# 中文注释：封装 _insert_after 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _insert_after(path: Path, marker: str, insertion: str, changes: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    if insertion in text or marker not in text:
        return
    path.write_text(text.replace(marker, marker + insertion, 1), encoding="utf-8")
    changes.append(str(path))


# 中文注释：实现 patch_advclip 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
def patch_advclip(root: Path) -> list[str]:
    changes: list[str] = []
    for rel in ("advclip.py", "train_downstream_cross.py", "train_downstream_solo.py"):
        path = root / rel
        if path.exists():
            _replace(path, 'default="cuda:1"', 'default="cuda:0"', changes)

    load_data = root / "utils" / "load_data.py"
    if load_data.exists():
        _insert_after(load_data, "import torch.utils.data as data\n", "from torch.utils.data import Dataset\n", changes)

    solo = root / "train_downstream_solo.py"
    if solo.exists():
        _replace(
            solo,
            "choices=['nus-wide', 'pascal', 'wikipedia', 'xmedianet']",
            "choices=['nus-wide', 'pascal', 'wikipedia', 'xmedianet', 'stl10', 'gtsrb', 'cifar10', 'imagenet']",
            changes,
        )
        _replace(
            solo,
            "def classify(args, encoder, train_loader, test_loader, num_class, feat_dim, device):\n\n    F = NonLinearClassifier",
            "def classify(args, encoder, train_loader, test_loader, num_class, feat_dim, device):\n\n    victim_name = str(args.victim).replace('/', '')\n    F = NonLinearClassifier",
            changes,
        )
        _replace(
            solo,
            "    elif args.dataset == 'nus-wide':\n        num_class = 81\n",
            "    elif args.dataset == 'nus-wide':\n        num_class = 81\n    elif args.dataset == 'stl10':\n        num_class = 10\n    elif args.dataset == 'gtsrb':\n        num_class = 43\n    elif args.dataset == 'cifar10':\n        num_class = 10\n    elif args.dataset == 'imagenet':\n        num_class = 1000\n",
            changes,
        )
    return sorted(set(changes))


# 中文注释：封装 _patch_tmm_eval_file 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _patch_tmm_eval_file(root: Path, changes: list[str]) -> None:
    eval_file = root / "EvalTransferAttack.py"
    if eval_file.exists():
        _replace(
            eval_file,
            "        text_attacker = TextAttacker(self.ref_model, self.tokenizer, cls=args.cls)\n",
            "        text_attacker = TextAttacker(self.ref_model, self.tokenizer, device=self.device, cls=args.cls)\n",
            changes,
        )
        _replace(
            eval_file,
            "        for images, texts, texts_ids in tqdm(self.data_loader, ascii=True):\n",
            "        for images, texts, texts_ids, _image_ids in tqdm(self.data_loader, ascii=True):\n",
            changes,
        )
        _replace(
            eval_file,
            "        score_matrix_i2t, score_matrix_t2i = self.retrieval_score(self.model, image_feats, image_embeds, text_feats,\n",
            "        score_matrix_i2t, score_matrix_t2i = self.retrieval_score(image_feats, image_embeds, text_feats,\n",
            changes,
        )
        _replace(eval_file, "config['k_test']", "self.config['k_test']", changes)
        _insert_after(
            eval_file,
            "        return score_matrix_i2t, score_matrix_t2i\n\n",
            """    @staticmethod
    def itm_eval(scores_i2t, scores_t2i, img2txt, txt2img):
        ranks = {}
        ranks_i2t = []
        for index, score in enumerate(scores_i2t):
            inds = score.argsort()[::-1]
            gt_txt = img2txt[index]
            rank = min([list(inds).index(i) for i in gt_txt])
            ranks_i2t.append(rank)
        ranks_t2i = []
        for index, score in enumerate(scores_t2i):
            inds = score.argsort()[::-1]
            ranks_t2i.append(list(inds).index(txt2img[index]))
        import numpy as np
        ranks_i2t = np.array(ranks_i2t)
        ranks_t2i = np.array(ranks_t2i)
        ranks['txt_r1'] = float(100.0 * len(np.where(ranks_i2t < 1)[0]) / len(ranks_i2t))
        ranks['txt_r5'] = float(100.0 * len(np.where(ranks_i2t < 5)[0]) / len(ranks_i2t))
        ranks['txt_r10'] = float(100.0 * len(np.where(ranks_i2t < 10)[0]) / len(ranks_i2t))
        ranks['img_r1'] = float(100.0 * len(np.where(ranks_t2i < 1)[0]) / len(ranks_t2i))
        ranks['img_r5'] = float(100.0 * len(np.where(ranks_t2i < 5)[0]) / len(ranks_t2i))
        ranks['img_r10'] = float(100.0 * len(np.where(ranks_t2i < 10)[0]) / len(ranks_t2i))
        ranks['r_mean'] = float((ranks['txt_r1'] + ranks['txt_r5'] + ranks['txt_r10'] + ranks['img_r1'] + ranks['img_r5'] + ranks['img_r10']) / 6)
        return ranks

""",
            changes,
        )


# 中文注释：封装 _patch_tmm_image_attack 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _patch_tmm_image_attack(root: Path, changes: list[str]) -> None:
    image_attack = root / "attack" / "imageAttack.py"
    if image_attack.exists():
        _insert_after(
            image_attack,
            "    def gkern(self, kernlen=21, nsig=3):\n        \"\"\"Returns a 2D Gaussian kernel array.\"\"\"\n        x = np.linspace(-nsig, nsig, kernlen)\n        kern1d = st.norm.pdf(x)\n        kernel_raw = np.outer(kern1d, kern1d)\n        kernel = kernel_raw / kernel_raw.sum()\n        return kernel\n",
            "\n    def get_kernel(self, device, kernel_size):\n        kernel = self.gkern(kernel_size, 3).astype(np.float32)\n        kernel = torch.from_numpy(kernel).to(device).view(1, 1, kernel_size, kernel_size)\n        return kernel.repeat(3, 1, 1, 1)\n",
            changes,
        )
        _insert_after(image_attack, "        self.image_processing = ImageProcessing()\n", "        self.preprocess = None\n", changes)
        _replace(
            image_attack,
            "epsilon_att = eosilon_att / ((attMap>self.args.att_mask).sum()/(image.shape[1]*image.shape[2]*image.shape[3])).detach().numpy()",
            "epsilon_att = eosilon_att / ((attMap>self.args.att_mask).sum()/(image.shape[1]*image.shape[2]*image.shape[3])).detach().cpu().numpy()",
            changes,
        )


# 中文注释：封装 _patch_tmm_multimodal 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _patch_tmm_multimodal(root: Path, changes: list[str]) -> None:
    multimodal = root / "attack" / "multimodalAttack.py"
    if multimodal.exists():
        _insert_after(multimodal, "import numpy as np\n", "from .imageAttack import ImageAttacker, images_normalize, ssim\n", changes)
        _insert_after(
            multimodal,
            "        self.cls = cls\n",
            "        self.image_normalize = images_normalize\n        self.ssim = ssim\n",
            changes,
        )
        _replace(
            multimodal,
            "        origin_embeds, text_adv_embed, text_input, origin_output, text_adv_input, text_adv = self.get_origin_and_adv_embeds(images, text, device, max_length)\n",
            "        origin_embeds, text_adv_embed, text_input, origin_output, text_adv_input, text_adv = self.get_origin_and_adv_embeds(images, text, device, max_length, k)\n",
            changes,
        )
        _replace(
            multimodal,
            "            _, _, _, _, text_adv_input, text_adv = self.get_origin_and_adv_embeds(adv, text, device, max_length)\n",
            "            _, _, _, _, text_adv_input, text_adv = self.get_origin_and_adv_embeds(adv, text, device, max_length, k)\n",
            changes,
        )
        _replace(
            multimodal,
            "        _, _, _, _, _, text_adv = self.get_origin_and_adv_embeds(adv, text, device, max_length)\n",
            "        _, _, _, _, _, text_adv = self.get_origin_and_adv_embeds(adv, text, device, max_length, k)\n",
            changes,
        )
        _replace(
            multimodal,
            "        image_attack = self.image_attacker.attack(attImage, attMap, num_iters)\n",
            "        image_attack = ImageAttacker(args.epsilon, args, cls=self.cls).attack(attImage, attMap, num_iters)\n",
            changes,
        )


# 中文注释：实现 patch_tmm 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
def patch_tmm(root: Path) -> list[str]:
    changes: list[str] = []
    _patch_tmm_eval_file(root, changes)
    _patch_tmm_image_attack(root, changes)
    _patch_tmm_multimodal(root, changes)
    return sorted(set(changes))


# 中文注释：串联 main 的主流程，集中处理运维与实验脚本的初始化、执行和退出条件。
def main() -> int:
    parser = argparse.ArgumentParser(description="Apply compatibility patches to downloaded official AdvCLIP/TMM repositories.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--advclip-root", default="")
    parser.add_argument("--tmm-root", default="")
    parser.add_argument("--skip-advclip", action="store_true")
    parser.add_argument("--skip-tmm", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    advclip_root = Path(args.advclip_root).resolve() if args.advclip_root else _default_advclip_root(project_root).resolve()
    tmm_root = Path(args.tmm_root).resolve() if args.tmm_root else _default_tmm_root(project_root).resolve()
    changed: list[str] = []
    if not args.skip_advclip:
        changed.extend(patch_advclip(advclip_root))
    if not args.skip_tmm:
        changed.extend(patch_tmm(tmm_root))
    for path in sorted(set(changed)):
        print(path)
    print(f"patched_files={len(set(changed))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
