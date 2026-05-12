# 文件说明：该文件属于攻击算法公共层，集中实现 text utils 相关逻辑。
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from mmsec_eval.model_adapters.hf_local import resolve_hf_model_source


_BERT_CACHE: dict[str, Any] = {}


# 定义 `_MlmEditMode` 的状态和行为边界，供攻击算法公共层在固定职责内复用。
@dataclass(frozen=True)
class _MlmEditMode:
    method: str
    budget_key: str
    mask_delta_key: str
    minimize_score: bool


# 加载 `bert mlm`，把外部文件、配置或运行产物转换为内存结构。
def _load_bert_mlm(device: str):
    key = f"bert:{device}"
    if key in _BERT_CACHE:
        return _BERT_CACHE[key]

    from transformers import BertForMaskedLM, BertTokenizerFast

    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    local_only = os.getenv("MMSEC_HF_LOCAL_ONLY", "1").strip().lower() not in {"0", "false", "no"}
    model_name = os.getenv("MMSEC_BERT_MLM_MODEL_NAME", "bert-base-uncased")
    source = resolve_hf_model_source(model_name, local_only=local_only, local_dir_name="bert_mlm")
    tokenizer = BertTokenizerFast.from_pretrained(source, local_files_only=local_only)
    model = BertForMaskedLM.from_pretrained(source, local_files_only=local_only).to(device)
    model.eval()
    _BERT_CACHE[key] = (tokenizer, model)
    return tokenizer, model


# 计算 `文本`，为指标、风险或调度决策提供数值依据。
def _score_text(adapter: Any, image: np.ndarray, text: str) -> float:
    return float(adapter.score_pairs([(image, str(text))], batch_size=1)[0])


# 执行 `tokens 来源 文本` 辅助逻辑，保持攻击算法公共层中的输入处理和结果输出一致。
def _tokens_from_text(text: str) -> list[str]:
    return [tok for tok in str(text).split() if tok]


# 执行 `noop 文本 result` 辅助逻辑，保持攻击算法公共层中的输入处理和结果输出一致。
def _noop_text_result(*, image: np.ndarray, text: str, adapter: Any, reason: str) -> tuple[str, dict[str, Any]]:
    score = float(_score_text(adapter, image, text))
    return str(text), {"method": "noop", "reason": reason, "score_orig": score, "score_new": score}


# 执行 `greedy token drop` 辅助逻辑，保持攻击算法公共层中的输入处理和结果输出一致。
def _greedy_token_drop(
    *,
    image: np.ndarray,
    text: str,
    adapter: Any,
    budget: int,
    budget_key: str,
    method: str,
    no_op_reason: str,
    improves: Callable[[float, float], bool],
) -> tuple[str, dict[str, Any]]:
    score_orig = _score_text(adapter, image, text)
    if budget <= 0:
        return str(text), {
            "method": "noop",
            "reason": no_op_reason,
            "score_orig": float(score_orig),
            "score_new": float(score_orig),
        }

    tokens = _tokens_from_text(text)
    if len(tokens) <= 1:
        return str(text), {
            "method": method,
            "reason": "too_few_tokens",
            "score_orig": float(score_orig),
            "score_new": float(score_orig),
            "num_edits": 0,
            "edits": [],
        }

    current_tokens = list(tokens)
    edits: list[dict[str, Any]] = []

    for _ in range(int(max(1, budget))):
        if len(current_tokens) <= 1:
            break
        base_text = " ".join(current_tokens)
        base_score = _score_text(adapter, image, base_text)
        best_idx = -1
        best_score = base_score
        for idx in range(len(current_tokens)):
            cand_tokens = current_tokens[:idx] + current_tokens[idx + 1 :]
            cand_text = " ".join(cand_tokens).strip()
            if not cand_text:
                continue
            cand_score = _score_text(adapter, image, cand_text)
            if improves(cand_score, best_score):
                best_score = cand_score
                best_idx = idx
        if best_idx < 0:
            break
        removed = current_tokens.pop(best_idx)
        edits.append(
            {
                "op": "drop",
                "old_token": str(removed),
                "score_before": float(base_score),
                "score_after": float(best_score),
            }
        )

    new_text = " ".join(current_tokens).strip() or str(text)
    return new_text, {
        "method": method,
        budget_key: int(budget),
        "num_edits": int(len(edits)),
        "edits": edits,
        "score_orig": float(score_orig),
        "score_new": float(_score_text(adapter, image, new_text)),
    }


# 推断 `fallback token drop 攻击`，从样本、配置或运行记录中提取统一名称。
def _fallback_token_drop_attack(
    *,
    image: np.ndarray,
    text: str,
    adapter: Any,
    eps_t: int,
) -> tuple[str, dict[str, Any]]:
    return _greedy_token_drop(
        image=image,
        text=text,
        adapter=adapter,
        budget=int(eps_t),
        budget_key="eps_t",
        method="token_drop_fallback",
        no_op_reason="eps_t<=0",
        improves=lambda candidate, best: candidate < best,
    )


# 执行 `fallback token drop repair` 辅助逻辑，保持攻击算法公共层中的输入处理和结果输出一致。
def _fallback_token_drop_repair(
    *,
    image: np.ndarray,
    text: str,
    adapter: Any,
    max_edits: int,
) -> tuple[str, dict[str, Any]]:
    return _greedy_token_drop(
        image=image,
        text=text,
        adapter=adapter,
        budget=int(max_edits),
        budget_key="max_edits",
        method="token_drop_repair",
        no_op_reason="max_edits<=0",
        improves=lambda candidate, best: candidate > best,
    )


# 计算 `是否 better`，为指标、风险或调度决策提供数值依据。
def _score_is_better(candidate: float, best: float, *, minimize_score: bool) -> bool:
    return candidate < best if minimize_score else candidate > best


# 筛选 `mask position`，按配置条件保留可用于评测或展示的数据。
def _select_mask_position(
    *,
    tokenizer: Any,
    input_ids: Any,
    image: np.ndarray,
    adapter: Any,
    current_text: str,
    base_score: float,
    minimize_score: bool,
) -> tuple[int, float]:
    cand_pos = [i for i in range(1, int(input_ids.shape[1]) - 1)]
    if not cand_pos:
        return -1, 0.0

    best_pos = int(cand_pos[0]) if minimize_score else -1
    best_delta = -1e9 if minimize_score else 0.0
    mask_id = int(tokenizer.mask_token_id)
    for pos in cand_pos[: min(24, len(cand_pos))]:
        ids_masked = input_ids.clone()
        ids_masked[0, pos] = mask_id
        masked_text = tokenizer.decode(ids_masked[0], skip_special_tokens=True)
        masked_score = _score_text(adapter, image, masked_text)
        delta = base_score - masked_score if minimize_score else masked_score - base_score
        if delta > best_delta:
            best_delta = float(delta)
            best_pos = int(pos)
    return best_pos, float(best_delta)


# 执行 `encode mlm inputs` 辅助逻辑，保持攻击算法公共层中的输入处理和结果输出一致。
def _encode_mlm_inputs(*, tokenizer: Any, text: str, device: str) -> tuple[Any, Any]:
    enc = tokenizer(text, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# 执行 `candidate token ids` 辅助逻辑，保持攻击算法公共层中的输入处理和结果输出一致。
def _candidate_token_ids(*, torch_mod: Any, mlm: Any, input_ids: Any, attention_mask: Any, best_pos: int, candidates_k: int) -> list[int]:
    with torch_mod.no_grad():
        out = mlm(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0, best_pos]
        top_ids = torch_mod.topk(logits, k=min(int(candidates_k), int(logits.shape[0]))).indices.tolist()
    return [int(token_id) for token_id in top_ids]


# 执行 `choose best replacement` 辅助逻辑，保持攻击算法公共层中的输入处理和结果输出一致。
def _choose_best_replacement(
    *,
    tokenizer: Any,
    ids_masked: Any,
    old_id: int,
    best_pos: int,
    top_ids: list[int],
    image: np.ndarray,
    adapter: Any,
    current_text: str,
    base_score: float,
    minimize_score: bool,
) -> tuple[str, float, str]:
    best_text = current_text
    best_score = float(base_score)
    best_token = ""
    for token_id in top_ids:
        if int(token_id) == old_id:
            continue
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        if not token or token.startswith("[") or token.startswith("##"):
            continue
        ids_candidate = ids_masked.clone()
        ids_candidate[0, best_pos] = int(token_id)
        cand_text = tokenizer.decode(ids_candidate[0], skip_special_tokens=True)
        cand_score = _score_text(adapter, image, cand_text)
        if _score_is_better(cand_score, best_score, minimize_score=minimize_score):
            best_score = float(cand_score)
            best_text = cand_text
            best_token = str(token)
    return best_text, best_score, best_token


# 执行 `mlm edit result` 辅助逻辑，保持攻击算法公共层中的输入处理和结果输出一致。
def _mlm_edit_result(
    *,
    mode: _MlmEditMode,
    budget: int,
    edits: list[dict[str, Any]],
    score_orig: float,
    score_new: float,
) -> dict[str, Any]:
    return {
        "method": mode.method,
        mode.budget_key: int(budget),
        "num_edits": int(len(edits)),
        "edits": edits,
        "score_orig": float(score_orig),
        "score_new": float(score_new),
    }


# 确认 `mlm edit record` 是字典记录，避免后续字段读取直接接触异常类型。
def _mlm_edit_record(
    *,
    mode: _MlmEditMode,
    best_pos: int,
    old_token: str,
    new_token: str,
    score_before: float,
    score_after: float,
    mask_delta: float,
) -> dict[str, Any]:
    return {
        "op": "replace",
        "pos": int(best_pos),
        "old_token": str(old_token),
        "new_token": str(new_token),
        "score_before": float(score_before),
        "score_after": float(score_after),
        mode.mask_delta_key: float(mask_delta),
    }


# 执行 `mlm 文本 edit` 流程，按配置驱动攻击算法公共层完成一次任务。
def _run_mlm_text_edit(
    *,
    image: np.ndarray,
    text: str,
    adapter: Any,
    budget: int,
    candidates_k: int,
    mode: _MlmEditMode,
) -> tuple[str, dict[str, Any]]:
    import torch

    device = getattr(adapter, "_device", "cpu")
    tokenizer, mlm = _load_bert_mlm(device)
    current_text = str(text)
    score_orig = _score_text(adapter, image, current_text)
    edits: list[dict[str, Any]] = []
    for _ in range(int(max(1, budget))):
        base_score = _score_text(adapter, image, current_text)
        input_ids, attention_mask = _encode_mlm_inputs(tokenizer=tokenizer, text=current_text, device=device)
        best_pos, mask_delta = _select_mask_position(
            tokenizer=tokenizer,
            input_ids=input_ids,
            image=image,
            adapter=adapter,
            current_text=current_text,
            base_score=base_score,
            minimize_score=mode.minimize_score,
        )
        if best_pos < 0:
            break
        ids_masked = input_ids.clone()
        ids_masked[0, best_pos] = int(tokenizer.mask_token_id)
        top_ids = _candidate_token_ids(
            torch_mod=torch,
            mlm=mlm,
            input_ids=ids_masked,
            attention_mask=attention_mask,
            best_pos=best_pos,
            candidates_k=candidates_k,
        )
        old_id = int(input_ids[0, best_pos].item())
        old_token = tokenizer.convert_ids_to_tokens(old_id)
        next_text, next_score, new_token = _choose_best_replacement(
            tokenizer=tokenizer,
            ids_masked=ids_masked,
            old_id=old_id,
            best_pos=best_pos,
            top_ids=top_ids,
            image=image,
            adapter=adapter,
            current_text=current_text,
            base_score=base_score,
            minimize_score=mode.minimize_score,
        )

        if next_text == current_text:
            break

        edits.append(
            _mlm_edit_record(
                mode=mode,
                best_pos=best_pos,
                old_token=old_token,
                new_token=new_token,
                score_before=base_score,
                score_after=next_score,
                mask_delta=mask_delta,
            )
        )
        current_text = next_text

    return current_text, _mlm_edit_result(
        mode=mode,
        budget=budget,
        edits=edits,
        score_orig=score_orig,
        score_new=_score_text(adapter, image, current_text),
    )


# 执行 `文本 replacement 攻击` 流程，按配置驱动攻击算法公共层完成一次任务。
def run_text_replacement_attack(
    *,
    image: np.ndarray,
    text: str,
    adapter: Any,
    eps_t: int,
    candidates_k: int,
    prefer_mlm: bool = True,
) -> tuple[str, dict[str, Any]]:
    if eps_t <= 0:
        return _noop_text_result(image=image, text=text, adapter=adapter, reason="eps_t<=0")
    if not hasattr(adapter, "score_pairs"):
        raise RuntimeError("text attack requires adapter.score_pairs")

    if prefer_mlm:
        try:
            current_text, result = _run_mlm_text_edit(
                image=image,
                text=text,
                adapter=adapter,
                budget=eps_t,
                candidates_k=candidates_k,
                mode=_MlmEditMode(
                    method="bert_mlm",
                    budget_key="eps_t",
                    mask_delta_key="drop_by_mask",
                    minimize_score=True,
                ),
            )
            if int(result["num_edits"]) <= 0:
                fallback_text, fallback_debug = _fallback_token_drop_attack(image=image, text=text, adapter=adapter, eps_t=eps_t)
                fallback_debug["fallback_reason"] = "bert_mlm_no_improvement"
                return fallback_text, fallback_debug
            return current_text, result
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            fallback_text, fallback_debug = _fallback_token_drop_attack(image=image, text=text, adapter=adapter, eps_t=eps_t)
            fallback_debug["fallback_reason"] = str(exc)
            return fallback_text, fallback_debug

    return _fallback_token_drop_attack(image=image, text=text, adapter=adapter, eps_t=eps_t)


# 执行 `文本 repair` 流程，按配置驱动攻击算法公共层完成一次任务。
def run_text_repair(
    *,
    image: np.ndarray,
    text: str,
    adapter: Any,
    max_edits: int,
    candidates_k: int,
    prefer_mlm: bool = True,
) -> tuple[str, dict[str, Any]]:
    if max_edits <= 0:
        return _noop_text_result(image=image, text=text, adapter=adapter, reason="max_edits<=0")
    if not hasattr(adapter, "score_pairs"):
        raise RuntimeError("text repair requires adapter.score_pairs")

    if prefer_mlm:
        try:
            current_text, result = _run_mlm_text_edit(
                image=image,
                text=text,
                adapter=adapter,
                budget=max_edits,
                candidates_k=candidates_k,
                mode=_MlmEditMode(
                    method="bert_mlm_repair",
                    budget_key="max_edits",
                    mask_delta_key="gain_by_mask",
                    minimize_score=False,
                ),
            )
            if int(result["num_edits"]) <= 0:
                fallback_text, fallback_debug = _fallback_token_drop_repair(
                    image=image,
                    text=text,
                    adapter=adapter,
                    max_edits=max_edits,
                )
                fallback_debug["fallback_reason"] = "bert_mlm_no_improvement"
                return fallback_text, fallback_debug
            return current_text, result
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            fallback_text, fallback_debug = _fallback_token_drop_repair(
                image=image,
                text=text,
                adapter=adapter,
                max_edits=max_edits,
            )
            fallback_debug["fallback_reason"] = str(exc)
            return fallback_text, fallback_debug

    return _fallback_token_drop_repair(image=image, text=text, adapter=adapter, max_edits=max_edits)
