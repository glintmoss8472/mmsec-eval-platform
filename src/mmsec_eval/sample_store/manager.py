from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from mmsec_eval.sample_store.schema import AdversarialAsset, CaseBundle, SampleAsset
from mmsec_eval.sample_store.serializer import save_image_png, write_json, write_jsonl
from mmsec_eval.types import EvalRecord, ModelOutput, Sample


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


class SampleStoreManager:
    def __init__(
        self,
        run_dir: str,
        save_images: bool = True,
        save_traces: bool = True,
        dataset_tag: str = "",
        model_tag: str = "",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.cases_dir = self.run_dir / "cases"
        self.index_path = self.run_dir / "cases_index.jsonl"
        self.save_images = save_images
        self.save_traces = save_traces
        self.dataset_tag = dataset_tag
        self.model_tag = model_tag
        self._index_rows: list[dict[str, Any]] = []

    def _validate_case_bundle(self, bundle: CaseBundle) -> None:
        if not bundle.sample.sample_id:
            raise ValueError("case bundle missing sample.sample_id")
        if "clean" not in bundle.outputs or "adv" not in bundle.outputs:
            raise ValueError("case bundle outputs must include clean and adv")
        if bundle.adversarial.perturbation_l2 < 0:
            raise ValueError("case bundle adversarial.perturbation_l2 must be >= 0")
        if not bundle.dataset_tag:
            raise ValueError("case bundle missing dataset_tag")
        if not bundle.model_tag:
            raise ValueError("case bundle missing model_tag")

    def persist_record(
        self,
        record: EvalRecord,
        *,
        defended_sample: Sample | None = None,
        pred_defended: ModelOutput | None = None,
        defense_refs: dict[str, str] | None = None,
        defense_gain_sample: float | None = None,
    ) -> dict[str, str]:
        sid = record.sample.sample_id
        out_dir = self.cases_dir / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        refs: dict[str, str] = {}
        if self.save_images:
            refs["clean_image"] = save_image_png(str(out_dir / "clean.png"), record.sample.image)
            refs["adv_image"] = save_image_png(str(out_dir / "adv.png"), record.attacked.sample.image)
            if defended_sample is not None:
                refs["defended_image"] = save_image_png(str(out_dir / "defended.png"), defended_sample.image)

        if self.save_traces and record.attacked.attack_trace:
            trace_rows = [
                {
                    "step": t.step,
                    "loss_total": t.loss_total,
                    "loss_parts": t.loss_parts,
                    "metadata": t.metadata,
                }
                for t in record.attacked.attack_trace
            ]
            refs["attack_trace"] = write_jsonl(str(out_dir / "attack_trace.jsonl"), trace_rows)
        if defense_refs:
            refs.update(defense_refs)

        outputs = {
            "clean": _json_ready(asdict(record.pred_clean)),
            "adv": _json_ready(asdict(record.pred_adv)),
        }
        if pred_defended is not None:
            outputs["defended"] = _json_ready(asdict(pred_defended))

        bundle = CaseBundle(
            sample=SampleAsset(
                sample_id=sid,
                text=record.sample.text,
                target_text=record.sample.target_text,
                metadata=dict(record.sample.metadata),
            ),
            adversarial=AdversarialAsset(
                sample_id=sid,
                perturbation_l0=record.attacked.perturbation_l0,
                perturbation_l2=record.attacked.perturbation_l2,
                perturbation_linf=record.attacked.perturbation_linf,
                metadata=dict(record.attacked.metadata),
            ),
            dataset_tag=str(record.sample.metadata.get("dataset", "") or self.dataset_tag or "unknown_dataset"),
            model_tag=str(self.model_tag or "unknown_model"),
            outputs=outputs,
            metrics=dict(record.metrics),
            judge=_json_ready(asdict(record.judge)) if record.judge else {},
            diagnostics=_json_ready(dict(record.diagnostics)),
            artifact_refs=refs,
        )
        self._validate_case_bundle(bundle)
        refs["case_bundle"] = write_json(str(out_dir / "case_bundle.json"), _json_ready(asdict(bundle)))

        index_row = {
            "sample_id": sid,
            "case_dir": str(out_dir),
            "judge_success": bool(record.judge.success) if record.judge else False,
            "perturbation_l2": float(record.attacked.perturbation_l2),
            "perturbation_linf": float(record.attacked.perturbation_linf),
        }
        if defense_gain_sample is not None:
            index_row["defense_gain_sample"] = float(defense_gain_sample)
        self._index_rows.append(index_row)
        return refs

    def flush(self) -> str:
        return write_jsonl(str(self.index_path), self._index_rows)
