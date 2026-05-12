from __future__ import annotations

import argparse
import base64
import binascii
import importlib.util
import logging
import os
import sys
import time
import traceback
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import transformers as hf_transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, GenerationConfig, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.generation.utils import GenerationMixin
from transformers.utils import import_utils as hf_import_utils
from packaging.version import Version


LOGGER = logging.getLogger("mmsec.local_openai_mm_server")

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


if not hasattr(hf_import_utils, "is_torch_fx_available"):
    def _is_torch_fx_available() -> bool:
        try:
            import torch.fx  # noqa: F401

            return True
        except ImportError:
            return False


    hf_import_utils.is_torch_fx_available = _is_torch_fx_available


if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}

if not hasattr(torch.nn.Module, "all_tied_weights_keys"):
    torch.nn.Module.all_tied_weights_keys = {}

if not hasattr(torch.nn.Module, "is_parallelizable"):
    torch.nn.Module.is_parallelizable = False


TRANSFORMERS_VERSION = Version(hf_transformers.__version__)


if DynamicCache is not None and TRANSFORMERS_VERSION >= Version("5.0.0") and not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())


if DynamicCache is not None and TRANSFORMERS_VERSION >= Version("5.0.0") and not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: None


if DynamicCache is not None and TRANSFORMERS_VERSION >= Version("5.0.0") and not hasattr(DynamicCache, "get_usable_length"):
    # Older remote-code VLMs expect this method to report the usable *past* length.
    # In Transformers 5 the cache may already include the prefill length here, so
    # returning get_seq_length() doubles kv_seq_len for Phi-3.5-Vision.
    DynamicCache.get_usable_length = lambda self, *args, **kwargs: 0

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3RotaryEmbedding

    if not hasattr(Qwen3ForCausalLM, "is_parallelizable"):
        Qwen3ForCausalLM.is_parallelizable = False

    _qwen3_rotary_init = Qwen3RotaryEmbedding.__init__

    def _patched_qwen3_rotary_init(self: Any, *args: Any, **kwargs: Any) -> None:
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        if config is not None and getattr(config, "rope_parameters", None) is None:
            config.rope_parameters = {
                "rope_type": "default",
                "rope_theta": float(getattr(config, "rope_theta", 1000000.0) or 1000000.0),
            }
        _qwen3_rotary_init(self, *args, **kwargs)

    Qwen3RotaryEmbedding.__init__ = _patched_qwen3_rotary_init
except (ImportError, AttributeError, TypeError, ValueError) as exc:
    LOGGER.debug("Qwen3 rotary compatibility patch skipped: %s", exc)


def _decode_image_url(url: str) -> Image.Image:
    value = str(url or "").strip()
    if not value:
        raise ValueError("missing image url")
    if value.startswith("data:"):
        try:
            _, payload = value.split(",", 1)
            raw = base64.b64decode(payload)
            return Image.open(BytesIO(raw)).convert("RGB")
        except (ValueError, binascii.Error, OSError) as exc:
            raise ValueError(f"invalid data uri: {exc}") from exc
    if value.startswith("http://") or value.startswith("https://"):
        resp = requests.get(value, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(value).convert("RGB")


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", "user") or "user")
        content = msg.get("content", "")
        parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            if content.strip():
                parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type", "") or "").lower()
                if part_type == "text":
                    text = str(part.get("text", "") or "").strip()
                    if text:
                        parts.append({"type": "text", "text": text})
                    continue
                if part_type in {"image_url", "image"}:
                    payload = part.get("image_url", part.get("image", ""))
                    if isinstance(payload, dict):
                        image_url = str(payload.get("url", "") or "").strip()
                    else:
                        image_url = str(payload or "").strip()
                    if image_url:
                        parts.append({"type": "image", "image": _decode_image_url(image_url)})
        if not parts:
            parts.append({"type": "text", "text": ""})
        normalized.append({"role": role, "content": parts})
    return normalized


def _is_qwen_processor(processor: Any) -> bool:
    marker = f"{processor.__class__.__module__}.{processor.__class__.__name__}".lower()
    return "qwen" in marker


def _apply_chat_template_no_thinking(processor: Any, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    """Apply a chat template while disabling Qwen-style thinking when supported."""
    if not _is_qwen_processor(processor):
        return processor.apply_chat_template(messages, **kwargs)
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        return processor.apply_chat_template(messages, **kwargs)


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    temperature: float | None = 0.0
    max_tokens: int | None = 128


def _load_minicpm_tokenizer(model_id: str, *, local_files_only: bool) -> Any:
    model_path = Path(model_id).expanduser()
    wrapper_cls = None
    if model_path.is_dir():
        wrapper_file = model_path / "modeling_minicpmv.py"
        if wrapper_file.is_file():
            spec = importlib.util.spec_from_file_location("local_minicpmv_wrapper", wrapper_file)
            if spec is not None and spec.loader is not None:
                try:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules.setdefault("local_minicpmv_wrapper", module)
                    spec.loader.exec_module(module)
                    wrapper_cls = getattr(module, "LlamaTokenizerWrapper", None)
                except (AttributeError, ImportError, OSError, RuntimeError):
                    wrapper_cls = None
    if wrapper_cls is not None:
        tokenizer = wrapper_cls.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
            use_fast=False,
        )
    return _patch_minicpm_tokenizer_attrs(tokenizer)


def _patch_minicpm_tokenizer_attrs(tokenizer: Any) -> Any:
    fallback_tokens = {
        "im_start": "<image>",
        "im_end": "</image>",
        "ref_start": "<ref>",
        "ref_end": "</ref>",
        "box_start": "<box>",
        "box_end": "</box>",
        "quad_start": "<quad>",
        "quad_end": "</quad>",
        "slice_start": "<slice>",
        "slice_end": "</slice>",
    }
    for attr, token in fallback_tokens.items():
        if not hasattr(tokenizer, attr):
            setattr(tokenizer, attr, token)
    if not hasattr(tokenizer, "bos_id"):
        tokenizer.bos_id = int(getattr(tokenizer, "bos_token_id", 1) or 1)
    if not hasattr(tokenizer, "eos_id"):
        tokenizer.eos_id = int(getattr(tokenizer, "eos_token_id", 2) or 2)
    if not hasattr(tokenizer, "unk_id"):
        tokenizer.unk_id = int(getattr(tokenizer, "unk_token_id", 0) or 0)
    for attr in fallback_tokens:
        id_attr = f"{attr}_id"
        if not hasattr(tokenizer, id_attr):
            setattr(tokenizer, id_attr, int(tokenizer.convert_tokens_to_ids(getattr(tokenizer, attr))))
    return tokenizer


def _patch_minicpm_runtime(model: Any) -> None:
    if not hasattr(model, "vpm_forward_features"):
        return
    original_vpm_forward_features = model.vpm_forward_features

    def _patched_vpm_forward_features(pixel_value: Any) -> Any:
        features = original_vpm_forward_features(pixel_value)
        try:
            target_dtype = model.llm.lm_head.weight.dtype
            if isinstance(features, torch.Tensor) and features.dtype != target_dtype:
                features = features.to(dtype=target_dtype)
        except (AttributeError, RuntimeError, TypeError) as exc:
            LOGGER.debug("MiniCPM feature dtype patch skipped: %s", exc)
        return features

    model.vpm_forward_features = _patched_vpm_forward_features
    resampler = getattr(model, "resampler", None)
    if resampler is None:
        return
    try:
        target_weight = model.llm.lm_head.weight
        target_dtype = target_weight.dtype
        target_device = target_weight.device
        resampler.to(device=target_device, dtype=target_dtype)
        for tensor_name in ("query", "pos_embed", "proj"):
            tensor = getattr(resampler, tensor_name, None)
            if isinstance(tensor, torch.Tensor) and (tensor.dtype != target_dtype or tensor.device != target_device):
                tensor.data = tensor.data.to(device=target_device, dtype=target_dtype)
    except (AttributeError, RuntimeError, TypeError) as exc:
        LOGGER.debug("MiniCPM resampler device patch skipped: %s", exc)
    llm = getattr(model, "llm", None)
    if llm is not None and hasattr(llm, "prepare_inputs_for_generation"):
        try:
            llm_cls = llm.__class__
            for name, value in GenerationMixin.__dict__.items():
                if name.startswith("__") or hasattr(llm_cls, name):
                    continue
                setattr(llm_cls, name, value)
            if not hasattr(llm, "generation_config"):
                llm.generation_config = GenerationConfig.from_model_config(llm.config)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            LOGGER.debug("MiniCPM generation mixin patch skipped: %s", exc)


def _has_meta_tensors(model: Any) -> bool:
    for getter_name in ("parameters", "buffers"):
        getter = getattr(model, getter_name, None)
        if getter is None:
            continue
        try:
            tensors = getter()
        except (AttributeError, RuntimeError, TypeError):
            continue
        for tensor in tensors:
            device = getattr(tensor, "device", None)
            if device is not None and getattr(device, "type", "") == "meta":
                return True
    return False


def _patch_ovis_model_class(model_id: str) -> Any:
    model_cls = get_class_from_dynamic_module(
        "modeling_ovis2_5.Ovis2_5",
        model_id,
        local_files_only=Path(model_id).expanduser().is_dir(),
        trust_remote_code=True,
    )
    original_tie_weights = getattr(model_cls, "tie_weights", None)
    if original_tie_weights is not None and not getattr(model_cls, "_mmsec_tie_weights_patched", False):
        def _patched_tie_weights(self: Any, *args: Any, **kwargs: Any) -> Any:
            return original_tie_weights(self)

        model_cls.tie_weights = _patched_tie_weights
        model_cls._mmsec_tie_weights_patched = True
    return model_cls


class LocalMultimodalServer:
    @staticmethod
    def _torch_dtype(dtype_name: str) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return dtype_map.get(str(dtype_name).strip().lower(), torch.float16)

    def __init__(self, model_id: str, *, public_model_id: str | None = None, dtype_name: str = "float16") -> None:
        self.model_id = model_id
        self.local_files_only = Path(model_id).expanduser().is_dir()
        self.public_model_id = str(public_model_id or model_id).strip() or model_id
        self.runtime_mode = "default"
        self.tokenizer = None
        is_minicpm, is_phi35_vision = self._name_based_model_flags()
        torch_dtype = self._torch_dtype(dtype_name)

        self.processor = self._load_processor(is_minicpm=is_minicpm, is_phi35_vision=is_phi35_vision)
        config = self._load_config(is_minicpm=is_minicpm, is_phi35_vision=is_phi35_vision)
        model_type = str(getattr(config, "model_type", "") or "").lower()
        is_qwen_vl = model_type in {"qwen2_5_vl", "qwen3_vl"}
        is_ovis = model_type in {"ovis2_5", "ovis2"}
        model_kwargs = self._model_kwargs(config=config, torch_dtype=torch_dtype, is_phi35_vision=is_phi35_vision)
        self._load_model(model_kwargs=model_kwargs, is_minicpm=is_minicpm, is_ovis=is_ovis)
        self._reload_meta_model_if_needed(model_kwargs=model_kwargs, is_minicpm=is_minicpm)
        self.model.eval()
        self._configure_runtime(
            is_minicpm=is_minicpm,
            is_phi35_vision=is_phi35_vision,
            is_qwen_vl=is_qwen_vl,
            is_ovis=is_ovis,
        )

    def _name_based_model_flags(self) -> tuple[bool, bool]:
        labels = f"{self.public_model_id} {self.model_id}".lower()
        is_minicpm = "minicpm" in labels
        is_phi35_vision = "phi-3.5-vision" in labels or "phi35_vision" in labels
        return is_minicpm, is_phi35_vision

    def _load_processor(self, *, is_minicpm: bool, is_phi35_vision: bool) -> Any:
        if is_minicpm:
            return None
        processor_kwargs = {
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
        }
        if is_phi35_vision:
            # Official Phi-3.5-Vision examples use num_crops=4 for practical
            # single-GPU inference; the default can create much longer contexts.
            processor_kwargs["num_crops"] = 4
        return AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)

    def _load_config(self, *, is_minicpm: bool, is_phi35_vision: bool) -> Any:
        config = AutoConfig.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        model_type = str(getattr(config, "model_type", "") or "").lower()
        if is_minicpm and model_type == "minicpmv":
            config.rope_parameters = {
                "rope_type": "default",
                "rope_theta": float(getattr(config, "rope_theta", 1000000.0) or 1000000.0),
            }
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict) and not rope_scaling.get("type"):
            config.rope_scaling = None
        if is_phi35_vision:
            setattr(config, "_attn_implementation", "eager")
            setattr(config, "_attn_implementation_internal", "eager")
        return config

    def _model_kwargs(self, *, config: Any, torch_dtype: torch.dtype, is_phi35_vision: bool) -> dict[str, Any]:
        kwargs = {
            "config": config,
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
        }
        if is_phi35_vision:
            kwargs["attn_implementation"] = "eager"
        return kwargs

    def _load_model(self, *, model_kwargs: dict[str, Any], is_minicpm: bool, is_ovis: bool) -> None:
        if is_minicpm:
            self.loader_name = "AutoModel"
            manual_device_kwargs = dict(model_kwargs)
            manual_device_kwargs.pop("device_map", None)
            self.model = AutoModel.from_pretrained(
                self.model_id,
                **manual_device_kwargs,
            )
            self._move_model_to_primary_device()
        elif is_ovis:
            self.loader_name = "Ovis2_5"
            ovis_cls = _patch_ovis_model_class(self.model_id)
            self.model = ovis_cls.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        else:
            self._load_transformers_vision_or_causal_model(model_kwargs)

    def _load_transformers_vision_or_causal_model(self, model_kwargs: dict[str, Any]) -> None:
        self.loader_name = "AutoModelForImageTextToText"
        try:
            if AutoModelForImageTextToText is None:
                raise RuntimeError("AutoModelForImageTextToText is unavailable")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
            return
        except (ImportError, OSError, RuntimeError, ValueError):
            self.loader_name = "AutoModelForCausalLM"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        except (OSError, RuntimeError, ValueError):
            manual_device_kwargs = dict(model_kwargs)
            manual_device_kwargs.pop("device_map", None)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **manual_device_kwargs,
            )
            self._move_model_to_primary_device()

    def _move_model_to_primary_device(self) -> None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(target_device)

    def _reload_meta_model_if_needed(self, *, model_kwargs: dict[str, Any], is_minicpm: bool) -> None:
        if _has_meta_tensors(self.model) and (is_minicpm or not getattr(self.model, "hf_device_map", None)):
            manual_device_kwargs = dict(model_kwargs)
            manual_device_kwargs.pop("device_map", None)
            if self.loader_name == "AutoModel":
                loader = AutoModel
            elif self.loader_name == "AutoModelForImageTextToText" and AutoModelForImageTextToText is not None:
                loader = AutoModelForImageTextToText
            else:
                loader = AutoModelForCausalLM
            self.model = loader.from_pretrained(
                self.model_id,
                **manual_device_kwargs,
            )
            self._move_model_to_primary_device()

    def _configure_runtime(self, *, is_minicpm: bool, is_phi35_vision: bool, is_qwen_vl: bool, is_ovis: bool) -> None:
        if is_minicpm:
            self.runtime_mode = "minicpm_chat"
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    local_files_only=self.local_files_only,
                )
                if getattr(self.processor, "tokenizer", None) is not None:
                    self.processor.tokenizer = _patch_minicpm_tokenizer_attrs(self.processor.tokenizer)
                self.tokenizer = getattr(self.processor, "tokenizer", None)
            except (ImportError, OSError, RuntimeError, ValueError):
                self.processor = None
                self.tokenizer = None
            if self.tokenizer is None:
                self.tokenizer = _load_minicpm_tokenizer(
                    self.model_id,
                    local_files_only=self.local_files_only,
                )
            _patch_minicpm_runtime(self.model)
        elif is_phi35_vision:
            self.runtime_mode = "phi_vision"
        elif is_qwen_vl and process_vision_info is not None:
            self.runtime_mode = "qwen_vl"
        elif is_ovis:
            self.runtime_mode = "ovis"

    @staticmethod
    def _extract_single_prompt(messages: list[dict[str, Any]]) -> tuple[Image.Image | None, str]:
        prompt_parts: list[str] = []
        prompt_image: Image.Image | None = None
        for msg in messages:
            for part in msg.get("content", []):
                if part.get("type") == "text":
                    text = str(part.get("text", "") or "").strip()
                    if text:
                        prompt_parts.append(text)
                elif part.get("type") == "image" and prompt_image is None:
                    image = part.get("image")
                    if isinstance(image, Image.Image):
                        prompt_image = image
        return prompt_image, "\n".join(prompt_parts).strip()

    def _generate_minicpm(self, prepared: list[dict[str, Any]], max_tokens: int, temperature: float) -> str:
        image, prompt = self._extract_single_prompt(prepared)
        result = self.model.chat(
            image=image,
            msgs=[{"role": "user", "content": prompt or ""}],
            context=None,
            tokenizer=self.tokenizer,
            processor=self.processor,
            sampling=float(temperature or 0.0) > 1e-6,
            temperature=max(float(temperature or 0.0), 0.1),
            max_new_tokens=int(max_tokens or 128),
            num_beams=1,
        )
        return str((result[0] if isinstance(result, tuple) else result) or "").strip()

    def _generate_phi(self, prepared: list[dict[str, Any]], max_tokens: int, temperature: float) -> str:
        image, prompt = self._extract_single_prompt(prepared)
        if image is not None:
            prompt_text = f"<|user|>\n<|image_1|>\n{prompt or ''}<|end|>\n<|assistant|>\n"
            inputs = self.processor(prompt_text, [image], return_tensors="pt").to(self.model.device)
        else:
            prompt_text = f"<|user|>\n{prompt or ''}<|end|>\n<|assistant|>\n"
            inputs = self.processor(prompt_text, images=None, return_tensors="pt").to(self.model.device)
        do_sample = float(temperature or 0.0) > 1e-6
        gen_kwargs = {"max_new_tokens": int(max_tokens or 128), "do_sample": do_sample, "eos_token_id": self.processor.tokenizer.eos_token_id, "use_cache": False}
        if do_sample:
            gen_kwargs["temperature"] = max(float(temperature or 0.0), 0.1)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        texts = self.processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return str(texts[0] if texts else "").strip()

    def _generate_qwen(self, prepared: list[dict[str, Any]], max_tokens: int, temperature: float) -> str:
        prompt_text = _apply_chat_template_no_thinking(self.processor, prepared, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(prepared)
        inputs = self.processor(text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)
        do_sample = float(temperature or 0.0) > 1e-6
        gen_kwargs: dict[str, Any] = {"max_new_tokens": int(max_tokens or 128), "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = max(float(temperature or 0.0), 0.1)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        texts = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return str(texts[0] if texts else "").strip()

    def _generate_ovis(self, prepared: list[dict[str, Any]], max_tokens: int, temperature: float) -> str:
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(messages=prepared, add_generation_prompt=True, enable_thinking=False, max_pixels=896 * 896)
        device = self.model.device
        do_sample = float(temperature or 0.0) > 1e-6
        gen_kwargs: dict[str, Any] = {
            "inputs": input_ids.to(device),
            "pixel_values": pixel_values.to(device) if pixel_values is not None else None,
            "grid_thws": grid_thws.to(device) if grid_thws is not None else None,
            "enable_thinking": False,
            "enable_thinking_budget": False,
            "max_new_tokens": int(max_tokens or 128),
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(float(temperature or 0.0), 0.1)
        with torch.inference_mode():
            output_ids = self.model.generate(**gen_kwargs)
        return str(self.model.text_tokenizer.decode(output_ids[0], skip_special_tokens=True)).strip()

    def _generate_default(self, prepared: list[dict[str, Any]], max_tokens: int, temperature: float) -> str:
        inputs = _apply_chat_template_no_thinking(
            self.processor,
            prepared,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        do_sample = float(temperature or 0.0) > 1e-6
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_tokens or 128),
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(float(temperature or 0.0), 0.1)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        texts = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return str(texts[0] if texts else "").strip()

    def generate(self, messages: list[dict[str, Any]], *, max_tokens: int, temperature: float) -> str:
        prepared = _normalize_messages(messages)
        if self.runtime_mode == "minicpm_chat" and self.tokenizer is not None and hasattr(self.model, "chat"):
            return self._generate_minicpm(prepared, max_tokens, temperature)
        if self.runtime_mode == "phi_vision" and self.processor is not None:
            return self._generate_phi(prepared, max_tokens, temperature)
        if self.runtime_mode == "qwen_vl" and self.processor is not None and process_vision_info is not None:
            return self._generate_qwen(prepared, max_tokens, temperature)
        if self.runtime_mode == "ovis":
            return self._generate_ovis(prepared, max_tokens, temperature)
        return self._generate_default(prepared, max_tokens, temperature)


def build_app(server: LocalMultimodalServer) -> FastAPI:
    app = FastAPI(title="Local Multimodal OpenAI-Compatible Server", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": server.model_id,
            "public_model_id": server.public_model_id,
            "loader": server.loader_name,
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": server.public_model_id,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        requested = str(req.model or "").strip()
        allowed_ids = {server.model_id, server.public_model_id}
        if requested and requested not in allowed_ids:
            raise HTTPException(
                status_code=400,
                detail=f"loaded model is {server.public_model_id} (source={server.model_id}), got {requested}",
            )
        try:
            content = server.generate(
                req.messages,
                max_tokens=int(req.max_tokens or 128),
                temperature=float(req.temperature or 0.0),
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
            print(
                "[local_openai_mm_server] chat completion failed",
                {
                    "model_id": server.model_id,
                    "public_model_id": server.public_model_id,
                    "loader": server.loader_name,
                    "runtime_mode": server.runtime_mode,
                    "requested_model": requested,
                    "message_count": len(req.messages or []),
                    "error": str(exc),
                },
                file=sys.stderr,
                flush=True,
            )
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": server.public_model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    return app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--public-model-id", default=os.getenv("LOCAL_VLM_PUBLIC_MODEL_ID", ""))
    parser.add_argument("--host", default=os.getenv("LOCAL_VLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LOCAL_VLM_PORT", "8011")))
    parser.add_argument("--dtype", default=os.getenv("LOCAL_VLM_DTYPE", "float16"))
    args = parser.parse_args()

    server = LocalMultimodalServer(args.model_id, public_model_id=args.public_model_id, dtype_name=args.dtype)
    app = build_app(server)

    import uvicorn

    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
