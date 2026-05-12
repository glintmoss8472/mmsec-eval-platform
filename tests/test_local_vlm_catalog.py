# 文件说明：该文件属于自动化测试，集中实现 test local vlm catalog 相关逻辑。
from __future__ import annotations

import re
from pathlib import Path

import yaml

from mmsec_api.services.model_runtime import MAIN_MODEL_SPECS
from mmsec_eval.model_adapters.local_vlm_catalog import (
    LOCAL_OPENAI_COMPAT_ADAPTERS,
    LOCAL_OPENAI_COMPAT_LOCAL_DIRS,
    LOCAL_OPENAI_COMPAT_MODEL_SPECS,
    local_vlm_calibration_map,
    local_vlm_launch_matrix,
    local_vlm_model_map,
    local_vlm_spec_by_adapter,
)
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.plugins.registry import list_plugins


# 验证 `本地 视觉语言模型 catalog drives runtime registration and scripts` 场景，防止相关行为在后续修改中退化。
def test_local_vlm_catalog_drives_runtime_registration_and_scripts() -> None:
    assert len(LOCAL_OPENAI_COMPAT_MODEL_SPECS) == 7
    assert LOCAL_OPENAI_COMPAT_ADAPTERS == tuple(spec.adapter for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS)
    assert LOCAL_OPENAI_COMPAT_LOCAL_DIRS == tuple(spec.local_dir for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS)
    assert local_vlm_model_map() == {spec.local_dir: spec.model_name_default for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS}
    assert local_vlm_launch_matrix() == tuple(
        (spec.adapter, spec.launch_script, spec.endpoint_port) for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS
    )

    runtime_by_adapter = {spec.adapter: spec for spec in MAIN_MODEL_SPECS}
    calibration = local_vlm_calibration_map()
    for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS:
        runtime = runtime_by_adapter[spec.adapter]
        assert runtime.model_name_env == spec.model_name_env
        assert runtime.endpoint_env == spec.endpoint_env
        assert runtime.model_name_default == spec.model_name_default
        assert runtime.endpoint_default == spec.endpoint_default
        assert runtime.launch_script == spec.launch_script
        assert runtime.launch_log == spec.launch_log
        assert spec.api_key_env == f"MMSEC_OPENAI_{spec.variant}_API_KEY_ENV"
        assert spec.api_key_env_default == f"LOCAL_{spec.variant}_API_KEY"
        assert spec.timeout_env == f"MMSEC_OPENAI_{spec.variant}_TIMEOUT"
        assert calibration[spec.adapter] == (spec.variant, spec.model_name_default, spec.endpoint_default)

    register_builtin_plugins()
    registered = set(list_plugins("model_adapter"))
    assert set(LOCAL_OPENAI_COMPAT_ADAPTERS).issubset(registered)


# 验证 `本地 视觉语言模型 catalog 是否 visible in frontend 模型 catalog` 场景，防止相关行为在后续修改中退化。
def test_local_vlm_catalog_is_visible_in_frontend_model_catalog() -> None:
    frontend_catalog = Path("frontend/src/lib/modelCatalog.ts").read_text(encoding="utf-8-sig")
    frontend_adapters = set(re.findall(r'adapter:\s*"([^"]+)"', frontend_catalog))

    assert set(LOCAL_OPENAI_COMPAT_ADAPTERS).issubset(frontend_adapters)


# 验证 `OpenAI bench configs match current 本地 视觉语言模型 catalog` 场景，防止相关行为在后续修改中退化。
def test_openai_bench_configs_match_current_local_vlm_catalog() -> None:
    config_by_adapter = {
        "openai_qwen3_vl": Path("configs/bench/bootstrap_qwen3_vl_openai.yaml"),
        "openai_qwen25_vl": Path("configs/bench/bootstrap_qwen25_vl_openai.yaml"),
        "openai_internvl35": Path("configs/bench/bootstrap_internvl35_8b_openai.yaml"),
    }

    assert not Path("configs/bench/bootstrap_internvl35_1b_openai.yaml").exists()
    for adapter, path in config_by_adapter.items():
        spec = local_vlm_spec_by_adapter(adapter)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = data["model"]
        assert model["openai_model_name"] == spec.model_name_default
        assert model["openai_base_url"] == spec.endpoint_default
        assert model["openai_api_key_env"] == spec.api_key_env_default
