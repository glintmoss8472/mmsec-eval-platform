# 文件说明：该文件属于模型适配层，集中实现 local vlm catalog 相关逻辑。
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


# 定义 `LocalVLMModelSpec` 的插件适配边界，把模型、攻击或评测能力暴露为统一接口。
@dataclass(frozen=True)
class LocalVLMModelSpec:
    adapter: str
    variant: str
    local_dir: str
    display_name: str
    model_name_default: str
    endpoint_default: str
    launch_script: str
    launch_log: str
    timeout_default: str = "90"

    # 推断 `模型 名称 环境`，从样本、配置或运行记录中提取统一名称。
    @property
    def model_name_env(self) -> str:
        return f"MMSEC_OPENAI_{self.variant}_MODEL_NAME"

    # 拼接 `endpoint 环境`，把配置中的主机、端口和路径合成实际访问入口。
    @property
    def endpoint_env(self) -> str:
        return f"MMSEC_OPENAI_{self.variant}_BASE_URL"

    # 实现 `LocalVLMModelSpec.prompt_order_env` 的对象行为，维护该类在模型适配层中的调用契约。
    @property
    def prompt_order_env(self) -> str:
        return f"MMSEC_OPENAI_{self.variant}_PROMPT_ORDER"

    # 实现 `LocalVLMModelSpec.timeout_env` 的对象行为，维护该类在模型适配层中的调用契约。
    @property
    def timeout_env(self) -> str:
        return f"MMSEC_OPENAI_{self.variant}_TIMEOUT"

    # 实现 `LocalVLMModelSpec.api_key_env` 的对象行为，维护该类在模型适配层中的调用契约。
    @property
    def api_key_env(self) -> str:
        return f"MMSEC_OPENAI_{self.variant}_API_KEY_ENV"

    # 实现 `LocalVLMModelSpec.api_key_env_default` 的对象行为，维护该类在模型适配层中的调用契约。
    @property
    def api_key_env_default(self) -> str:
        return f"LOCAL_{self.variant}_API_KEY"

    # 拼接 `endpoint port`，把配置中的主机、端口和路径合成实际访问入口。
    @property
    def endpoint_port(self) -> int:
        port = urlparse(self.endpoint_default).port
        if port is None:
            raise ValueError(f"local VLM endpoint has no port: {self.endpoint_default}")
        return int(port)


LOCAL_OPENAI_COMPAT_MODEL_SPECS: tuple[LocalVLMModelSpec, ...] = (
    LocalVLMModelSpec(
        adapter="openai_qwen35_9b",
        variant="QWEN35_9B",
        local_dir="qwen35_9b",
        display_name="通义千问三点五九十亿参数模型（Qwen3.5-9B）",
        model_name_default="Qwen/Qwen3.5-9B",
        endpoint_default="http://127.0.0.1:8011/v1",
        launch_script="scripts/run_local_qwen35_9b_server.sh",
        launch_log="qwen35_9b.log",
    ),
    LocalVLMModelSpec(
        adapter="openai_qwen3_vl",
        variant="QWEN3_VL",
        local_dir="qwen3_vl",
        display_name="通义千问三视觉语言模型八十亿参数版（Qwen3-VL-8B）",
        model_name_default="Qwen/Qwen3-VL-8B-Instruct",
        endpoint_default="http://127.0.0.1:8012/v1",
        launch_script="scripts/run_local_qwen3_vl_server.sh",
        launch_log="qwen3_vl_8b.log",
    ),
    LocalVLMModelSpec(
        adapter="openai_qwen25_vl",
        variant="QWEN25_VL",
        local_dir="qwen25_vl",
        display_name="通义千问二点五视觉语言模型七十亿参数版（Qwen2.5-VL-7B）",
        model_name_default="Qwen/Qwen2.5-VL-7B-Instruct",
        endpoint_default="http://127.0.0.1:8013/v1",
        launch_script="scripts/run_local_qwen25_vl_server.sh",
        launch_log="qwen25_vl_7b.log",
    ),
    LocalVLMModelSpec(
        adapter="openai_internvl35",
        variant="INTERNVL35",
        local_dir="internvl35",
        display_name="书生万象三点五八十亿参数模型（InternVL3.5-8B）",
        model_name_default="OpenGVLab/InternVL3_5-8B-HF",
        endpoint_default="http://127.0.0.1:8014/v1",
        launch_script="scripts/run_local_internvl35_8b_server.sh",
        launch_log="internvl35_8b.log",
    ),
    LocalVLMModelSpec(
        adapter="openai_minicpm_v",
        variant="MINICPM_V",
        local_dir="minicpm_v",
        display_name="迷你通用处理模型视觉版四点五（MiniCPM-V 4.5）",
        model_name_default="openbmb/MiniCPM-V-4_5",
        endpoint_default="http://127.0.0.1:8015/v1",
        launch_script="scripts/run_local_minicpm_v_server.sh",
        launch_log="minicpm_v_45.log",
    ),
    LocalVLMModelSpec(
        adapter="openai_ovis25",
        variant="OVIS25",
        local_dir="ovis25",
        display_name="奥维斯二点五九十亿参数模型（Ovis2.5-9B）",
        model_name_default="AIDC-AI/Ovis2.5-9B",
        endpoint_default="http://127.0.0.1:8016/v1",
        launch_script="scripts/run_local_ovis25_server.sh",
        launch_log="ovis25_9b.log",
    ),
    LocalVLMModelSpec(
        adapter="openai_gemma3_12b",
        variant="GEMMA3_12B",
        local_dir="gemma3_12b",
        display_name="谷歌 Gemma 三代一百二十亿参数模型（Gemma 3-12B）",
        model_name_default="google/gemma-3-12b-it",
        endpoint_default="http://127.0.0.1:8017/v1",
        launch_script="scripts/run_local_gemma3_12b_server.sh",
        launch_log="gemma3_12b.log",
    ),
)

LOCAL_OPENAI_COMPAT_ADAPTERS: tuple[str, ...] = tuple(spec.adapter for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS)
LOCAL_OPENAI_COMPAT_LOCAL_DIRS: tuple[str, ...] = tuple(spec.local_dir for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS)


# 整理 `本地 视觉语言模型 spec by adapter`，描述当前服务器运行环境、模型入口或部署状态。
def local_vlm_spec_by_adapter(adapter: str) -> LocalVLMModelSpec:
    for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS:
        if spec.adapter == adapter:
            return spec
    raise KeyError(f"unknown local VLM adapter: {adapter}")


# 定位 `本地 视觉语言模型 spec by 本地 目录`，把配置值或请求上下文转换成实际文件系统路径。
def local_vlm_spec_by_local_dir(local_dir: str) -> LocalVLMModelSpec:
    for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS:
        if spec.local_dir == local_dir:
            return spec
    raise KeyError(f"unknown local VLM directory: {local_dir}")


# 推断 `本地 视觉语言模型 模型 map`，从样本、配置或运行记录中提取统一名称。
def local_vlm_model_map() -> dict[str, str]:
    return {spec.local_dir: spec.model_name_default for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS}


# 执行 `本地 视觉语言模型 calibration map` 辅助逻辑，保持模型适配层中的输入处理和结果输出一致。
def local_vlm_calibration_map() -> dict[str, tuple[str, str, str]]:
    return {
        spec.adapter: (spec.variant, spec.model_name_default, spec.endpoint_default)
        for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS
    }


# 构建 `本地 视觉语言模型 launch 矩阵`，把图像和文本两两配对后整理成指标计算所需的二维结果。
def local_vlm_launch_matrix() -> tuple[tuple[str, str, int], ...]:
    return tuple((spec.adapter, spec.launch_script, spec.endpoint_port) for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS)
