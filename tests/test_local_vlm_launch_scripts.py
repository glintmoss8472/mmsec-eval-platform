from __future__ import annotations

from pathlib import Path

from mmsec_eval.model_adapters.local_vlm_catalog import LOCAL_OPENAI_COMPAT_MODEL_SPECS


def test_local_vlm_launch_scripts_only_kill_their_own_ports():
    for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS:
        content = Path(spec.launch_script).read_text(encoding="utf-8")
        assert 'TARGET_PORT="${TARGET_PORT:-' in content
        assert 'source "${SCRIPT_DIR}/_local_vlm_server_env.sh"' in content

    shared_env = Path("scripts", "_local_vlm_server_env.sh").read_text(encoding="utf-8")
    assert 'CLEANUP_PORTS="${CLEANUP_PORTS:-$(mmsec_default_cleanup_ports "${TARGET_PORT}")}"' in shared_env
    assert 'MMSEC_LOCAL_VLM_SINGLE_TENANT' in shared_env
    assert 'MMSEC_LOCAL_VLM_ALL_PORTS:-8011 8012 8013 8014 8015 8016 8017' in shared_env
    assert 'pkill -f "local_openai_mm_server.py.*--port ${port}" || true' in shared_env
