# 文件说明：该文件属于自动化测试，集中实现 test report render 相关逻辑。
from mmsec_eval.viz.render_report import render_report_html


# 验证 `报告 render` 场景，防止相关行为在后续修改中退化。
def test_report_render():
    html = render_report_html({"asr": 0.5}, [{"sample_id": "x"}], run_dir=".")
    assert "Summary" in html
    assert "sample_id" in html

