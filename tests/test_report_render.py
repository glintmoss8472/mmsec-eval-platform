from mmsec_eval.viz.render_report import render_report_html


def test_report_render():
    html = render_report_html({"asr": 0.5}, [{"sample_id": "x"}], run_dir=".")
    assert "Summary" in html
    assert "sample_id" in html

