# 文件说明：该文件属于自动化测试，集中实现 test documentation contract 相关逻辑。
from pathlib import Path


# 中文注释：验证 test_technical_architecture_guide_covers_required_engineering_topics 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_technical_architecture_guide_covers_required_engineering_topics() -> None:
    doc = Path("docs/technical_architecture_and_extension_guide.md").read_text(encoding="utf-8")

    required_phrases = [
        "项目定位",
        "顶层架构",
        "目录职责",
        "攻击方法扩展规范",
        "模型适配扩展规范",
        "数据集扩展规范",
        "指标体系",
        "异常处理规范",
        "测试策略",
        "代码质量门禁",
        "当前已知技术债",
    ]
    for phrase in required_phrases:
        assert phrase in doc

    assert "前 K 召回率（Recall at K）" in doc
    assert "条件攻击成功率（conditional attack success rate）" in doc


# 中文注释：验证 test_readme_links_primary_technical_documents 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_readme_links_primary_technical_documents() -> None:
    readme = Path("README.md").read_text(encoding="utf-8-sig")

    assert "docs/technical_architecture_and_extension_guide.md" in readme
    assert "docs/engineering_governance_review_20260501.md" in readme
    assert "docs/strict_paper_reproduction_protocol.md" in readme
    assert "http://127.0.0.1:5173/engineering" not in readme
    assert "http://127.0.0.1:5173/compliance" not in readme
    assert "http://127.0.0.1:8000/api/v1/system/compliance" in readme
