from vix.core.report import build_report, render_markdown, write_report


def test_build_and_render_report():
    rep = build_report(
        version="v1",
        total=100,
        class_dist={"a": 80, "b": 5},
        pass_count=60,
        review_count=40,
        duplicate_groups=[["x", "y"], ["p", "q", "r"]],  # 1 + 2 = 3 redundant
        label_issues=[1, 2, 3],
        coverage={"a": {"under_represented": False}, "b": {"under_represented": True}},
    )
    assert abs(rep["duplicate_rate"] - 0.03) < 1e-9
    assert rep["review_ratio"] == 0.4
    assert rep["n_suspected_label_errors"] == 3
    assert rep["under_represented_classes"] == ["b"]

    md = render_markdown(rep)
    assert "資料集健檢報告" in md
    assert "`b`" in md


def test_report_diff_and_write(tmp_path):
    prev = {"total_images": 50, "class_distribution": {"a": 50}}
    rep = build_report(version="v2", total=60, class_dist={"a": 55, "c": 5}, prev=prev)
    assert rep["diff"]["total_delta"] == 10
    assert rep["diff"]["new_classes"] == ["c"]

    paths = write_report(rep, tmp_path / "rep")
    assert (tmp_path / "rep" / "health_report.md").exists()
    assert (tmp_path / "rep" / "health_report.json").exists()
