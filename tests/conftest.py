import sys
from html import escape
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _report_name_from_args(args):
    test_files = []

    def _as_test_file(text):
        base = text.split("::", 1)[0]
        path = Path(base)
        if path.suffix == ".py" and path.name.startswith("test_"):
            return path
        return None

    for arg in args:
        text = str(arg)
        if text.startswith("-"):
            continue
        path = _as_test_file(text)
        if path is not None:
            test_files.append(path)

    unique_files = {str(p).lower(): p for p in test_files}
    if len(unique_files) == 1:
        sensor_name = next(iter(unique_files.values())).stem.removeprefix("test_")
        if sensor_name:
            return f"report_{sensor_name}.html"

    discovered = sorted((ROOT / "tests").rglob("test_*.py"))
    if len(discovered) == 1:
        sensor_name = discovered[0].stem.removeprefix("test_")
        if sensor_name:
            return f"report_{sensor_name}.html"

    return "report_all.html"


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    user_set_html = any(str(arg).startswith("--html") for arg in config.invocation_params.args)
    if user_set_html:
        return

    report_dir = ROOT / "tests" / "test_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_name = _report_name_from_args(config.invocation_params.args)
    config.option.htmlpath = str(report_dir / report_name)


def pytest_html_results_table_header(cells):
    cells.insert(3, '<th class="col-testmeta">Test Description</th>')
    cells.insert(4, '<th class="col-plot">Plot</th>')


def _format_test_meta(report):
    meta = getattr(report, "test_meta", None)
    if not meta:
        return '<div style="color:#666;">n/a</div>'

    description = escape(str(meta.get("description", "")))
    goal = escape(str(meta.get("goal", "")))
    passing = escape(str(meta.get("passing_criteria", "")))
    return (
        '<div style="min-width:340px;max-width:520px;line-height:1.35;">'
        f"<div><strong>Test Description:</strong> {description}</div>"
        f"<div><strong>Test Goal:</strong> {goal}</div>"
        f"<div><strong>Passing Criteria:</strong> {passing}</div>"
        "</div>"
    )


def pytest_html_results_table_row(report, cells):
    cells.insert(3, f'<td class="col-testmeta">{_format_test_meta(report)}</td>')

    images = []
    for extra in getattr(report, "extras", []):
        if extra.get("format_type") != "image":
            continue
        content = extra.get("content")
        if not content:
            continue
        images.append(
            f'<a href="{content}" target="_blank" rel="noopener noreferrer">'
            f'<img src="{content}" alt="plot" '
            f'style="max-width:320px;height:auto;display:block;margin:4px 0;cursor:zoom-in;" />'
            f"</a>"
        )

    if images:
        cells.insert(4, f'<td class="col-plot">{"".join(images)}</td>')
    else:
        cells.insert(4, '<td class="col-plot"></td>')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when != "call":
        return

    marker = item.get_closest_marker("test_meta")
    if marker:
        report.test_meta = {
            "description": marker.kwargs.get("description", ""),
            "goal": marker.kwargs.get("goal", ""),
            "passing_criteria": marker.kwargs.get("passing_criteria", ""),
        }

    item_extra = getattr(item, "extra", None)
    if not item_extra:
        return

    extras = getattr(report, "extras", [])
    extras.extend([dict(extra) for extra in item_extra])
    report.extras = extras
    if hasattr(report, "extra"):
        report.extra = extras
