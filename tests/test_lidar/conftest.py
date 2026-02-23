"""
Pytest Configuration and HTML Report Hooks

This module configures the pytest test runner for the GNC-sensors project.
It handles automatic HTML report generation with custom columns for test
metadata (description, goal, passing criteria) and embedded plot images.

The hooks integrate with pytest-html to produce self-contained test reports
that include both textual descriptions and visual verification plots for
each test case.
"""

import sys
from html import escape
from pathlib import Path

import pytest

# Resolve the project root directory (one level above the tests/ folder)
ROOT = Path(__file__).resolve().parents[2]

# Ensure the project root is on the Python import path so that the
# sensors package can be imported without installation
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _report_name_from_args(args):
    """
    Determine the HTML report filename based on the pytest command-line arguments.

    If exactly one test file is being run, the report is named after that
    file's sensor (e.g. "report_lidar.html" for test_lidar.py). If no
    specific file can be identified, the fallback name "report_all.html"
    is used.

    :param args: List of command-line arguments passed to pytest.
    :return: Report filename string (e.g. "report_lidar.html").
    """
    # Collect all test file paths from the command-line arguments
    test_files = []

    def _as_test_file(text):
        """
        Try to parse a command-line argument as a test file path.

        Strips any ::test_name suffix, then checks if the remaining
        path ends in .py and starts with "test_".

        :param text: Raw command-line argument string.
        :return: Path object if it looks like a test file, otherwise None.
        """
        # Split off any pytest node ID suffix (e.g. "test_lidar.py::test_foo")
        base = text.split("::", 1)[0]
        path = Path(base)

        # Only accept .py files whose name starts with "test_"
        if path.suffix == ".py" and path.name.startswith("test_"):
            return path
        return None

    for arg in args:
        text = str(arg)

        # Skip flags/options (arguments starting with a dash)
        if text.startswith("-"):
            continue

        # Try to interpret this argument as a test file
        path = _as_test_file(text)
        if path is not None:
            test_files.append(path)

    # Deduplicate test files by lowercased path string
    unique_files = {str(p).lower(): p for p in test_files}

    # If exactly one test file was specified, derive the report name from it
    if len(unique_files) == 1:
        # Extract the sensor name by removing the "test_" prefix from the stem
        sensor_name = next(iter(unique_files.values())).stem.removeprefix("test_")
        if sensor_name:
            return f"report_{sensor_name}.html"

    # Fallback: identify the sensor from the conftest directory name.
    # This conftest lives inside tests/test_<sensor>/, so the parent
    # directory name encodes the sensor being tested.
    conftest_dir = Path(__file__).resolve().parent
    if conftest_dir.name.startswith("test_"):
        sensor_name = conftest_dir.name.removeprefix("test_")
        if sensor_name:
            return f"report_{sensor_name}.html"

    # Default report name when no sensor can be identified
    return "report_all.html"


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """
    Pytest configuration hook that sets up automatic HTML report output.

    Runs early (tryfirst=True) during pytest startup. If the user has not
    already specified an --html flag, this hook creates the test_reports/
    directory and sets the report path automatically.

    :param config: The pytest Config object for this session.
    """
    # Check whether the user explicitly passed --html on the command line
    user_set_html = any(str(arg).startswith("--html") for arg in config.invocation_params.args)

    # If the user already chose an HTML path, do not override it
    if user_set_html:
        return

    # Create the output directory for test reports if it does not exist.
    # The report lives alongside this conftest inside tests/test_lidar/.
    report_dir = Path(__file__).resolve().parent / "test_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Determine the appropriate report filename from the CLI arguments
    report_name = _report_name_from_args(config.invocation_params.args)

    # Tell pytest-html where to write the report
    config.option.htmlpath = str(report_dir / report_name)


def pytest_html_results_table_header(cells):
    """
    Customize the HTML report table by inserting extra column headers.

    Adds a "Test Description" column (for test metadata) and a "Plot"
    column (for embedded verification images) after the default columns.

    :param cells: List of HTML header cell strings, modified in place.
    """
    # Insert the test description column at position 3
    cells.insert(3, '<th class="col-testmeta">Test Description</th>')

    # Insert the plot/image column at position 4
    cells.insert(4, '<th class="col-plot">Plot</th>')


def _format_test_meta(report):
    """
    Format test metadata (description, goal, passing criteria) as an HTML block.

    Reads the test_meta attribute from the report object and renders it
    as a styled HTML div. Returns a grey "n/a" placeholder if no metadata
    is attached.

    :param report: The pytest test report object.
    :return: HTML string containing the formatted metadata.
    """
    # Retrieve the test_meta dict attached during pytest_runtest_makereport
    meta = getattr(report, "test_meta", None)
    if not meta:
        return '<div style="color:#666;">n/a</div>'

    # Escape all strings to prevent HTML injection
    description = escape(str(meta.get("description", "")))
    goal = escape(str(meta.get("goal", "")))
    passing = escape(str(meta.get("passing_criteria", "")))

    # Build a styled HTML block showing description, goal, and passing criteria
    return (
        '<div style="min-width:340px;max-width:520px;line-height:1.35;">'
        f"<div><strong>Test Description:</strong> {description}</div>"
        f"<div><strong>Test Goal:</strong> {goal}</div>"
        f"<div><strong>Passing Criteria:</strong> {passing}</div>"
        "</div>"
    )


def pytest_html_results_table_row(report, cells):
    """
    Populate the custom columns for each test row in the HTML report.

    Inserts the formatted test metadata into the "Test Description" column
    and embeds any attached plot images into the "Plot" column.

    :param report: The pytest test report object for this row.
    :param cells: List of HTML cell strings for this row, modified in place.
    """
    # Insert the formatted test metadata cell
    cells.insert(3, f'<td class="col-testmeta">{_format_test_meta(report)}</td>')

    # Collect any image extras that were attached during the test
    images = []
    for extra in getattr(report, "extras", []):
        # Only process entries marked as image format
        if extra.get("format_type") != "image":
            continue

        # Get the image content (typically a base64 data URI or file path)
        content = extra.get("content")
        if not content:
            continue

        # Build an img tag wrapped in a clickable link for zoom-in viewing
        images.append(
            f'<a href="{content}" target="_blank" rel="noopener noreferrer">'
            f'<img src="{content}" alt="plot" '
            f'style="max-width:320px;height:auto;display:block;margin:4px 0;cursor:zoom-in;" />'
            f"</a>"
        )

    # Insert the plot images cell, or an empty cell if there are no images
    if images:
        cells.insert(4, f'<td class="col-plot">{"".join(images)}</td>')
    else:
        cells.insert(4, '<td class="col-plot"></td>')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Post-test hook that attaches test metadata and plot extras to the report.

    Uses hookwrapper=True to run after the default report is created.
    Extracts the @pytest.mark.test_meta marker kwargs (description, goal,
    passing_criteria) and stores them on the report object. Also transfers
    any extra attachments (like plot images) from the test item to the report.

    :param item: The pytest test item that just ran.
    :param call: The pytest CallInfo object for this test phase.
    """
    # Yield to let the default report be created first
    outcome = yield
    report = outcome.get_result()

    # Only process the "call" phase (not setup or teardown)
    if report.when != "call":
        return

    # Look for the @pytest.mark.test_meta marker on the test function
    marker = item.get_closest_marker("test_meta")
    if marker:
        # Store the marker kwargs as a dict on the report for later use
        report.test_meta = {
            "description": marker.kwargs.get("description", ""),
            "goal": marker.kwargs.get("goal", ""),
            "passing_criteria": marker.kwargs.get("passing_criteria", ""),
        }

    # Check if the test item has any extra attachments (e.g. plot images)
    item_extra = getattr(item, "extra", None)
    if not item_extra:
        return

    # Merge the test item's extras into the report's extras list
    extras = getattr(report, "extras", [])
    extras.extend([dict(extra) for extra in item_extra])
    report.extras = extras

    # Also set report.extra for compatibility with older pytest-html versions
    if hasattr(report, "extra"):
        report.extra = extras
