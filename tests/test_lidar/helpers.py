"""
Shared test helpers for the LiDAR test suite.

Provides plot embedding, chi-squared variance bounds, and binomial
confidence interval utilities used across multiple test modules.
"""

import base64
import io

import numpy as np


def attach_plot_to_html_report(request, fig, name):
    """
    Embed a matplotlib figure into the pytest HTML report as an inline PNG.

    The figure is rendered to an in memory byte buffer, Base64 encoded, and
    appended to the ``extras`` list on the current test node.  When the
    ``pytest-html`` plugin is active, each extra entry becomes a collapsible
    image in the final HTML report.  If the plugin is not installed the
    function silently does nothing, so tests still pass without it.

    :param request:  the pytest ``request`` fixture, giving access to the
                     plugin manager and the current test node
    :param fig:      a ``matplotlib.figure.Figure`` to embed
    :param name:     a short label shown beside the image in the report
    """
    # Render the figure into a BytesIO buffer as a PNG at 160 DPI
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)

    # Try to retrieve the pytest-html plugin from the plugin manager
    html_plugin = request.config.pluginmanager.getplugin("html")

    # Only proceed if the plugin is loaded and exposes an extras helper
    if html_plugin is not None and hasattr(html_plugin, "extras"):
        # Encode the raw PNG bytes as a Base64 ASCII string for inline HTML
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # Fetch the existing extras list (or start a new one)
        extra = getattr(request.node, "extra", [])

        # Append a PNG extra entry; pytest-html will render it as an <img> tag
        extra.append(html_plugin.extras.png(png_b64, name=name))

        # Attach the updated list back to the test node for the report hook
        request.node.extra = extra


def chi2_variance_bounds(n, confidence=0.99):
    """
    Compute chi-squared confidence interval ratio bounds for sample variance.

    Uses the Wilson Hilferty normal approximation for chi-squared quantiles.
    Returns (lower_ratio, upper_ratio) such that s^2 / sigma^2 should lie
    within [lower_ratio, upper_ratio] at the given confidence level.

    The Wilson Hilferty approximation transforms chi-squared quantiles into
    a standard normal problem:
        chi2_q  ~  nu * (1 - 2/(9*nu) +/- z * sqrt(2/(9*nu)))^3
    Dividing by nu gives a ratio directly comparable to s^2 / sigma^2.

    :param n:          number of samples
    :param confidence: two-sided confidence level (default 0.99)
    :return: (lower_ratio, upper_ratio) for s^2 / sigma^2
    """
    # Lookup table mapping common two-sided confidence levels to z-scores
    z_table = {0.99: 2.576, 0.95: 1.960, 0.90: 1.645}
    # Fall back to 99 % if the requested level is not in the table
    z = z_table.get(confidence, 2.576)

    nu = float(n - 1)  # degrees of freedom: n minus one
    # Wilson Hilferty intermediate term: a = 2 / (9 * nu)
    a = 2.0 / (9.0 * nu)
    # Lower chi-squared quantile via the cube approximation
    chi2_lo = nu * (1.0 - a - z * np.sqrt(a)) ** 3  # lower chi-squared quantile
    # Upper chi-squared quantile via the cube approximation
    chi2_hi = nu * (1.0 - a + z * np.sqrt(a)) ** 3  # upper chi-squared quantile
    # Dividing by nu converts quantiles into variance ratio bounds
    return chi2_lo / nu, chi2_hi / nu


def binomial_ci(k, n, confidence=0.99):
    """
    Normal approximation confidence interval for a binomial proportion.

    The point estimate is p_hat = k / n.  The margin of error follows from
    the standard error of a binomial proportion:
        margin = z * sqrt(p_hat * (1 - p_hat) / n)
    The interval is clamped to [0, 1] because proportions cannot exceed
    that range.

    :param k:          number of successes
    :param n:          number of trials
    :param confidence: two-sided confidence level
    :return: (p_hat, ci_lower, ci_upper)
    """
    # Same z-score lookup used across all statistical helpers
    z_table = {0.99: 2.576, 0.95: 1.960, 0.90: 1.645}
    z = z_table.get(confidence, 2.576)

    # Maximum likelihood estimate of the true proportion
    p_hat = k / n
    # Half width of the confidence interval based on the normal approximation
    margin = z * np.sqrt(p_hat * (1.0 - p_hat) / n)
    # Clamp the lower bound at 0 and the upper bound at 1
    return p_hat, max(p_hat - margin, 0.0), min(p_hat + margin, 1.0)
