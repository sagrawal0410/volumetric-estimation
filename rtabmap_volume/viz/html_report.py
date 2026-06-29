"""HTML report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Template

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>RTAB-Map Volume Report</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; max-width: 960px; }
    h1, h2 { color: #1a1a2e; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background: #f4f4f8; }
    .warn { color: #c0392b; }
    .conf-high { color: #27ae60; font-weight: bold; }
    .conf-medium { color: #f39c12; font-weight: bold; }
    .conf-low { color: #e74c3c; font-weight: bold; }
    img { max-width: 100%; margin: 0.5rem 0; border: 1px solid #eee; }
    code { background: #f4f4f8; padding: 2px 6px; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>RTAB-Map Volume Estimation Report</h1>
  <p><strong>Input:</strong> {{ input_path }}</p>
  <p><strong>Command:</strong> <code>{{ command }}</code></p>
  <p><strong>Config:</strong> {{ config_path }}</p>

  <h2>Geometry Summary</h2>
  <table>
    {% for k, v in geometry_stats.items() %}
    <tr><th>{{ k }}</th><td>{{ v }}</td></tr>
    {% endfor %}
  </table>

  <h2>Final Estimate</h2>
  <p class="conf-{{ confidence }}">
    {{ final_volume_m3 }} m³ ({{ final_volume_liters }} L) —
    Confidence: {{ confidence }} ({{ confidence_score }})
  </p>
  <p>Recommended estimator: <strong>{{ recommended_estimator }}</strong></p>
  {% if upper_bound_m3 %}<p>Upper bound (convex hull): {{ upper_bound_m3 }} m³</p>{% endif %}

  <h2>All Estimates</h2>
  <table>
    <tr><th>Method</th><th>Volume (m³)</th><th>Liters</th><th>Reliable</th></tr>
    {% for name, est in all_estimates.items() %}
    <tr>
      <td>{{ name }}</td>
      <td>{{ est.value_m3 }}</td>
      <td>{{ est.value_liters }}</td>
      <td>{{ est.reliable }}</td>
    </tr>
    {% endfor %}
  </table>

  {% if warnings %}
  <h2>Warnings</h2>
  <ul class="warn">
    {% for w in warnings %}<li>{{ w }}</li>{% endfor %}
  </ul>
  {% endif %}

  <h2>Visualizations</h2>
  {% for img in screenshots %}
  {% if img.exists %}
  <h3>{{ img.name }}</h3>
  <img src="../screenshots/{{ img.name }}" alt="{{ img.name }}">
  {% endif %}
  {% endfor %}
</body>
</html>
"""


def generate_html_report(
    out_path: Path,
    data: dict[str, Any],
    screenshots_dir: Path,
) -> None:
    screenshot_names = [
        "raw_geometry.png",
        "cropped_geometry.png",
        "segmented_object.png",
        "repaired_mesh.png",
        "voxel_grid.png",
        "volume_methods_barplot.png",
    ]
    screenshots = []
    for name in screenshot_names:
        p = screenshots_dir / name
        screenshots.append(type("S", (), {"name": name, "exists": p.exists()})())

    tmpl = Template(REPORT_TEMPLATE)
    html = tmpl.render(
        screenshots=screenshots,
        **data,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
