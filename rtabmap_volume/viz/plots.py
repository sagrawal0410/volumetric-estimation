"""Matplotlib plots for volume reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_volume_methods(estimates: dict[str, dict], out_path: Path) -> None:
    names = []
    values = []
    colors = []
    for name, est in estimates.items():
        if isinstance(est, dict) and est.get("value_m3") is not None:
            names.append(name.replace("_", "\n"))
            values.append(est["value_m3"])
            colors.append("#2ecc71" if est.get("reliable") else "#e74c3c")

    if not names:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, values, color=colors)
    ax.set_ylabel("Volume (m³)")
    ax.set_title("Volume Estimates by Method")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_alpha_volumes(alpha_volumes: dict[float, float | None], out_path: Path) -> None:
    alphas = []
    vols = []
    for a, v in sorted(alpha_volumes.items()):
        if v is not None:
            alphas.append(a)
            vols.append(v)
    if not alphas:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, vols, "o-")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Volume (m³)")
    ax.set_title("Alpha Shape Volume vs Alpha")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
