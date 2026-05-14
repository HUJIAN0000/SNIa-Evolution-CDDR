# -*- coding: utf-8 -*-
"""
cosmo_tools.py
--------------
A small, dependency-light cosmology MCMC post-processing toolbox.

Design goals
- Stable API for scripts like `nature_solver_curvature.py`
- Works even if optional plotting deps are missing (corner / getdist)
- Produces a Figure-4-like triangle plot:
  * 2D: filled blue credible regions
  * 1D: dual-shaded 2σ (tan) + 1σ (sienna) bands + median dashed line
  * Titles on diagonal panels show median ±1σ in LaTeX

Public API (backward compatible)
- calculate_stats(samples, labels)
- print_results(stats_list, lnL_best=None, num_data=None)
- plot_corner(samples, labels, filename="corner.png", truths=None)
- plot_getdist_advanced(samples, labels, stats_list=None, filename="getdist_result.pdf", **kwargs)
- plot_getdist_comparison(samples_list, labels, legend_labels=None, colors=None, filename="comparison.pdf")
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

# Optional deps
HAS_CORNER = False
HAS_GETDIST = False
try:
    import corner  # type: ignore
    HAS_CORNER = True
except Exception:
    HAS_CORNER = False

try:
    from getdist import plots, MCSamples  # type: ignore
    HAS_GETDIST = True
except Exception:
    HAS_GETDIST = False

if not (HAS_CORNER or HAS_GETDIST):
    warnings.warn(
        "cosmo_tools: neither 'corner' nor 'getdist' is available. "
        "Plotting functions will be disabled.",
        RuntimeWarning,
    )


# =========================
# 1) Statistics
# =========================
def calculate_stats(samples: np.ndarray, labels: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Compute summary statistics for each parameter.

    Returns a list of dicts, each containing:
      - name, median, upper, lower, best
      - 1sigma [p16, p84]
      - 2sigma [p2.275, p97.725]
      - title_fmt: e.g. "\\Omega_m = 0.284_{-0.012}^{+0.013}"
    """
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("samples must be a 2D array of shape (nsamples, ndim)")
    ndim = samples.shape[1]
    if len(labels) != ndim:
        raise ValueError(f"labels length ({len(labels)}) != ndim ({ndim})")

    stats_list: List[Dict[str, Any]] = []
    for i in range(ndim):
        p2_low, p1_low, med, p1_high, p2_high = np.percentile(
            samples[:, i], [2.275, 16.0, 50.0, 84.0, 97.725]
        )
        up = float(p1_high - med)
        down = float(med - p1_low)

        stats_list.append(
            {
                "name": labels[i],
                "median": float(med),
                "upper": up,
                "lower": down,
                "best": float(med),
                "1sigma": [float(p1_low), float(p1_high)],
                "2sigma": [float(p2_low), float(p2_high)],
                "title_fmt": f"{labels[i]} = {med:.3f}_{{-{down:.3f}}}^{{+{up:.3f}}}",
            }
        )
    return stats_list


def print_results(stats_list: Sequence[Dict[str, Any]], lnL_best: Optional[float] = None, num_data: Optional[int] = None) -> None:
    """
    Print a compact table of median ±1σ constraints.
    If lnL_best and num_data are provided, also prints AIC/BIC.
    """
    ndim = len(stats_list)
    print("\n" + "=" * 56)
    print("MCMC constraints (median and +upper / -lower)")
    print("=" * 56)
    for st in stats_list:
        print(f"{st['name']:<10} = {st['median']:.4f}  +{st['upper']:.4f}  -{st['lower']:.4f}")

    if lnL_best is not None:
        print("-" * 56)
        print(f"Best lnL = {lnL_best:.2f}")
        if num_data is not None and num_data > 0:
            aic = 2 * ndim - 2 * lnL_best
            bic = ndim * math.log(num_data) - 2 * lnL_best
            print(f"AIC      = {aic:.2f}")
            print(f"BIC      = {bic:.2f}")
    print("=" * 56 + "\n")


# =========================
# 2) Plot helpers
# =========================
def _default_names(ndim: int) -> List[str]:
    return [f"p{i}" for i in range(ndim)]


def _make_mcsamples(samples: np.ndarray, labels: Sequence[str], label: Optional[str] = None,
                    smooth_1d: float = 0.5, smooth_2d: float = 0.7) -> "MCSamples":
    if not HAS_GETDIST:
        raise RuntimeError("getdist is not available")
    samples = np.asarray(samples)
    names = _default_names(samples.shape[1])
    settings = {"smooth_scale_1D": smooth_1d, "smooth_scale_2D": smooth_2d}
    return MCSamples(samples=samples, names=names, labels=list(labels), label=label, settings=settings)


def plot_corner(samples: np.ndarray, labels: Sequence[str], filename: str = "corner.png", truths: Optional[Sequence[float]] = None) -> None:
    """Draw a classic corner plot (requires `corner`)."""
    if not HAS_CORNER:
        print("⚠️ plot_corner skipped: 'corner' is not installed.")
        return

    fig = corner.corner(
        np.asarray(samples),
        bins=30,
        labels=list(labels),
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        smooth=True,
        smooth1d=True,
        plot_contours=True,
        fill_contours=True,
        truths=truths,
        title_fmt=".3f",
    )
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Corner plot saved: {filename}")


# =========================
# 3) GetDist plots
# =========================
def plot_getdist_advanced(
    samples: Union[np.ndarray, Sequence[np.ndarray]],
    labels: Sequence[str],
    stats_list: Optional[Sequence[Dict[str, Any]]] = None,
    filename: str = "getdist_result.pdf",
    **kwargs: Any,
) -> None:
    """
    Publication-style triangle plot.

    Backward compatible:
      - If `samples` is a list/tuple of arrays -> dispatch to plot_getdist_comparison
      - Otherwise treat as a single chain, and mimic Nature Figure-4 style:
          * 2D filled contours in blue
          * 1D dual-shaded 2σ (tan) and 1σ (sienna) regions
          * dashed median line
          * legend in upper-right
    """
    # If a multi-chain list is passed, forward to comparison helper
    if isinstance(samples, (list, tuple)) and len(samples) > 0 and isinstance(samples[0], (np.ndarray, list, tuple)):
        legend_labels = kwargs.get("legend_labels", kwargs.get("legends", None))
        colors = kwargs.get("colors", None)
        return plot_getdist_comparison(
            samples_list=list(samples), labels=labels, legend_labels=legend_labels, colors=colors, filename=filename
        )

    if not HAS_GETDIST:
        print("⚠️ plot_getdist_advanced skipped: 'getdist' is not installed.")
        return

    chain = np.asarray(samples)
    if stats_list is None:
        stats_list = calculate_stats(chain, labels)

    # Build getdist object
    mc = _make_mcsamples(chain, labels)

    # Plot
    width_inch = float(kwargs.get("width_inch", 8.0))
    contour_color = kwargs.get("contour_color", "#1f77b4")  # blue
    g = plots.get_subplot_plotter(width_inch=width_inch)
    g.settings.axes_fontsize = kwargs.get("axes_fontsize", 12)
    g.settings.lab_fontsize = kwargs.get("lab_fontsize", 14)
    g.settings.figure_legend_frame = False

    g.triangle_plot(mc, filled=True, contour_colors=[contour_color])

    # Apply diagonal titles + dual shading
    names = _default_names(len(labels))
    added_legend = False
    for i in range(len(labels)):
        ax = g.subplots[i, i]
        st = stats_list[i]

        # Diagonal title
        ax.set_title(f"${st['title_fmt']}$", fontsize=12)

        # 1D density from getdist
        try:
            dens = mc.get1DDensity(names[i])
        except Exception:
            dens = None

        if dens is None:
            continue

        x = np.asarray(getattr(dens, "x", []))
        p = np.asarray(getattr(dens, "P", []))
        if x.size == 0 or p.size == 0:
            continue

        # 2σ band
        lo2, hi2 = st["2sigma"]
        m2 = (x >= lo2) & (x <= hi2)
        if np.any(m2):
            ax.fill_between(
                x[m2], 0, p[m2],
                color="tan", alpha=0.40,
                label=r"$2\sigma$ Region" if not added_legend else None,
                zorder=1,
            )

        # 1σ band
        lo1, hi1 = st["1sigma"]
        m1 = (x >= lo1) & (x <= hi1)
        if np.any(m1):
            ax.fill_between(
                x[m1], 0, p[m1],
                color="sienna", alpha=0.70,
                label=r"$1\sigma$ Region" if not added_legend else None,
                zorder=2,
            )

        # Median line
        ax.axvline(st["median"], color="black", ls="--", lw=1.5, zorder=3)

        if (np.any(m1) or np.any(m2)) and not added_legend:
            added_legend = True

    # Global legend (use first diagonal handles)
    try:
        handles, leg_labels = g.subplots[0, 0].get_legend_handles_labels()
        if handles:
            g.fig.legend(handles, leg_labels, loc="upper right", bbox_to_anchor=(0.95, 0.95), frameon=False, fontsize=12)
    except Exception:
        pass

    g.export(filename)
    print(f"✅ GetDist plot saved: {filename}")


def plot_getdist_comparison(
    samples_list: Sequence[np.ndarray],
    labels: Sequence[str],
    legend_labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    filename: str = "comparison.pdf",
) -> None:
    """
    Overlay multiple chains on one triangle plot (requires getdist).
    """
    if not HAS_GETDIST:
        print("⚠️ plot_getdist_comparison skipped: 'getdist' is not installed.")
        return

    if len(samples_list) == 0:
        raise ValueError("samples_list is empty")

    mc_list: List["MCSamples"] = []
    for i, s in enumerate(samples_list):
        lbl = legend_labels[i] if (legend_labels is not None and i < len(legend_labels)) else f"Set {i+1}"
        mc_list.append(_make_mcsamples(np.asarray(s), labels, label=lbl))

    g = plots.get_subplot_plotter(width_inch=10)
    g.settings.axes_fontsize = 12
    g.settings.lab_fontsize = 14
    g.settings.legend_fontsize = 12
    g.settings.figure_legend_frame = False

    plot_kwargs: Dict[str, Any] = {"filled": True, "legend_loc": "upper right"}
    if colors is not None:
        plot_kwargs["contour_colors"] = list(colors)

    g.triangle_plot(mc_list, **plot_kwargs)
    g.export(filename)
    print(f"✅ Comparison plot saved: {filename}")
