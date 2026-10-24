#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CosmoMatcher 1.4: constrained matching and covariance preservation.

GUI: python CosmoMatcher.py
CLI: python CosmoMatcher.py run --config examples/demo_sgl.json
API: from CosmoMatcher import CosmoBackend

The two-stage MILP first maximizes complete target coverage, then minimizes
the sum of squared endpoint mismatches at that fixed cardinality. Matching
uses a fiducial flat LCDM cosmology; it is not itself cosmology-independent.
Original scientific concept and v1.3 source: Jian Hu.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import platform
import gc
import subprocess
import tempfile
import shlex
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.integrate import quad
from scipy.optimize import Bounds, LinearConstraint, milp

__version__ = "1.4.3"
JLA_KEYS = ("v0", "va", "vb", "v0a", "v0b", "vab")
Log = Callable[[str], None]


class CosmoMatcherError(ValueError):
    """Invalid input, failed validation, or uncertified optimization."""


@dataclass
class RunResult:
    matched: pd.DataFrame
    index_map: pd.DataFrame
    matrices: dict[str, np.ndarray]
    summary: dict[str, Any]
    output_dir: Path


def _delimiter(line: str) -> str:
    for sep in (",", "\t", ";"):
        if sep in line:
            return sep
    return "whitespace"


def _tokens(line: str, sep: str) -> list[str]:
    if sep == "whitespace":
        return shlex.split(line, comments=False)
    return next(csv.reader([line], delimiter=sep, skipinitialspace=True))


def smart_read_csv(path, nrows=None, file_label="Data") -> pd.DataFrame:
    """Read a headered table without silently dropping malformed records.

    CSV, TSV, semicolon, whitespace, UTF-8 BOM and commented JLA headers are
    supported. Row positions in this parsed table are covariance positions
    unless an explicit covariance-index column is configured.
    """
    if not path:
        raise CosmoMatcherError(f"Missing {file_label} file.")
    raw = Path(path).read_bytes()
    content = None
    for enc in ("utf-8-sig", "gbk", "latin1"):
        try:
            content = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    lines = [(i + 1, s.strip()) for i, s in enumerate(content.splitlines()) if s.strip()]
    header_pos = None
    for pos, (_, line) in enumerate(lines):
        candidate = line.lstrip("#").strip()
        sep = _delimiter(candidate)
        names = _tokens(candidate, sep)
        if line.startswith("#"):
            known = {"name", "cid", "z", "zhd", "zcmb", "zl", "z_lens", "id"}
            if not (set(s.lower() for s in names) & known):
                continue
        header_pos = pos
        break
    if header_pos is None:
        raise CosmoMatcherError(f"{file_label}: no table header found.")
    if not names or len(set(names)) != len(names) or any(not n for n in names):
        raise CosmoMatcherError(f"{file_label}: empty or duplicate column names.")
    records = []
    for line_no, line in lines[header_pos + 1:]:
        if line.startswith("#"):
            continue
        row = _tokens(line, sep)
        if len(row) != len(names):
            raise CosmoMatcherError(
                f"{file_label}, line {line_no}: {len(row)} fields, expected {len(names)}. "
                "No records were skipped; fix the input before matching."
            )
        records.append(row)
        if nrows is not None and len(records) >= nrows:
            break
    if not records:
        raise CosmoMatcherError(f"{file_label}: no data rows.")
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(names)
    writer.writerows(records)
    buffer.seek(0)
    # Keep all identifiers lossless, including a user-configured custom ID column.
    # Scientific columns are converted explicitly when used, never by guessing.
    df = pd.read_csv(buffer, dtype=str)
    return df


def load_matrix(path, expected_n=None, square=True) -> np.ndarray:
    """Read arrays or text: N + N*N entries, or element count N*N + entries.

    A single-column text representation is reshaped in row-major order. The
    header is classified against the actual payload, never added to the matrix.
    """
    p = Path(path)
    if p.suffix.lower() == ".npy":
        data = np.load(p, allow_pickle=False)
    elif p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=False) as archive:
            key = "covariance" if "covariance" in archive else None
            if key is None:
                if len(archive.files) != 1:
                    raise CosmoMatcherError(f"{p.name}: .npz must have one array or a 'covariance' array.")
                key = archive.files[0]
            data = archive[key]
    else:
        with p.open(encoding="utf-8-sig") as handle:
            first = ""
            for line in handle:
                clean = line.split("#", 1)[0].strip()
                if clean:
                    first = clean
                    break
            if not first:
                raise CosmoMatcherError(f"{p.name}: empty matrix.")
            header = first.isdecimal()
            if header:
                declared = int(first)
                data = np.loadtxt(handle)
                if declared > 0 and data.size == declared * declared:
                    n = declared
                elif declared > 0 and data.size == declared and math.isqrt(declared)**2 == declared:
                    n = math.isqrt(declared)
                else:
                    raise CosmoMatcherError(
                        f"{p.name}: declared header {declared}, found {data.size} elements after it. "
                        "Expected either N followed by N*N entries, or N*N followed by N*N entries."
                    )
                data = data.reshape(n, n)
            else:
                sep = "," if "," in first else ";" if ";" in first else None
                data = np.loadtxt(p, delimiter=sep, encoding="utf-8-sig", ndmin=0 if square else 2)
    data = np.asarray(data, dtype=float)
    if data.ndim == 0:
        data = data.reshape(1, 1)
    if data.ndim == 1 and square:
        n = math.isqrt(data.size)
        if n * n != data.size:
            raise CosmoMatcherError(f"{p.name}: flattened matrix is not square.")
        data = data.reshape(n, n)
    if data.ndim != 2 or (square and data.shape[0] != data.shape[1]):
        raise CosmoMatcherError(f"{p.name}: invalid matrix shape {data.shape}.")
    if expected_n is not None and data.shape != (expected_n, expected_n):
        raise CosmoMatcherError(f"{p.name}: shape {data.shape}, expected {(expected_n, expected_n)}.")
    if not np.isfinite(data).all():
        raise CosmoMatcherError(f"{p.name}: non-finite matrix entries.")
    return data


def load_pantheon_matrix(path):
    return load_matrix(path)


def load_jla_matrix(path):
    return load_matrix(path)


def validate_covariance(matrix, label="covariance", check_psd=True, symmetry_rtol=1e-6) -> dict:
    c = np.asarray(matrix, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.isfinite(c).all():
        raise CosmoMatcherError(f"{label}: expected a finite square matrix.")
    if c.size == 0:
        return {"shape": [0, 0], "symmetric": True, "psd_checked": False}
    scale = max(float(np.max(np.abs(c))), np.finfo(float).tiny)
    if not np.isfinite(symmetry_rtol) or not 0 <= symmetry_rtol <= .001:
        raise CosmoMatcherError("symmetry_rtol must be between 0 and 0.001; large asymmetry cannot be silently repaired.")
    difference = np.abs(c-c.T)
    max_difference = float(np.max(difference))
    if max_difference > symmetry_rtol*scale:
        row, col = np.unravel_index(np.argmax(difference), c.shape)
        raise CosmoMatcherError(
            f"{label}: matrix is not symmetric after reshaping {len(c)}x{len(c)}. "
            f"Max |C-C.T|={difference[row,col]:.9g} at zero-based ({row},{col}); "
            f"C[row,col]={c[row,col]:.9g}, C[col,row]={c[col,row]:.9g}. "
            "Check that this is the correct covariance file and that its entries use row-major order. "
            "Single-column storage is supported; the header is excluded."
        )
    # Decimal text releases may differ by a few final digits across the diagonal.
    # Validate the averaged matrix; callers normalize explicitly and record this.
    if max_difference:
        c = .5*c+.5*c.T
    if np.min(np.diag(c)) < -scale * 1e-10:
        raise CosmoMatcherError(f"{label}: negative variance.")
    minimum = None
    if check_psd:
        try:
            np.linalg.cholesky(c)
        except np.linalg.LinAlgError:
            minimum = float(np.linalg.eigvalsh((c + c.T) / 2)[0])
            if minimum < -scale * max(1, len(c)) * 1e-10:
                raise CosmoMatcherError(f"{label}: not positive semidefinite (min eigenvalue={minimum:g}).")
    return {"shape": list(c.shape), "symmetric": True, "psd_checked": bool(check_psd),
            "minimum_eigenvalue_if_needed": minimum, "asymmetry_max_abs": max_difference,
            "asymmetry_relative_to_matrix_max": max_difference/scale,
            "symmetry_rtol": float(symmetry_rtol),
            "roundoff_symmetrized_for_validation": bool(max_difference)}


def normalize_covariance(matrix, label="covariance", check_psd=True, symmetry_rtol=1e-6):
    """Only average roundoff-level asymmetry, retaining explicit diagnostics."""
    matrix = np.asarray(matrix, dtype=float)
    report = validate_covariance(matrix, label, check_psd, symmetry_rtol)
    needed = report.get("roundoff_symmetrized_for_validation", False)
    report["roundoff_correction_applied"] = needed
    result = .5*matrix+.5*matrix.T if needed else matrix
    return result, report


def assemble_jla(components: dict, alpha: float, beta: float, statistical=None) -> np.ndarray:
    """JLA six-component covariance plus an explicitly specified statistical term.

    Cross-component matrices need not individually be positive semidefinite.
    alpha/beta are nuisance parameters; recompute this function when they vary.
    """
    missing = set(JLA_KEYS) - set(components)
    if missing:
        raise CosmoMatcherError(f"Missing JLA components: {sorted(missing)}")
    if not np.isfinite([alpha, beta]).all():
        raise CosmoMatcherError("JLA alpha and beta must be finite.")
    shape = np.asarray(components["v0"]).shape
    if any(np.asarray(components[k]).shape != shape for k in JLA_KEYS):
        raise CosmoMatcherError("All JLA components must have the same shape.")
    c = (components["v0"] + alpha**2 * components["va"] + beta**2 * components["vb"]
         + 2 * alpha * components["v0a"] - 2 * beta * components["v0b"]
         - 2 * alpha * beta * components["vab"])
    if statistical is not None:
        d = np.asarray(statistical, dtype=float)
        if d.ndim == 1:
            if d.shape != (shape[0],) or np.any(d < 0) or not np.isfinite(d).all():
                raise CosmoMatcherError("Invalid JLA statistical variance vector.")
            c = c + np.diag(d)
        elif d.shape == shape:
            c = c + d
        else:
            raise CosmoMatcherError("Invalid JLA statistical covariance shape.")
    return c


def jla_statistical_variance(table, alpha, beta, columns=None) -> np.ndarray:
    """Use JLA dmb/dx1/dcolour and intra-light-curve covariance terms.

    Official JLA dmb already includes the prescribed intrinsic/lensing/redshift
    contributions. Do not add those contributions a second time.
    """
    aliases = {"dmb": ("dmb",), "dx1": ("dx1",), "dc": ("dcolour", "dc"),
               "cov_m_s": ("cov_m_s", "cov_mb_x1"),
               "cov_m_c": ("cov_m_c", "cov_mb_c"),
               "cov_s_c": ("cov_s_c", "cov_x1_c")}
    resolved = {}
    for key, choices in aliases.items():
        name = (columns or {}).get(key) or next((a for a in choices if a in table.columns), None)
        if name is None:
            raise CosmoMatcherError(f"JLA statistical term needs column {key}; provide mappings or a statistical matrix.")
        resolved[key] = pd.to_numeric(table[name], errors="raise").to_numpy(float)
    if not all(np.isfinite(a).all() for a in resolved.values()):
        raise CosmoMatcherError("Non-finite JLA statistical inputs.")
    if any(np.any(resolved[k] < 0) for k in ("dmb", "dx1", "dc")):
        raise CosmoMatcherError("JLA uncertainty columns must be non-negative.")
    d = (resolved["dmb"]**2 + alpha**2 * resolved["dx1"]**2 + beta**2 * resolved["dc"]**2
         + 2*alpha*resolved["cov_m_s"] - 2*beta*resolved["cov_m_c"]
         - 2*alpha*beta*resolved["cov_s_c"])
    if np.any(d < 0):
        raise CosmoMatcherError("JLA statistical variance is negative; check covariance signs and units.")
    return d


@lru_cache(maxsize=65536)
def _distance(z: float, omega_m: float) -> float:
    return quad(lambda t: (omega_m*(1+t)**3 + 1-omega_m)**-0.5, 0, z,
                epsabs=1e-11, epsrel=1e-11)[0]


def comoving_distance(z, omega_m=0.3):
    """Dimensionless radial comoving distance H0*Dc/c in flat LCDM."""
    values = np.asarray(z, dtype=float)
    if not 0 <= omega_m <= 1 or not np.isfinite(omega_m):
        raise CosmoMatcherError("Omega_m must be finite and between 0 and 1.")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise CosmoMatcherError("Redshifts must be finite and non-negative.")
    flat = np.array([_distance(float(v), float(omega_m)) for v in values.ravel()])
    return float(flat[0]) if values.ndim == 0 else flat.reshape(values.shape)


def propagate_covariance(jacobian, covariance):
    j, c = np.asarray(jacobian, float), np.asarray(covariance, float)
    if j.ndim != 2 or c.shape != (j.shape[1], j.shape[1]):
        raise CosmoMatcherError("Jacobian and covariance dimensions do not agree.")
    if not np.isfinite(j).all() or not np.isfinite(c).all():
        raise CosmoMatcherError("Non-finite Jacobian or covariance.")
    return j @ c @ j.T


def luminosity_distance(mu, covariance=None):
    """DL in Mpc, and first-order propagated covariance, for calibrated mu."""
    mu = np.asarray(mu, float)
    if mu.ndim != 1 or not np.isfinite(mu).all():
        raise CosmoMatcherError("Distance moduli must be a finite 1-D vector.")
    d = np.power(10.0, (mu-25)/5)
    if not np.isfinite(d).all():
        raise CosmoMatcherError("Distance conversion overflow; check magnitudes and units.")
    j = np.diag(np.log(10)/5*d)
    return d, None if covariance is None else propagate_covariance(j, covariance)


def flat_lens_distance_ratio(m_l, m_s, z_l, z_s, covariance=None):
    """R=Dls/Ds in spatial flatness, with a common fixed SN absolute magnitude.

    Redshifts here are the actual matched SN redshifts. The returned estimate
    is transferred to the target anchors only through the matching approximation.
    A redshift-varying absolute magnitude, curvature, and redshift uncertainties
    require a different observable/Jacobian. Out-of-range R is never clipped.
    """
    ml, ms, zl, zs = [np.asarray(v, float) for v in (m_l, m_s, z_l, z_s)]
    if any(v.ndim != 1 or v.shape != ml.shape for v in (ms, zl, zs)):
        raise CosmoMatcherError("Lens-ratio vectors must have the same 1-D shape.")
    if not all(np.isfinite(v).all() for v in (ml, ms, zl, zs)) or np.any(zl <= 0) or np.any(zs <= zl):
        raise CosmoMatcherError("Lens-ratio inputs require finite magnitudes and 0 < z_l < z_s.")
    q = (1+zs)/(1+zl)*np.power(10.0, 0.2*(ml-ms))
    if not np.isfinite(q).all():
        raise CosmoMatcherError("Lens-ratio overflow.")
    j = np.hstack((-np.diag(np.log(10)/5*q), np.diag(np.log(10)/5*q)))
    return 1-q, None if covariance is None else propagate_covariance(j, covariance), j


@dataclass
class _Problem:
    A: Any
    lower: np.ndarray
    upper: np.ndarray
    costs: np.ndarray
    coverage: np.ndarray
    entries: np.ndarray
    formulation: str
    n_targets: int
    candidate_triplets: int
    edge_roles: np.ndarray | None = None
    edge_targets: np.ndarray | None = None
    edge_tracers: np.ndarray | None = None


def build_problem(target_z1, tracer_z, target_z2=None, *, tol=0.05, tol_type="dd",
                  omega_m=0.3, groups=None, formulation="triplet", max_candidates=2_000_000,
                  enforce_tracer_order=True):
    """Construct the explicit paper MILP or its equivalent compact formulation.

    Compact SGL uses target activation y_i and lens/source assignment edges.
    Each role sum equals y_i; each tracer resource is used at most once across
    both roles. This avoids enumerating the Cartesian product of endpoint lists.
    """
    z1, zt = np.asarray(target_z1, float), np.asarray(tracer_z, float)
    z2 = None if target_z2 is None else np.asarray(target_z2, float)
    if z1.ndim != 1 or zt.ndim != 1 or (z2 is not None and z2.shape != z1.shape):
        raise CosmoMatcherError("Invalid redshift array shapes.")
    if not np.isfinite(tol) or tol < 0 or tol_type not in {"dd", "dz"}:
        raise CosmoMatcherError("Tolerance must be non-negative; criterion is dd or dz.")
    if formulation not in {"triplet", "compact"}:
        raise CosmoMatcherError("Formulation must be triplet or compact.")
    if any(not np.isfinite(z).all() or np.any(z <= 0) for z in (z1, zt) + (() if z2 is None else (z2,))):
        raise CosmoMatcherError("Matching requires finite positive redshifts.")
    if z2 is not None and np.any(z2 <= z1):
        raise CosmoMatcherError("Every SGL requires z_source > z_lens.")
    nt, ns = len(z1), len(zt)
    g = np.arange(ns) if groups is None else np.asarray(groups)
    if g.shape != (ns,):
        raise CosmoMatcherError("Tracer group array length does not agree with the tracer catalogue.")
    g, labels = pd.factorize(g, sort=True)
    if np.any(g < 0):
        raise CosmoMatcherError("Missing tracer object identifiers.")
    ng = len(labels)
    if max_candidates < 1:
        raise CosmoMatcherError("max_candidates must be positive.")
    if tol_type == "dd":
        t, a = comoving_distance(zt, omega_m), comoving_distance(z1, omega_m)
        b = None if z2 is None else comoving_distance(z2, omega_m)
    else:
        t, a, b = zt, z1, z2
    # No shared mutable criterion: candidate construction is safe for concurrent API calls.
    def find(values):
        order = np.argsort(t, kind="stable")
        ordered = t[order]
        result = []
        for v in values:
            width = tol*v if tol_type == "dd" else tol
            lo = np.searchsorted(ordered, np.nextafter(v-width, -np.inf), side="left")
            hi = np.searchsorted(ordered, np.nextafter(v+width, np.inf), side="right")
            idx = np.sort(order[lo:hi])
            err = np.abs(t[idx]-v)/(v if tol_type == "dd" else 1)
            keep = err <= tol
            result.append((idx[keep], err[keep]**2))
        return result
    left = find(a)
    right = None if b is None else find(b)
    if z2 is not None and formulation == "compact":
        et, es, er, ec, forbidden = [], [], [], [], []
        triplet_count = 0
        for i, ((li, lc), (si, sc)) in enumerate(zip(left, right)):
            if len(li) and len(si):
                # Count feasible combinations without allocating a product array.
                lc_group = dict(zip(*np.unique(g[li], return_counts=True)))
                sc_group = dict(zip(*np.unique(g[si], return_counts=True)))
                triplet_count += len(li)*len(si)-sum(n*sc_group.get(k, 0) for k, n in lc_group.items())
                if enforce_tracer_order and np.max(zt[li]) >= np.min(zt[si]):
                    start_l, start_s = len(es), len(es)+len(li)
                    for il, j in enumerate(li):
                        invalid = np.flatnonzero((zt[si] <= zt[j]) & (g[si] != g[j]))
                        forbidden.extend((start_l+il, start_s+int(k)) for k in invalid)
                        triplet_count -= len(invalid)
                        if len(forbidden) > max_candidates:
                            raise CosmoMatcherError("Too many tracer-order constraints; reduce tolerance or use triplet formulation.")
            for role, (idx, cost) in enumerate((left[i], right[i])):
                et.extend([i]*len(idx)); es.extend(idx.tolist())
                er.extend([role]*len(idx)); ec.extend(cost.tolist())
        et, es, er = [np.asarray(v, dtype=int) for v in (et, es, er)]
        nv = nt + len(es)
        if nv > max_candidates:
            raise CosmoMatcherError(f"Compact model has {nv} variables; increase max_candidates explicitly.")
        edgecol = nt+np.arange(len(es))
        rows = np.concatenate((np.arange(2*nt), 2*et+er, 2*nt+g[es]))
        cols = np.concatenate((np.repeat(np.arange(nt), 2), edgecol, edgecol))
        vals = np.concatenate((-np.ones(2*nt), np.ones(2*len(es))))
        if forbidden:
            rows = np.r_[rows, np.repeat(2*nt+ng+np.arange(len(forbidden)), 2)]
            cols = np.r_[cols, nt+np.asarray(forbidden).ravel()]
            vals = np.r_[vals, np.ones(2*len(forbidden))]
        A = sparse.csc_matrix((vals, (rows, cols)), shape=(2*nt+ng+len(forbidden), nv))
        return _Problem(A, np.r_[np.zeros(2*nt), np.full(ng+len(forbidden), -np.inf)],
                        np.r_[np.zeros(2*nt), np.ones(ng+len(forbidden))], np.r_[np.zeros(nt), ec],
                        np.r_[np.ones(nt), np.zeros(len(es))], np.empty((0, 3), int),
                        "compact", nt, int(triplet_count), er, et, es)
    entries, costs = [], []
    for i, (li, lc) in enumerate(left):
        if z2 is None:
            entries.extend((i, int(j), -1) for j in li)
            costs.extend(lc.tolist())
        else:
            si, sc = right[i]
            # Check before allocating a potentially large Cartesian product.
            remaining = max_candidates-len(entries)
            if len(li)*len(si) > remaining:
                raise CosmoMatcherError("Triplet enumeration exceeds max_candidates; use formulation='compact'.")
            if not len(li) or not len(si):
                continue
            jl, js = np.repeat(li, len(si)), np.tile(si, len(li))
            c = np.repeat(lc, len(si)) + np.tile(sc, len(li))
            valid = g[jl] != g[js]
            if enforce_tracer_order:
                valid &= zt[jl] < zt[js]
            entries.extend(zip(np.full(np.sum(valid), i), jl[valid], js[valid]))
            costs.extend(c[valid].tolist())
        if len(entries) > max_candidates:
            raise CosmoMatcherError("Candidate limit exceeded; increase max_candidates explicitly.")
    e = np.asarray(entries, int).reshape(-1, 3)
    nv = len(e)
    rows = np.r_[e[:, 0], nt+g[e[:, 1]]]
    cols = np.tile(np.arange(nv), 2)
    if z2 is not None:
        rows = np.r_[rows, nt+g[e[:, 2]]]
        cols = np.r_[cols, np.arange(nv)]
    A = sparse.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(nt+ng, nv))
    return _Problem(A, np.full(nt+ng, -np.inf), np.ones(nt+ng), np.asarray(costs),
                    np.ones(nv), e, "triplet" if z2 is not None else "bipartite", nt, nv)


def solve_problem(problem: _Problem, *, time_limit=None, log_func=print, extra_constraints=None,
                  tie_break="catalogue_order", tie_cost_tolerance=1e-12):
    p = problem
    if tie_break not in {"catalogue_order", "none"}:
        raise CosmoMatcherError("tie_break must be catalogue_order or none.")
    if not np.isfinite(tie_cost_tolerance) or not 0 <= tie_cost_tolerance <= 1e-8:
        raise CosmoMatcherError("tie_cost_tolerance must lie between 0 and 1e-8.")
    fixed_lower, fixed_upper = np.zeros(len(p.costs)), np.ones(len(p.costs))
    base = [LinearConstraint(p.A, p.lower, p.upper)]
    if extra_constraints is not None:
        extra = [extra_constraints] if isinstance(extra_constraints, LinearConstraint) else list(extra_constraints)
        if any(not isinstance(item, LinearConstraint) or item.A.shape[1] != len(p.costs) for item in extra):
            raise CosmoMatcherError("Extra constraints must be LinearConstraint objects with one column per model variable.")
        base.extend(extra)
    if not len(p.costs):
        for constraint in base:
            ax = constraint.A @ np.empty(0)
            if np.any(ax < constraint.lb) or np.any(ax > constraint.ub):
                raise CosmoMatcherError("Empty model cannot satisfy the supplied constraints.")
        return np.empty((0, 3), int), {"maximum_cardinality": 0, "secondary_cost": 0.0,
                                      "stages": [], "optimal": True}
    options = {"mip_rel_gap": 0.0}
    if time_limit is not None:
        if not np.isfinite(time_limit) or time_limit <= 0:
            raise CosmoMatcherError("time_limit must be positive.")
        options["time_limit"] = float(time_limit)
    stages = []
    def solve(c, constraints, label, record=True):
        started = time.perf_counter()
        r = milp(c=c, constraints=constraints, bounds=Bounds(fixed_lower, fixed_upper),
                 integrality=np.ones(len(c), dtype=np.int8), options=options)
        if record:
            stages.append({"name": label, "status": int(r.status), "message": str(r.message),
                       "seconds": time.perf_counter()-started,
                       "mip_gap": None if getattr(r, "mip_gap", None) is None else float(r.mip_gap),
                       "dual_bound": None if getattr(r, "mip_dual_bound", None) is None else float(r.mip_dual_bound),
                       "objective": None if r.fun is None else float(r.fun)})
        if r.status != 0 or r.x is None:
            raise CosmoMatcherError(f"{label} has no certified optimum: {r.message}. Increase the time limit; no 'optimal' output is written.")
        x = np.rint(r.x).astype(int)
        if np.max(np.abs(x-r.x)) > 1e-5 or np.any(x < fixed_lower) or np.any(x > fixed_upper):
            raise CosmoMatcherError("Solver returned a non-integral or out-of-bounds assignment.")
        for constraint in constraints:
            ax = constraint.A @ x
            if np.any(ax < constraint.lb-1e-7) or np.any(ax > constraint.ub+1e-7):
                raise CosmoMatcherError("Solver assignment violates matching constraints.")
        return x
    log_func("MILP stage 1: maximize complete target coverage.")
    x = solve(-p.coverage, base, "maximum_cardinality")
    k = int(round(p.coverage @ x))
    if k and np.max(p.costs) > 0:
        log_func(f"MILP stage 2: minimize squared mismatch with exactly {k} targets.")
        cardinality = LinearConstraint(sparse.csc_matrix(p.coverage[None, :]), k, k)
        x = solve(p.costs/np.max(p.costs), base+[cardinality], "minimum_mismatch")
        if int(round(p.coverage @ x)) != k:
            raise CosmoMatcherError("Second stage changed the maximum cardinality.")
    # Canonicalize an optimal face in target catalogue order. No photometry enters
    # this ranking. Sequential fixing is lexicographic; a weighted rank sum is not.
    best_cost = float(p.costs @ x)
    canonical = {"rule": tie_break, "cost_tolerance": float(tie_cost_tolerance),
                 "steps": 0, "seconds": 0.0, "second_stage_cost": best_cost}
    if k and tie_break == "catalogue_order":
        begun = time.perf_counter()
        cardinality = LinearConstraint(sparse.csc_matrix(p.coverage[None, :]), k, k)
        canonical_constraints = base + [cardinality]
        if np.max(p.costs) > 0:
            # Scale this row to keep the declared cost tolerance above MILP's
            # feasibility roundoff, then verify the unscaled cost explicitly.
            coefficient = 1e6 / float(np.max(p.costs))
            canonical_constraints.append(LinearConstraint(
                sparse.csc_matrix((p.costs*coefficient)[None, :]),
                -np.inf, (best_cost+tie_cost_tolerance)*coefficient))
        if p.formulation == "compact":
            ns = int(np.max(p.edge_tracers, initial=-1))+1
        else:
            ns = int(np.max(p.entries[:, 1:], initial=-1))+1
        stride = ns+1
        for target in range(p.n_targets):
            rank = np.zeros(len(p.costs))
            if p.formulation == "compact":
                edges = np.flatnonzero(p.edge_targets == target)
                mask = np.r_[target, p.n_targets+edges]
                rank[target] = -stride**2
                rank[p.n_targets+edges] = p.edge_tracers[edges]*np.where(p.edge_roles[edges] == 0, stride, 1)
            else:
                mask = np.flatnonzero(p.entries[:, 0] == target)
                if not len(mask):
                    continue
                entries = p.entries[mask]
                rank[mask] = -stride**2+entries[:, 1]*stride+np.maximum(entries[:, 2], 0)
            x = solve(rank, canonical_constraints, f"canonical_target_{target}", record=False)
            fixed_lower[mask] = fixed_upper[mask] = x[mask]
            canonical["steps"] += 1
        canonical["seconds"] = time.perf_counter()-begun
        if int(round(p.coverage @ x)) != k or float(p.costs @ x) > best_cost+tie_cost_tolerance+1e-13:
            raise CosmoMatcherError("Canonical assignment exceeded the declared numerical cost tolerance.")
        log_func(f"Canonical catalogue-order assignment: {canonical['steps']} decisions, {canonical['seconds']:.3f} s.")
    canonical["cost_change"] = float(p.costs @ x)-best_cost
    if p.formulation == "compact":
        chosen = []
        active = np.flatnonzero(x[p.n_targets:])
        for i in np.flatnonzero(x[:p.n_targets]):
            edges = active[p.edge_targets[active] == i]
            l = edges[p.edge_roles[edges] == 0]
            s = edges[p.edge_roles[edges] == 1]
            if len(l) != 1 or len(s) != 1:
                raise CosmoMatcherError("Incomplete SGL activation in compact model.")
            chosen.append((i, p.edge_tracers[l[0]], p.edge_tracers[s[0]]))
        selected = np.asarray(chosen, int).reshape(-1, 3)
    else:
        selected = p.entries[x.astype(bool)]
    return selected, {"maximum_cardinality": k, "secondary_cost": float(p.costs @ x),
                      "stages": stages, "optimal": True, "canonicalization": canonical}


def greedy_match(z1, zt, z2=None, *, groups, tol, tol_type, omega_m,
                 enforce_tracer_order=True, max_candidates=2_000_000):
    """Transparent order-dependent baseline; single-target mode uses sorted edges."""
    p = build_problem(z1, zt, z2, groups=groups, tol=tol, tol_type=tol_type,
                      omega_m=omega_m, formulation="compact" if z2 is not None else "triplet",
                      enforce_tracer_order=enforce_tracer_order, max_candidates=max_candidates)
    used, assigned, selected = set(), set(), []
    if z2 is None:
        for pos in np.argsort(p.costs, kind="stable"):
            i, j, _ = p.entries[pos]
            if i not in assigned and groups[j] not in used:
                selected.append((i, j, -1)); assigned.add(i); used.add(groups[j])
    else:
        for i in range(len(z1)):
            def candidates(role):
                idx = np.flatnonzero((p.edge_targets == i) & (p.edge_roles == role))
                return sorted(idx, key=lambda k: (p.costs[p.n_targets+k], p.edge_tracers[k]))
            lenses, sources = candidates(0), candidates(1)
            pick = None
            # Do not abandon a target merely because its nearest lens choice blocks the source.
            for l in lenses:
                j = p.edge_tracers[l]
                if groups[j] in used:
                    continue
                for s in sources:
                    k = p.edge_tracers[s]
                    if (groups[k] not in used and groups[k] != groups[j]
                            and (not enforce_tracer_order or zt[j] < zt[k])):
                        pick = (i, j, k); break
                if pick is not None:
                    break
            if pick:
                selected.append(pick); used.update((groups[pick[1]], groups[pick[2]]))
    return np.asarray(selected, int).reshape(-1, 3), p


def _id_column(table, configured=None):
    if configured:
        if configured not in table:
            raise CosmoMatcherError(f"Missing tracer ID column: {configured}")
        return configured
    return next((n for n in ("CID", "SNID", "Name", "name", "ID", "id") if n in table), None)


def _prepare_tables(file_paths, col_cfg, params, mode, log):
    sn_raw = smart_read_csv(file_paths.get("tracer"), file_label="Tracer catalogue")
    obj_raw = smart_read_csv(file_paths.get("obj"), file_label="Target catalogue")
    if any(c.startswith("__cm_") for df in (sn_raw, obj_raw) for c in df.columns):
        raise CosmoMatcherError("Input column names beginning '__cm_' are reserved.")
    for df, names in ((sn_raw, [col_cfg.get("tracer_z_col")]),
                      (obj_raw, [col_cfg.get("z1_col")] + ([col_cfg.get("z2_col")] if mode == "sgls" else []))):
        if any(n is None or n not in df.columns for n in names):
            raise CosmoMatcherError(f"Redshift column missing; requested {names}, available {list(df.columns)}")
    sn, obj = sn_raw.copy(), obj_raw.copy()
    sn["__cm_row"] = np.arange(len(sn))
    obj["__cm_row"] = np.arange(len(obj))
    cov_col = col_cfg.get("cov_index_col")
    if cov_col:
        if cov_col not in sn:
            raise CosmoMatcherError(f"Missing covariance row column: {cov_col}")
        idx = pd.to_numeric(sn[cov_col], errors="raise").to_numpy(float)
        if not np.isfinite(idx).all() or np.any(idx < 0) or np.any(idx != np.floor(idx)) or len(set(idx)) != len(idx):
            raise CosmoMatcherError("Covariance indices must be unique, non-negative zero-based integers.")
        sn["__cm_cov"] = idx.astype(int)
    else:
        sn["__cm_cov"] = sn["__cm_row"]
        log("Covariance alignment: original catalogue row order. Supply cov_index_col for reordered/pre-filtered catalogues.")
    for df, columns in ((sn, [col_cfg["tracer_z_col"]]),
                        (obj, [col_cfg["z1_col"]] + ([col_cfg["z2_col"]] if mode == "sgls" else []))):
        for c in columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        valid = np.ones(len(df), bool)
        for c in columns:
            v = df[c].to_numpy(float)
            valid &= np.isfinite(v) & (v > 0)
        if df is obj and mode == "sgls":
            valid &= df[col_cfg["z2_col"]].to_numpy() > df[col_cfg["z1_col"]].to_numpy()
        if not valid.all():
            rejected = df.loc[~valid, "__cm_row"].tolist()
            policy = params.get("invalid_rows", "error")
            if policy != "drop":
                raise CosmoMatcherError(f"Invalid redshifts/geometry at original rows {rejected[:20]}; choose invalid_rows='drop' explicitly to exclude them.")
            log(f"Excluded {len(rejected)} invalid rows, preserving original covariance positions: {rejected[:20]}")
            df.drop(index=df.index[~valid], inplace=True)
        df.reset_index(drop=True, inplace=True)
    if sn.empty or obj.empty:
        raise CosmoMatcherError("No valid tracers or targets remain.")
    id_col = _id_column(sn, col_cfg.get("tracer_id_col"))
    policy = params.get("unique_by", "auto")
    if policy not in {"auto", "row", "object"}:
        raise CosmoMatcherError("unique_by must be auto, row, or object.")
    use_id = policy == "object" or (policy == "auto" and id_col is not None)
    if use_id:
        if id_col is None or sn[id_col].isna().any() or sn[id_col].astype(str).str.strip().eq("").any():
            raise CosmoMatcherError("Physical-object uniqueness requires a complete tracer_id_col.")
        groups = sn[id_col].astype(str).str.strip().to_numpy()
    else:
        groups = sn["__cm_row"].to_numpy()
    return sn_raw, obj_raw, sn, obj, groups, id_col, "object" if use_id else "row"


def _load_covariances(file_paths, col_cfg, params, sn_type, sn_raw, sn, log):
    components, matrix, kind = {}, None, params.get("covariance_kind", "total")
    if kind not in {"total", "systematic", "statistical"}:
        raise CosmoMatcherError("covariance_kind must be total, systematic, or statistical.")
    expected = None if col_cfg.get("cov_index_col") else len(sn_raw)
    if sn_type == "jla":
        present = [bool(file_paths.get(f"cov_jla_{k}")) for k in JLA_KEYS]
        if any(present):
            if not all(present):
                raise CosmoMatcherError("JLA covariance processing requires all six component files.")
            if "alpha" not in params or "beta" not in params:
                raise CosmoMatcherError("Provide explicit JLA alpha and beta for covariance assembly.")
            alpha, beta = float(params["alpha"]), float(params["beta"])
            for k in JLA_KEYS:
                components[k] = load_matrix(file_paths[f"cov_jla_{k}"], expected)
            n = len(components["v0"])
            stat = None
            stat_mode = params.get("jla_stat_mode", "auto")
            if file_paths.get("cov_jla_stat"):
                stat = load_matrix(file_paths["cov_jla_stat"], n)
                stat_mode = "file"
            elif stat_mode == "auto":
                if len(sn_raw) != n:
                    raise CosmoMatcherError("JLA auto statistical assembly needs the full catalogue in covariance order.")
                stat = jla_statistical_variance(sn_raw, alpha, beta, col_cfg.get("jla_stat_columns"))
                if col_cfg.get("cov_index_col"):
                    indices = pd.to_numeric(sn_raw[col_cfg["cov_index_col"]]).to_numpy(int)
                    ordered_stat = np.empty(n)
                    if set(indices) != set(range(n)):
                        raise CosmoMatcherError("JLA automatic statistical covariance needs every parent covariance row.")
                    ordered_stat[indices] = stat
                    stat = ordered_stat
            elif stat_mode != "none":
                raise CosmoMatcherError("jla_stat_mode must be auto or none, or provide cov_jla_stat.")
            if stat_mode == "none":
                kind = "jla_six_component_only"
                log("JLA statistical term omitted explicitly: output is a component combination, not a complete error budget.")
            else:
                kind = "total"
            matrix = assemble_jla(components, alpha, beta, stat)
    else:
        path = file_paths.get("covariance") or file_paths.get("cov_pantheon") or file_paths.get("cov_des")
        if path:
            matrix = load_matrix(path, expected)
    validation = None
    if matrix is not None:
        if np.max(sn["__cm_cov"].to_numpy()) >= len(matrix):
            raise CosmoMatcherError("Catalogue covariance indices exceed the parent matrix size.")
        matrix, validation = normalize_covariance(matrix, "parent SN covariance", params.get("check_psd", True), params.get("symmetry_rtol", 1e-6))
        if validation.get("roundoff_correction_applied"):
            log(f"Recorded roundoff symmetrization (C+C.T)/2: max difference={validation['asymmetry_max_abs']:.9g}.")
        log(f"Loaded {len(matrix)}x{len(matrix)} SN covariance ({kind}).")
    elif params.get("require_covariance", False):
        raise CosmoMatcherError("This run requires a covariance matrix; no complete covariance was supplied.")
    return matrix, components, kind, validation


def _hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _matrix_blocks(matrix, indices, k, sgl):
    full = matrix[np.ix_(indices, indices)]
    result = {"full": full}
    if sgl:
        result.update(lens=full[:k, :k], source=full[k:, k:], cross=full[:k, k:])
    else:
        result["matched"] = full
    return result


def _plot_run(out, stem, problem, matrices, matched, sgl):
    # Figure/Agg avoids GUI backends and imports no tkinter in HPC runs.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.ticker import MaxNLocator
    def save(fig, name):
        FigureCanvasAgg(fig)
        fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig = Figure(figsize=(8, 4.8)); ax = fig.subplots()
    if problem.A.shape[0] and problem.A.shape[1]:
        ax.spy(problem.A, markersize=4 if problem.A.shape[1] < 100 else .3, aspect="auto")
    ax.set(xlabel="Decision variables", ylabel="Constraints", title=f"{problem.formulation}: {problem.A.shape[0]} x {problem.A.shape[1]}")
    save(fig, f"matrix_A_sparsity_{stem}")
    fig = Figure(figsize=(7, 4.5)); ax = fig.subplots()
    columns = ["relative_comoving_error_l", "relative_comoving_error_s"] if sgl else ["relative_comoving_error"]
    maximum_error = max((float(matched[c].max()) for c in columns if len(matched)), default=0.)
    bins = np.linspace(0, max(maximum_error, 1e-4)*1.01, 21)
    for c in columns:
        ax.hist(matched[c].to_numpy(float), bins=bins, alpha=.6, label=c.rsplit("_", 1)[-1] if sgl else "matched")
    ax.set(xlabel="Absolute relative comoving-distance mismatch", ylabel="Endpoint count")
    ax.legend(); save(fig, f"mismatch_{stem}")
    for key in ("full", "lens", "source", "cross", "matched", "ratio", "joint"):
        if key not in matrices or not matrices[key].size:
            continue
        c = matrices[key]
        fig = Figure(figsize=(7.2, 5.6), layout="constrained"); ax = fig.subplots()
        signed = np.min(c) < 0
        limit = float(np.max(np.abs(c)))
        im = ax.imshow(c, cmap="coolwarm" if signed else "viridis", aspect="auto", interpolation="nearest",
                       vmin=-limit if signed else None, vmax=limit if signed else None)
        ax.set(xlabel="Column index", ylabel="Row index", title=key)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        unit = "dimensionless" if key == "ratio" else "mag²" if key in {"full", "lens", "source", "cross", "matched"} else "observable units squared"
        fig.colorbar(im, ax=ax, label=f"Covariance ({unit})")
        if key == "full" and sgl:
            boundary = c.shape[0]//2-.5
            ax.axhline(boundary, color="black", lw=.7, ls="--")
            ax.axvline(boundary, color="black", lw=.7, ls="--")
        save(fig, f"plot_cov_{stem}_{key}")


class CosmoBackend:
    comoving_distance = staticmethod(comoving_distance)

    @staticmethod
    def integrand(z, wm):
        return (wm*(1+z)**3 + 1-wm)**-0.5

    @staticmethod
    def run_main_logic(mode="sgls", sn_type="pantheon", algo="milp", tol_type="dd",
                       file_paths=None, col_cfg=None, params=None, log_func=print) -> RunResult:
        """Run and return tables, complete covariance blocks and a provenance report.

        Exceptions are raised to API/CLI callers, not swallowed. Output columns
        default to all original tracer columns. No GUI is imported or started.
        """
        start = time.perf_counter()
        file_paths, col_cfg, params = dict(file_paths or {}), dict(col_cfg or {}), dict(params or {})
        mode = "cluster" if mode in {"single", "cluster"} else mode
        sn_type = sn_type.lower()
        if mode not in {"sgls", "cluster"} or algo not in {"milp", "greedy"} or tol_type not in {"dd", "dz"}:
            raise CosmoMatcherError("Modes: sgls/cluster; algorithms: milp/greedy; criteria: dd/dz.")
        if sn_type not in {"pantheon", "pantheon+", "jla", "des", "des-dovekie", "generic"}:
            raise CosmoMatcherError("Tracer covariance format: pantheon, jla, des, or generic.")
        wm, tol = float(params.get("wm", .3)), float(params.get("tol", .05 if tol_type == "dd" else .005))
        if not 0 <= wm <= 1 or not np.isfinite(wm):
            raise CosmoMatcherError("Omega_m must lie between 0 and 1.")
        if params.get("invalid_rows", "error") not in {"drop", "error"}:
            raise CosmoMatcherError("invalid_rows must be error or drop.")
        log_func(f"CosmoMatcher {__version__}: {mode}, {algo}, {tol_type}, tolerance={tol:g}")
        sn_raw, obj_raw, sn, obj, groups, id_col, unique_by = _prepare_tables(file_paths, col_cfg, params, mode, log_func)
        c, components, covariance_kind, covariance_validation = _load_covariances(file_paths, col_cfg, params, sn_type, sn_raw, sn, log_func)
        load_seconds = time.perf_counter()-start
        zt = sn[col_cfg["tracer_z_col"]].to_numpy(float)
        z1 = obj[col_cfg["z1_col"]].to_numpy(float)
        z2 = obj[col_cfg["z2_col"]].to_numpy(float) if mode == "sgls" else None
        candidate_start = time.perf_counter()
        common = {"groups": groups, "tol": tol, "tol_type": tol_type, "omega_m": wm,
                  "max_candidates": int(params.get("max_candidates", 2_000_000)),
                  "enforce_tracer_order": bool(params.get("enforce_tracer_order", True))}
        if algo == "milp":
            problem = build_problem(z1, zt, z2, formulation=params.get("formulation", "compact"), **common)
            # Empty resource constraints contribute nothing and are safe to prune.
            active_rows = np.asarray(problem.A.getnnz(axis=1)).ravel() > 0
            problem.A = problem.A[active_rows].tocsc()
            problem.lower, problem.upper = problem.lower[active_rows], problem.upper[active_rows]
            candidate_seconds = time.perf_counter()-candidate_start
            solve_start = time.perf_counter()
            selected, solver = solve_problem(problem, time_limit=params.get("time_limit"), log_func=log_func,
                                             tie_break=params.get("tie_break", "catalogue_order"),
                                             tie_cost_tolerance=float(params.get("tie_cost_tolerance", 1e-12)))
            solve_seconds = time.perf_counter()-solve_start
        else:
            selected, problem = greedy_match(z1, zt, z2, **common)
            candidate_seconds = time.perf_counter()-candidate_start
            solve_seconds = 0.0
            solver = {"optimal": False, "note": "Order-dependent greedy baseline; not certified optimal."}
        if len(selected):
            selected = selected[np.argsort(selected[:, 0], kind="stable")]
        k = len(selected)
        ti = selected[:, 0]
        roles = ["sn_lens", "sn_source"] if mode == "sgls" else ["sn_matched"]
        chosen = [selected[:, 1], selected[:, 2]] if mode == "sgls" else [selected[:, 1]]
        all_chosen = np.concatenate(chosen)
        if len(set(groups[all_chosen])) != len(all_chosen):
            raise CosmoMatcherError("Post-solve validation found repeated tracer resources.")
        ds = comoving_distance(zt, wm)
        d1, d2 = comoving_distance(z1, wm), None if z2 is None else comoving_distance(z2, wm)
        matched = obj.iloc[ti].drop(columns=[n for n in obj if n.startswith("__cm_")]).reset_index(drop=True)
        if "obj_index" in matched.columns:
            matched.rename(columns={"obj_index": "input_obj_index"}, inplace=True)
        matched.insert(0, "obj_index", obj.iloc[ti]["__cm_row"].to_numpy(int))
        matched["target_z_l" if mode == "sgls" else "target_z"] = z1[ti]
        if mode == "sgls":
            matched["target_z_s"] = z2[ti]
        output_cols = col_cfg.get("output_cols")
        if output_cols is None:
            output_cols = [n for n in sn_raw.columns]
        if any(n not in sn for n in output_cols):
            raise CosmoMatcherError("An output column does not exist in the tracer catalogue.")
        map_rows, tracer_output = [], {}
        for role_no, (role, idx) in enumerate(zip(roles, chosen)):
            baseline_z = z1 if role_no == 0 else z2
            baseline_d = d1 if role_no == 0 else d2
            for name in dict.fromkeys(output_cols):
                out_name = f"{role}_{name}"
                if out_name in matched or out_name in tracer_output:
                    raise CosmoMatcherError(f"Output column collision: {out_name}")
                tracer_output[out_name] = sn.iloc[idx][name].to_numpy()
            tracer_output[f"{role}_idx"] = sn.iloc[idx]["__cm_cov"].to_numpy(int)
            tracer_output[f"{role}_original_index"] = sn.iloc[idx]["__cm_cov"].to_numpy(int)
            tracer_output[f"{role}_redshift"] = zt[idx]
            suffix = "_l" if role_no == 0 else "_s"
            if mode != "sgls":
                suffix = ""
            tracer_output[f"delta_z{suffix}"] = zt[idx]-baseline_z[ti]
            tracer_output[f"relative_comoving_error{suffix}"] = np.abs(ds[idx]-baseline_d[ti])/baseline_d[ti]
            for paired_row, j in enumerate(idx):
                map_rows.append({"matrix_position": role_no*k+paired_row, "matched_row": paired_row,
                                 "role": role, "catalogue_row": int(sn.iloc[j]["__cm_row"]),
                                 "covariance_row": int(sn.iloc[j]["__cm_cov"]),
                                 "tracer_id": str(sn.iloc[j][id_col]) if id_col else None,
                                 "redshift": float(zt[j])})
        matched = pd.concat([matched, pd.DataFrame(tracer_output, index=matched.index)], axis=1)
        index_map = pd.DataFrame(map_rows, columns=["matrix_position", "matched_row", "role", "catalogue_row", "covariance_row", "tracer_id", "redshift"])
        cov_indices = sn.iloc[all_chosen]["__cm_cov"].to_numpy(int)
        matrices = {} if c is None else _matrix_blocks(c, cov_indices, k, mode == "sgls")
        if c is not None:
            validate_covariance(matrices["full"], "matched SN covariance", params.get("check_psd", True))
            for component, source in components.items():
                for name, block in _matrix_blocks(source, cov_indices, k, mode == "sgls").items():
                    matrices[f"jla_{component}_{name}"] = block
        # Optional true target-observable covariance and cross-probe covariance.
        target_covariance_validation = None
        if file_paths.get("target_covariance"):
            target_c = load_matrix(file_paths["target_covariance"], len(obj_raw))
            target_c, target_covariance_validation = normalize_covariance(target_c, "target covariance", params.get("check_psd", True), params.get("symmetry_rtol", 1e-6))
            target_idx = obj.iloc[ti]["__cm_row"].to_numpy(int)
            matrices["target"] = target_c[np.ix_(target_idx, target_idx)]
            if c is not None:
                cross = np.zeros((k, len(cov_indices)))
                if file_paths.get("target_sn_cross"):
                    parent_cross = load_matrix(file_paths["target_sn_cross"], square=False)
                    if parent_cross.shape != (len(obj_raw), len(c)):
                        raise CosmoMatcherError("target_sn_cross must have shape (parent targets, parent SN covariance rows).")
                    cross = parent_cross[np.ix_(target_idx, cov_indices)]
                matrices["target_sn_cross"] = cross
                matrices["joint"] = np.block([[matrices["target"], cross], [cross.T, matrices["full"]]])
                validate_covariance(matrices["joint"], "joint target/SN covariance", params.get("check_psd", True))
        if file_paths.get("target_sn_cross") and (not file_paths.get("target_covariance") or c is None):
            raise CosmoMatcherError("Cross-probe assembly needs both target covariance and SN covariance.")
        magnitude_col, mu_col = col_cfg.get("magnitude_col"), col_cfg.get("mu_col")
        observable_notes = []
        if magnitude_col and magnitude_col not in sn:
            raise CosmoMatcherError(f"Missing magnitude column: {magnitude_col}")
        if mu_col and mu_col not in sn:
            raise CosmoMatcherError(f"Missing distance-modulus column: {mu_col}")
        if magnitude_col:
            mag = pd.to_numeric(sn[magnitude_col], errors="raise").to_numpy(float)
            if sn_type == "jla" and magnitude_col == "mb" and params.get("jla_standardize", True):
                color = col_cfg.get("jla_colour_col") or next((n for n in ("colour", "c") if n in sn), None)
                stretch = col_cfg.get("jla_stretch_col", "x1")
                if color is None or stretch not in sn or "alpha" not in params or "beta" not in params:
                    raise CosmoMatcherError("Standardizing JLA mb requires x1, colour, alpha and beta.")
                mag = mag + float(params["alpha"])*sn[stretch].to_numpy(float)-float(params["beta"])*sn[color].to_numpy(float)
                delta_m = float(params.get("delta_M", 0.0))
                if delta_m:
                    mass = col_cfg.get("jla_host_mass_col", "3rdvar")
                    if mass not in sn:
                        raise CosmoMatcherError("Nonzero JLA host step needs the host-mass column.")
                    mag = mag-delta_m*(sn[mass].to_numpy(float) >= 10)
                observable_notes.append("JLA mb standardized with configured alpha, beta and fixed delta_M.")
            if not np.isfinite(mag[all_chosen]).all():
                raise CosmoMatcherError("Selected standardized magnitudes are non-finite.")
            if mode == "sgls" and k:
                ratio, ratio_c, _ = flat_lens_distance_ratio(mag[chosen[0]], mag[chosen[1]],
                                                            zt[chosen[0]], zt[chosen[1]], matrices.get("full"))
                matched["R_flat"] = ratio
                matched["R_flat_physical"] = (ratio > 0) & (ratio < 1)
                if ratio_c is not None:
                    matrices["ratio"] = ratio_c
                observable_notes.append("R_flat assumes spatial flatness, common SN absolute magnitude, exact redshifts, and first-order covariance propagation; target-transfer mismatch is not included as extra noise.")
            if "absolute_magnitude" in params and not mu_col:
                mu = mag-float(params["absolute_magnitude"])
            else:
                mu = None
        else:
            mu = None
        if mu_col:
            mu = pd.to_numeric(sn[mu_col], errors="raise").to_numpy(float)
        if mu is not None and k:
            distance, distance_c = luminosity_distance(mu[all_chosen], matrices.get("full"))
            for role_no, role in enumerate(roles):
                matched[f"{role}_DL_Mpc"] = distance[role_no*k:(role_no+1)*k]
            if distance_c is not None:
                matrices["luminosity_distance"] = distance_c
            observable_notes.append("DL covariance uses a fixed calibration/absolute magnitude; additional calibration nuisance uncertainty must be supplied by the scientific analysis.")
        errors = np.concatenate([matched[n].to_numpy(float) for n in matched if n.startswith("relative_comoving_error")])
        all_ids = sn.iloc[all_chosen][id_col].astype(str).str.strip() if id_col else None
        summary = {"version": __version__, "mode": mode, "sn_type": sn_type, "algorithm": algo,
                   "criterion": tol_type, "tolerance": tol, "omega_m": wm,
                   "input_target_rows": len(obj_raw), "valid_targets": len(obj),
                   "input_tracer_rows": len(sn_raw), "valid_tracers": len(sn),
                   "matched_targets": k, "yield_percent": 100*k/len(obj),
                   "distinct_catalogue_rows": len(set(all_chosen)),
                   "distinct_physical_ids": None if all_ids is None else int(all_ids.nunique()),
                   "tracer_id_column": id_col, "uniqueness": unique_by,
                   "enforce_tracer_order": common["enforce_tracer_order"],
                   "candidate_triplets_or_pairs": problem.candidate_triplets,
                   "variables": len(problem.costs), "constraints": problem.A.shape[0], "nonzeros": problem.A.nnz,
                   "formulation": problem.formulation, "solver": solver,
                   "mean_relative_comoving_error": None if not len(errors) else float(np.mean(errors)),
                   "max_relative_comoving_error": None if not len(errors) else float(np.max(errors)),
                   "covariance_kind": covariance_kind if c is not None else "not_supplied",
                   "covariance_validation": covariance_validation,
                   "target_covariance_validation": target_covariance_validation,
                   "covariance_alignment": "explicit_zero_based_index" if col_cfg.get("cov_index_col") else "original_catalogue_row_order",
                   "cross_probe_assumption": "supplied" if file_paths.get("target_sn_cross") else "zero_assumed" if "joint" in matrices else "not_assembled",
                   "observable_notes": observable_notes,
                   "matrix_order": "all lens-role SN, then all source-role SN" if mode == "sgls" else "matched target order",
                   "timing_seconds": {"load_validate": load_seconds, "candidate_or_greedy": candidate_seconds, "milp_solve": solve_seconds},
                   "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
                   "configuration": {"file_paths": file_paths, "col_cfg": col_cfg, "params": params},
                   "inputs": {name: {"path": str(Path(path).resolve()), "sha256": _hash_file(path)}
                              for name, path in file_paths.items() if path}}
        out = Path(params.get("output_dir") or f"cosmomatcher_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()%1_000_000_000:09d}").resolve()
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{mode}_{algo}_{tol_type}_{sn_type}"
        export_start = time.perf_counter()
        matched.to_csv(out / f"matched_{stem}.csv", index=False)
        index_map.to_csv(out / f"index_map_{stem}.csv", index=False)
        for key, matrix in matrices.items():
            np.savetxt(out / f"cov_{stem}_{key}.txt", matrix, fmt="%.16e")
        if matrices:
            np.savez_compressed(out / f"cov_{stem}.npz", **matrices)
        sparse.save_npz(out / "constraint_matrix.npz", problem.A)
        summary["timing_seconds"]["export"] = time.perf_counter()-export_start
        plot_start = time.perf_counter()
        if params.get("plots", True):
            try:
                _plot_run(out, stem, problem, matrices, matched, mode == "sgls")
            except Exception as exc:
                summary["plot_warning"] = str(exc)
                log_func(f"Plot warning (numerical results saved): {exc}")
        summary["timing_seconds"]["plots"] = time.perf_counter()-plot_start
        summary["timing_seconds"]["total"] = time.perf_counter()-start
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            summary["process_lifetime_peak_rss_MiB"] = rss/(1024**2 if sys.platform == "darwin" else 1024)
        except ImportError:
            pass
        (out / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
        log_func(f"Finished: {k}/{len(obj)} targets, {len(all_chosen)} tracer rows; results: {out}")
        return RunResult(matched, index_map, matrices, summary, out)


def run_config(config, log_func=print):
    return CosmoBackend.run_main_logic(**config, log_func=log_func)


def _comparison_record(result, label):
    fields = ("algorithm", "criterion", "tolerance", "omega_m", "matched_targets", "yield_percent",
              "distinct_catalogue_rows", "distinct_physical_ids", "mean_relative_comoving_error",
              "max_relative_comoving_error", "candidate_triplets_or_pairs", "variables", "formulation")
    row = {"label": label, **{k: result.summary[k] for k in fields}}
    row["solve_seconds"] = result.summary["timing_seconds"]["milp_solve"]
    row["output_directory"] = str(result.output_dir)
    return row


def run_ablation(config, dd_tolerance=.05, dz_tolerance=.005, log_func=print):
    """Run all four algorithm/criterion combinations, with isolated outputs."""
    base = copy.deepcopy(config)
    parent = Path(base.get("params", {}).get("output_dir", "ablation_results")).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    rows, results = [], []
    for criterion, tolerance in (("dz", dz_tolerance), ("dd", dd_tolerance)):
        for algorithm in ("greedy", "milp"):
            case = copy.deepcopy(base)
            case.update(algo=algorithm, tol_type=criterion)
            label = f"{algorithm}_{criterion}"
            case.setdefault("params", {}).update(tol=float(tolerance), output_dir=str(parent / label))
            result = run_config(case, log_func)
            rows.append(_comparison_record(result, label)); results.append(result)
    table = pd.DataFrame(rows)
    table.to_csv(parent / "ablation_summary.csv", index=False)
    if base.get("params", {}).get("plots", True):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(8, 4.5)); FigureCanvasAgg(fig); ax = fig.subplots()
        positions = np.arange(2)
        for offset, algorithm in ((-.18, "greedy"), (.18, "milp")):
            data = table.loc[table.algorithm == algorithm]
            bars = ax.bar(positions+offset, data.matched_targets, .36, label=algorithm.upper())
            ax.bar_label(bars)
        ax.set(xticks=positions, xticklabels=[f"Delta z <= {dz_tolerance:g}", f"Delta d/d <= {dd_tolerance:g}"], ylabel="Matched targets")
        ax.legend(); fig.tight_layout()
        fig.savefig(parent / "ablation_yield.png", dpi=180); fig.savefig(parent / "ablation_yield.pdf")
        fig = Figure(figsize=(8, 4.5)); FigureCanvasAgg(fig); ax = fig.subplots()
        for result in results:
            if result.summary["criterion"] != "dd":
                continue
            columns = [n for n in result.matched if n.startswith("relative_comoving_error")]
            values = np.concatenate([result.matched[n].to_numpy(float) for n in columns])
            ax.hist(values, bins=np.linspace(0, max(dd_tolerance, 1e-12), 21), alpha=.5, label=result.summary["algorithm"].upper())
        ax.set(xlabel="Absolute relative comoving-distance mismatch", ylabel="Matched endpoints")
        ax.legend(); fig.tight_layout()
        fig.savefig(parent / "ablation_mismatch.png", dpi=180); fig.savefig(parent / "ablation_mismatch.pdf")
    return table


def _jaccard(a, b):
    return len(a & b)/len(a | b) if a or b else 1.0


def run_sensitivity(config, tolerances=(.03, .04, .05, .06, .07), omega_values=(.28, .3, .32), log_func=print):
    """Sweep tolerance and fiducial Omega_m; compare members as well as yield."""
    base = copy.deepcopy(config)
    parent = Path(base.get("params", {}).get("output_dir", "sensitivity_results")).resolve()
    fiducial = (float(base.get("params", {}).get("tol", .05)), float(base.get("params", {}).get("wm", .3)))
    combinations = list(dict.fromkeys([fiducial]+[(float(t), float(w)) for w in omega_values for t in tolerances]))
    rows = []
    reference_targets, reference_assignments = None, None
    for case_no, (t, w) in enumerate(combinations):
        case = copy.deepcopy(base)
        case.setdefault("params", {}).update(tol=t, wm=w, output_dir=str(parent / f"case_{case_no:03d}"))
        result = run_config(case, log_func)
        targets = set(result.matched["obj_index"].tolist())
        role_columns = [n for n in ("sn_lens_idx", "sn_source_idx", "sn_matched_idx") if n in result.matched]
        assignments = set(result.matched[["obj_index"]+role_columns].itertuples(index=False, name=None))
        if reference_targets is None:
            reference_targets, reference_assignments = targets, assignments
        row = _comparison_record(result, f"case_{case_no:03d}")
        row.update(fiducial=case_no == 0, target_jaccard=_jaccard(targets, reference_targets),
                   assignment_jaccard=_jaccard(assignments, reference_assignments))
        rows.append(row)
    table = pd.DataFrame(rows)
    parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(parent / "sensitivity_summary.csv", index=False)
    return table


def window_gls(redshifts, values, covariance, anchors, *, tolerance=.05, omega_m=.3,
               linear=True, check_psd=True):
    """Optional all-candidate benchmark with overlap covariance W C W.T.

    This is not the one-SN matching protocol. Local linear GLS estimates the
    observable at each anchor, using all points in its comoving-distance window.
    A constant-window estimate is available with linear=False. Singular local
    covariance or an unidentified slope is rejected rather than regularized
    silently. The compressed covariance can be singular for overlapping windows;
    callers must not blindly invert it or multiply independent window likelihoods.
    """
    z, y, anchors = [np.asarray(v, float) for v in (redshifts, values, anchors)]
    c = np.asarray(covariance, float)
    if z.ndim != 1 or y.shape != z.shape or anchors.ndim != 1 or not len(anchors) or c.shape != (len(z), len(z)):
        raise CosmoMatcherError("Invalid window-GLS dimensions.")
    if not np.isfinite(y).all() or not np.isfinite(tolerance) or tolerance < 0 or np.any(z <= 0) or np.any(anchors <= 0):
        raise CosmoMatcherError("Window GLS requires finite values, positive redshifts, and a non-negative tolerance.")
    c, covariance_validation = normalize_covariance(c, "window GLS parent covariance", check_psd)
    d, da = comoving_distance(z, omega_m), comoving_distance(anchors, omega_m)
    W = np.zeros((len(anchors), len(z)))
    counts = []
    for i, anchor in enumerate(anchors):
        chosen = np.flatnonzero(np.abs(d-da[i])/da[i] <= tolerance)
        counts.append(len(chosen))
        if len(chosen) < (2 if linear else 1):
            raise CosmoMatcherError(f"Anchor {i} has too few candidates for the requested local fit.")
        X = np.c_[np.ones(len(chosen)), z[chosen]-anchor] if linear else np.ones((len(chosen), 1))
        local = c[np.ix_(chosen, chosen)]
        try:
            np.linalg.cholesky(local)
            ci_x = np.linalg.solve(local, X)
            normal = X.T @ ci_x
            if np.linalg.matrix_rank(normal) != normal.shape[0]:
                raise np.linalg.LinAlgError("unidentified local slope")
            W[i, chosen] = np.linalg.solve(normal, ci_x.T)[0]
        except np.linalg.LinAlgError as exc:
            raise CosmoMatcherError(f"Anchor {i}: invalid local GLS system ({exc}).") from exc
    compressed = W @ c @ W.T
    return {"estimates": W @ y, "covariance": compressed, "weights": W,
            "candidate_counts": np.asarray(counts), "covariance_rank": int(np.linalg.matrix_rank(compressed)),
            "covariance_validation": covariance_validation,
            "note": "Local smoothness and fiducial matching cosmology are assumptions; redshift-transfer bias is not automatically included."}


def read_config(path):
    p = Path(path).resolve()
    config = json.loads(p.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or any(not isinstance(config.get(k, {}), dict) for k in ("file_paths", "col_cfg", "params")):
        raise CosmoMatcherError("Configuration and its file_paths/col_cfg/params sections must be JSON objects.")
    allowed = {"mode", "sn_type", "algo", "tol_type", "file_paths", "col_cfg", "params"}
    unknown = set(config)-allowed
    if unknown:
        raise CosmoMatcherError(f"Unknown configuration sections: {sorted(unknown)}")
    for name, value in config.get("file_paths", {}).items():
        if value and not Path(value).is_absolute():
            config["file_paths"][name] = str((p.parent / value).resolve())
    out = config.get("params", {}).get("output_dir")
    if out and not Path(out).is_absolute():
        config["params"]["output_dir"] = str((p.parent / out).resolve())
    return config


@dataclass
class _GUIJob:
    """Plain process state: never owns a Tk object or a GUI callback."""
    temporary: Any
    process: subprocess.Popen
    log_path: Path
    output_dir: Path
    offset: int = 0
    pending: bytes = b""


def _start_gui_job(config, kind="run", dd=.05, dz=.005,
                   tolerances=(.03, .04, .05, .06, .07), omegas=(.28, .3, .32)):
    if kind not in {"run", "ablation", "sensitivity"}:
        raise CosmoMatcherError(f"Unknown GUI task: {kind}")
    # JSON is the process boundary; no widgets, Tk variables or callbacks cross it.
    plain = json.loads(json.dumps(config, ensure_ascii=False, allow_nan=False))
    for key, value in plain.setdefault("file_paths", {}).items():
        if value:
            plain["file_paths"][key] = str(Path(value).resolve())
    params = plain.setdefault("params", {})
    output = Path(params.get("output_dir") or
                  f"cosmomatcher_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()%1_000_000_000:09d}").resolve()
    params["output_dir"] = str(output)
    temporary = tempfile.TemporaryDirectory(prefix="cosmomatcher_gui_")
    try:
        folder = Path(temporary.name)
        config_path, log_path = folder / "config.json", folder / "worker.log"
        config_path.write_text(json.dumps(plain, ensure_ascii=False, indent=2), encoding="utf-8")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), kind, "--config", str(config_path)]
        if kind == "ablation":
            command += ["--dd-tolerance", str(dd), "--dz-tolerance", str(dz)]
        elif kind == "sensitivity":
            command += ["--tolerances", ",".join(map(str, tolerances)),
                        "--omega-values", ",".join(map(str, omegas))]
        env = os.environ.copy()
        env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        with log_path.open("wb") as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, env=env, shell=False)
        return _GUIJob(temporary, process, log_path, output)
    except Exception:
        temporary.cleanup()
        raise


def _read_gui_job(job):
    """Read only available bytes from a regular file, on Tk's main thread."""
    with job.log_path.open("rb") as stream:
        stream.seek(job.offset)
        chunk = stream.read()
        job.offset += len(chunk)
    pieces = (job.pending + chunk).split(b"\n")
    job.pending = pieces.pop()
    if job.process.poll() is not None and job.pending:
        pieces.append(job.pending)
        job.pending = b""
    return [line.decode("utf-8", errors="replace").rstrip("\r") for line in pieces]


def _cleanup_gui_job(job):
    """Reap a process before deleting its temporary log/config files."""
    if job.process.poll() is None:
        job.process.terminate()
        try:
            job.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            job.process.kill()
            job.process.wait(timeout=3)
    job.temporary.cleanup()


def launch_gui(initial_config=None):
    """Classic interface; Tk lives on the main thread, computation in a process."""
    if threading.current_thread() is not threading.main_thread():
        raise CosmoMatcherError("GUI must run on the Python main thread. Launch python CosmoMatcher.py in a terminal.")
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except ImportError as exc:
        raise CosmoMatcherError("Tkinter is unavailable. Use the run CLI or install Python Tk.") from exc
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise CosmoMatcherError("No graphical display. Use 'run --config ...' or open the GUI locally.") from exc
    root.title(f"CosmoMatcher V{__version__} (Classic Interface)")
    root.geometry(f"1000x{min(980, max(650, root.winfo_screenheight()-100))}")
    root.minsize(800, 650)
    paths = {k: tk.StringVar(master=root) for k in ("tracer", "obj", "covariance", "output_dir", "cov_jla_stat", "target_covariance", "target_sn_cross")}
    paths.update({f"cov_jla_{k}": tk.StringVar(master=root) for k in JLA_KEYS})
    defaults = {"mode": "sgls", "sn_type": "pantheon", "algo": "milp", "tol_type": "dd",
                "wm": "0.300", "tol": "0.05", "formulation": "compact", "unique_by": "auto", "tie_break": "catalogue_order",
                "alpha": "0.14", "beta": "3.1", "jla_stat_mode": "auto", "time_limit": "",
                "absolute_magnitude": "", "dd_tolerance": "0.05", "dz_tolerance": "0.005",
                "tolerances": "0.03,0.04,0.05,0.06,0.07", "omega_values": "0.28,0.3,0.32"}
    variables = {k: tk.StringVar(master=root, value=v) for k, v in defaults.items()}
    variables.update({k: tk.BooleanVar(master=root, value=v) for k, v in (("plots", True), ("require_covariance", False), ("drop_invalid", False))})
    columns = {k: tk.StringVar(master=root) for k in ("tracer_z_col", "z1_col", "z2_col", "tracer_id_col", "cov_index_col", "magnitude_col", "mu_col")}
    preserved = {}
    tracer_names, target_names = [], []
    widgets, buttons = {}, []
    state = {"job": None, "after_id": None, "closing": False, "close_started": None, "tail": []}
    wrapper = ttk.Frame(root); wrapper.pack(fill="both", expand=True)
    canvas = tk.Canvas(wrapper, highlightthickness=0)
    scrollbar = ttk.Scrollbar(wrapper, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y"); canvas.pack(side="left", fill="both", expand=True)
    body = ttk.Frame(canvas); window = canvas.create_window((0, 0), window=body, anchor="nw")
    body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
    top = ttk.Frame(body); top.pack(fill="x", padx=10, pady=5)
    f_mode = ttk.LabelFrame(top, text="Matching Mode", padding=5)
    f_mode.grid(row=0, column=0, padx=5, pady=2, sticky="ew")
    f_sn = ttk.LabelFrame(top, text="Supernova Sample Type", padding=5)
    f_sn.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
    f_ablation = ttk.LabelFrame(top, text="Ablation Study Settings", padding=5)
    f_ablation.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
    for col, (text, value) in enumerate((("MILP Global Optimization", "milp"), ("Greedy Algorithm", "greedy")), 1):
        ttk.Radiobutton(f_ablation, text=text, variable=variables["algo"], value=value).grid(row=0, column=col, padx=5)
    ttk.Label(f_ablation, text="Algorithm:").grid(row=0, column=0, padx=5)
    ttk.Label(f_ablation, text="| Tolerance Criterion:").grid(row=0, column=3, padx=5)
    for col, (text, value) in enumerate((("Δd/d (Comoving Distance)", "dd"), ("Δz (Absolute Redshift)", "dz")), 4):
        ttk.Radiobutton(f_ablation, text=text, variable=variables["tol_type"], value=value).grid(row=0, column=col, padx=5)
    f_files = ttk.LabelFrame(body, text="1. Data Loading", padding=10); f_files.pack(fill="x", padx=10, pady=2)
    f_cov = ttk.LabelFrame(body, text="2. Covariance Matrix Settings (Required for Joint Constraints)", padding=10)
    f_cov.pack(fill="x", padx=10, pady=2)
    f_cols = ttk.LabelFrame(body, text="3. Redshift Column Mapping", padding=10); f_cols.pack(fill="x", padx=10, pady=2)
    for key, label in (("z1_col", "Lens Redshift (zl):"), ("z2_col", "Source Redshift (zs):"), ("tracer_z_col", "SN Data Redshift (z):")):
        widgets[key+"_label"] = ttk.Label(f_cols, text=label)
        widgets[key] = ttk.Combobox(f_cols, textvariable=columns[key], width=13, state="readonly")
    f_out = ttk.LabelFrame(body, text="4. Output Columns to Keep", padding=10); f_out.pack(fill="x", padx=10, pady=4)
    list_frame = ttk.Frame(f_out); list_frame.pack(fill="x")
    list_scroll = ttk.Scrollbar(list_frame, orient="vertical")
    box = tk.Listbox(list_frame, selectmode="multiple", height=5, exportselection=False, yscrollcommand=list_scroll.set)
    list_scroll.configure(command=box.yview); list_scroll.pack(side="right", fill="y"); box.pack(side="left", fill="x", expand=True)
    def select_defaults():
        recommended = {"cid", "idsurvey", "zhd", "m_b_corr", "m_b_corr_err_diag", "mu_sh0es", "name", "z", "mb", "x1", "c", "colour", "dmb", "dx1", "dc", "dcolour", "zcmb"}
        box.selection_clear(0, "end")
        for i, name in enumerate(tracer_names):
            if name.lower() in recommended:
                box.selection_set(i)
    ttk.Button(f_out, text="Select Recommended Columns", command=select_defaults).pack(pady=3)
    f_run = ttk.LabelFrame(body, text="5. Execution Parameters", padding=10); f_run.pack(fill="x", padx=10, pady=4)
    ttk.Label(f_run, text="Omega_m:").pack(side="left")
    ttk.Entry(f_run, textvariable=variables["wm"], width=8).pack(side="left", padx=5)
    ttk.Label(f_run, text="Tolerance Value (e.g., 0.05 or 0.005):").pack(side="left", padx=(10, 0))
    ttk.Entry(f_run, textvariable=variables["tol"], width=8).pack(side="left", padx=5)
    log_widget = tk.Text(root, height=8, state="disabled"); log_widget.pack(fill="x", padx=10, pady=5)
    footer = ttk.Frame(root); footer.pack(fill="x", padx=10, pady=(0, 8))
    status = tk.StringVar(master=root, value="Ready / 就绪"); ttk.Label(footer, textvariable=status).pack(side="left", padx=5)
    def log(message):
        log_widget.configure(state="normal"); log_widget.insert("end", message+"\n")
        log_widget.see("end"); log_widget.configure(state="disabled")
    def refresh_columns():
        nonlocal tracer_names, target_names
        if paths["tracer"].get():
            names = list(smart_read_csv(paths["tracer"].get(), nrows=1).columns)
            if names != tracer_names:
                tracer_names = names; box.delete(0, "end")
                for name in names:
                    box.insert("end", name)
                box.selection_set(0, "end")
            for key, suggestions in (("tracer_z_col", ("zHD", "zcmb", "z", "redshift")),
                                     ("tracer_id_col", ("CID", "SNID", "Name", "name", "ID")),
                                     ("magnitude_col", ("m_b_corr",)), ("mu_col", ()), ("cov_index_col", ())):
                if key in widgets:
                    widgets[key]["values"] = ("",)+tuple(names)
                if columns[key].get() not in names:
                    columns[key].set(next((n for n in suggestions if n in names), ""))
        if paths["obj"].get():
            target_names = list(smart_read_csv(paths["obj"].get(), nrows=1).columns)
            for key, suggestions in (("z1_col", ("zl", "z_lens", "z", "redshift")), ("z2_col", ("zs", "z_source"))):
                widgets[key]["values"] = target_names
                if columns[key].get() not in target_names:
                    columns[key].set(next((n for n in suggestions if n in target_names), ""))
    def file_row(parent, label, key, directory=False, callback=False):
        row = ttk.Frame(parent); row.pack(fill="x", pady=2)
        label_widget = ttk.Label(row, text=label, width=24, anchor="w"); label_widget.pack(side="left")
        ttk.Entry(row, textvariable=paths[key]).pack(side="left", fill="x", expand=True, padx=5)
        def browse():
            value = filedialog.askdirectory() if directory else filedialog.askopenfilename()
            if value:
                paths[key].set(value)
                if callback:
                    try:
                        refresh_columns(); log(f"Loaded columns: {value}")
                    except Exception as exc:
                        messagebox.showerror("Input error", str(exc))
        ttk.Button(row, text="Browse", command=browse).pack(side="right")
        return label_widget
    file_row(f_files, "Photometric Data (SN):", "tracer", callback=True)
    target_label = file_row(f_files, "Lenses/Clusters:", "obj", callback=True)
    file_row(f_files, "Output Directory (optional):", "output_dir", directory=True)
    def refresh_mode():
        for key in ("z1_col", "z2_col", "tracer_z_col"):
            widgets[key+"_label"].pack_forget(); widgets[key].pack_forget()
        single = variables["mode"].get() == "cluster"
        target_label.configure(text="Cluster/Single Object Data:" if single else "Strong Lens Data:")
        widgets["z1_col_label"].configure(text="Cluster Redshift (z):" if single else "Lens Redshift (zl):")
        for key in (("z1_col", "tracer_z_col") if single else ("z1_col", "z2_col", "tracer_z_col")):
            widgets[key+"_label"].pack(side="left", padx=(2, 0)); widgets[key].pack(side="left", padx=(2, 8))
    def refresh_cov():
        for child in f_cov.winfo_children():
            child.destroy()
        if variables["sn_type"].get() != "jla":
            file_row(f_cov, "Full Covariance Matrix:", "covariance")
            ttk.Label(f_cov, text="Single column is supported: header N or N*N, followed by N*N entries.").pack(anchor="w", pady=3)
        else:
            grid = ttk.Frame(f_cov); grid.pack(fill="x")
            labels = {"v0": "Mag Cov (v0)", "va": "Stretch Cov (va)", "vb": "Colour Cov (vb)",
                      "v0a": "Mag-Stretch (v0a)", "v0b": "Mag-Colour (v0b)", "vab": "Stretch-Colour (vab)"}
            for i, key in enumerate(JLA_KEYS):
                r, col = i//2, (i%2)*3
                ttk.Label(grid, text=labels[key]).grid(row=r, column=col, sticky="e", padx=3, pady=3)
                ttk.Entry(grid, textvariable=paths[f"cov_jla_{key}"], width=28).grid(row=r, column=col+1, sticky="ew", padx=3)
                def browse_jla(k=key):
                    value = filedialog.askopenfilename()
                    if value:
                        paths[f"cov_jla_{k}"].set(value)
                ttk.Button(grid, text="...", width=3, command=browse_jla).grid(row=r, column=col+2, padx=3)
            grid.columnconfigure(1, weight=1); grid.columnconfigure(4, weight=1)
            values = ttk.Frame(f_cov); values.pack(fill="x", pady=5)
            for key in ("alpha", "beta"):
                ttk.Label(values, text=f"JLA {key}:").pack(side="left", padx=3)
                ttk.Entry(values, textvariable=variables[key], width=8).pack(side="left", padx=3)
            ttk.Label(values, text="(configured nuisance values)").pack(side="left", padx=5)
        ttk.Checkbutton(f_cov, text="Require covariance extraction (otherwise allow matching only)", variable=variables["require_covariance"]).pack(anchor="w", pady=3)
    for label, value in (("Strong Gravitational Lenses (2 SNe)", "sgls"), ("Galaxy Clusters (1 SN)", "cluster")):
        ttk.Radiobutton(f_mode, text=label, variable=variables["mode"], value=value, command=refresh_mode).pack(side="left", padx=5)
    for label, value in (("Pantheon+ (Single Matrix)", "pantheon"), ("JLA (6-Component Matrices)", "jla")):
        ttk.Radiobutton(f_sn, text=label, variable=variables["sn_type"], value=value, command=refresh_cov).pack(side="left", padx=5)
    def advanced():
        dialog = tk.Toplevel(root); dialog.title("Advanced Settings / 高级设置"); dialog.geometry("800x760")
        dialog.transient(root); dialog.grab_set()
        panel = ttk.Frame(dialog, padding=10); panel.pack(fill="both", expand=True); panel.columnconfigure(1, weight=1)
        rows = (("formulation", "SGL formulation", ("compact", "triplet")),
                ("unique_by", "SN uniqueness", ("auto", "object", "row")),
                ("tie_break", "Optimal tie rule", ("catalogue_order", "none")),
                ("tracer_id_col", "Physical SN ID", ("",)+tuple(tracer_names)),
                ("cov_index_col", "Explicit covariance row", ("",)+tuple(tracer_names)),
                ("magnitude_col", "Standardized magnitude / JLA mb", ("",)+tuple(tracer_names)),
                ("mu_col", "Calibrated distance modulus", ("",)+tuple(tracer_names)),
                ("absolute_magnitude", "Fixed M_B (optional)", None),
                ("time_limit", "Time limit per MILP stage (s)", None),
                ("jla_stat_mode", "JLA statistical term", ("auto", "none")),
                ("dd_tolerance", "2x2 dd tolerance", None), ("dz_tolerance", "2x2 dz tolerance", None),
                ("tolerances", "Sensitivity tolerances", None), ("omega_values", "Sensitivity Omega_m", None))
        for r, (key, label, choices) in enumerate(rows):
            ttk.Label(panel, text=label).grid(row=r, column=0, sticky="w", padx=4, pady=3)
            variable = columns[key] if key in columns else variables[key]
            widget = ttk.Combobox(panel, textvariable=variable, values=choices, state="readonly") if choices is not None else ttk.Entry(panel, textvariable=variable)
            widget.grid(row=r, column=1, sticky="ew", padx=4, pady=3)
        extras = ttk.LabelFrame(panel, text="Optional extra covariance inputs", padding=4)
        extras.grid(row=len(rows), column=0, columnspan=2, sticky="ew", pady=4)
        for key, label in (("cov_jla_stat", "JLA statistical matrix"), ("target_covariance", "Target covariance"), ("target_sn_cross", "True target/SN cross")):
            file_row(extras, label, key)
        controls = ttk.Frame(panel); controls.grid(row=len(rows)+1, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(controls, text="Export PNG/PDF plots", variable=variables["plots"]).pack(anchor="w")
        ttk.Checkbutton(controls, text="Explicitly drop invalid-redshift rows", variable=variables["drop_invalid"]).pack(anchor="w")
        ttk.Button(panel, text="Done / 完成", command=dialog.destroy).grid(row=len(rows)+2, column=0, columnspan=2, pady=8)
    def capture():
        cfg = copy.deepcopy(preserved.get("col_cfg", {}))
        for key in columns:
            cfg.pop(key, None)
            if columns[key].get():
                cfg[key] = columns[key].get()
        if variables["mode"].get() == "cluster":
            cfg.pop("z2_col", None)
        cfg["output_cols"] = [box.get(i) for i in box.curselection()]
        params = copy.deepcopy(preserved.get("params", {}))
        params.update(wm=float(variables["wm"].get()), tol=float(variables["tol"].get()),
                      formulation=variables["formulation"].get(), unique_by=variables["unique_by"].get(),
                      tie_break=variables["tie_break"].get(),
                      plots=variables["plots"].get(), require_covariance=variables["require_covariance"].get(),
                      invalid_rows="drop" if variables["drop_invalid"].get() else "error", jla_stat_mode=variables["jla_stat_mode"].get())
        params.pop("output_dir", None)
        if paths["output_dir"].get():
            params["output_dir"] = paths["output_dir"].get()
        for key in ("alpha", "beta", "time_limit", "absolute_magnitude"):
            params.pop(key, None)
            if variables[key].get():
                params[key] = float(variables[key].get())
        keys = ["tracer", "obj", "target_covariance", "target_sn_cross"]
        keys += [f"cov_jla_{k}" for k in JLA_KEYS]+["cov_jla_stat"] if variables["sn_type"].get() == "jla" else ["covariance"]
        fp = {k: paths[k].get() for k in keys if paths[k].get()}
        return {**{k: variables[k].get() for k in ("mode", "sn_type", "algo", "tol_type")}, "file_paths": fp, "col_cfg": cfg, "params": params}
    def apply_config(config):
        nonlocal preserved
        preserved = copy.deepcopy(config)
        for key in ("mode", "sn_type", "algo", "tol_type"):
            if key in config:
                variables[key].set(config[key])
        for key, value in config.get("file_paths", {}).items():
            key = "covariance" if key in {"cov_pantheon", "cov_des"} else key
            if key in paths:
                paths[key].set(value)
        for key, value in config.get("params", {}).items():
            if key in variables:
                variables[key].set(value)
            elif key == "output_dir":
                paths[key].set(value)
            elif key == "invalid_rows":
                variables["drop_invalid"].set(value == "drop")
        refresh_columns()
        for key, value in config.get("col_cfg", {}).items():
            if key in columns:
                columns[key].set(value or "")
        wanted = config.get("col_cfg", {}).get("output_cols")
        if wanted is not None:
            box.selection_clear(0, "end")
            for i, name in enumerate(tracer_names):
                if name in wanted:
                    box.selection_set(i)
        refresh_mode(); refresh_cov()
    def load_config():
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            try:
                apply_config(read_config(path))
            except Exception as exc:
                messagebox.showerror("Configuration error", str(exc))
    def save_config():
        try:
            config = capture()
            path = filedialog.asksaveasfilename(defaultextension=".json")
            if path:
                Path(path).write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Configuration error", str(exc))
    def start(kind="run"):
        try:
            refresh_columns(); config = capture()
            if not config["file_paths"].get("tracer") or not config["file_paths"].get("obj"):
                raise CosmoMatcherError("Please load photometric and target data files.")
            dd, dz = float(variables["dd_tolerance"].get()), float(variables["dz_tolerance"].get())
            tolerances = [float(v.strip()) for v in variables["tolerances"].get().split(",")]
            omegas = [float(v.strip()) for v in variables["omega_values"].get().split(",")]
        except Exception as exc:
            messagebox.showerror("Configuration error", str(exc)); return
        if state["job"] is not None or state["closing"]:
            return
        try:
            state["job"] = _start_gui_job(config, kind, dd, dz, tolerances, omegas)
        except Exception as exc:
            messagebox.showerror("Run failed", f"{type(exc).__name__}: {exc}"); return
        state["tail"] = []
        for b in buttons:
            b.configure(state="disabled")
        status.set("Running / 正在计算")
        log(f"Computing in process {state['job'].process.pid}")
    button = ttk.Button(f_run, text="Start Matching and Extract Covariances", command=start)
    button.pack(side="right"); buttons.append(button)
    for label, command in (("Advanced...", advanced), ("2x2 Ablation", lambda: start("ablation")),
                           ("Sensitivity", lambda: start("sensitivity")), ("Load config", load_config), ("Save config", save_config)):
        b = ttk.Button(footer, text=label, command=command); b.pack(side="right", padx=3); buttons.append(b)
    def destroy_gui():
        if state["after_id"] is not None:
            root.after_cancel(state["after_id"])
            state["after_id"] = None
        root.destroy()
    def close_gui():
        if state["closing"]:
            return
        state["closing"] = True
        state["close_started"] = time.monotonic()
        if state["job"] is None:
            destroy_gui()
        else:
            status.set("Stopping / 正在退出")
            if state["job"].process.poll() is None:
                state["job"].process.terminate()
    def poll():
        state["after_id"] = None
        job = state["job"]
        if job is not None:
            for line in _read_gui_job(job):
                log(line)
                state["tail"] = (state["tail"] + [line])[-12:]
            code = job.process.poll()
            if state["closing"] and code is None and time.monotonic()-state["close_started"] > 3:
                job.process.kill()
            if code is not None:
                # Drain bytes written between the first read and process exit.
                for line in _read_gui_job(job):
                    log(line)
                    state["tail"] = (state["tail"] + [line])[-12:]
                output = job.output_dir
                _cleanup_gui_job(job)
                state["job"] = None
                if not state["closing"]:
                    for b in buttons:
                        b.configure(state="normal")
                    status.set("Finished / 完成" if code == 0 else "Failed / 失败")
                    if code == 0:
                        log(f"Output: {output}")
                    else:
                        messagebox.showerror("Run failed", f"Process exited with code {code}\n" + "\n".join(state["tail"]))
        if state["closing"] and state["job"] is None:
            destroy_gui()
            return
        state["after_id"] = root.after(100, poll)
    refresh_mode(); refresh_cov()
    if initial_config:
        apply_config(initial_config)
    root.protocol("WM_DELETE_WINDOW", close_gui)
    state["after_id"] = root.after(100, poll)
    try:
        root.mainloop()
    finally:
        if state["job"] is not None:
            _cleanup_gui_job(state["job"])
            state["job"] = None
        try:
            if root.winfo_exists():
                destroy_gui()
        except tk.TclError:
            pass
        # Clear Tk variables while still on the creating thread; break callback cycles.
        paths.clear(); variables.clear(); columns.clear()
        gc.collect()



def main(argv=None):
    args_list = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    gui = sub.add_parser("gui", help="Open the graphical interface")
    gui.add_argument("--config")
    for command in ("run", "ablation", "sensitivity"):
        p = sub.add_parser(command)
        p.add_argument("--config", help="JSON config; relative paths resolve beside that JSON")
        for name in ("tracer", "targets", "covariance", "output", "tracer-z-col", "z-l-col", "z-s-col",
                     "tracer-id-col", "cov-index-col", "magnitude-col", "mu-col"):
            p.add_argument(f"--{name}")
        for name, choices in (("mode", ("sgls", "cluster")), ("sn-type", ("pantheon", "jla", "des", "generic")),
                              ("algorithm", ("milp", "greedy")), ("criterion", ("dd", "dz")),
                              ("formulation", ("triplet", "compact")), ("unique-by", ("auto", "row", "object")),
                              ("jla-stat-mode", ("auto", "none")), ("tie-break", ("catalogue_order", "none"))):
            p.add_argument(f"--{name}", choices=choices)
        for name in ("omega-m", "tolerance", "alpha", "beta", "time-limit", "absolute-magnitude"):
            p.add_argument(f"--{name}", type=float)
        p.add_argument("--no-plots", action="store_true")
        p.add_argument("--require-covariance", action="store_true")
        p.add_argument("--drop-invalid", action="store_true")
        if command == "ablation":
            p.add_argument("--dd-tolerance", type=float, default=.05)
            p.add_argument("--dz-tolerance", type=float, default=.005)
        if command == "sensitivity":
            p.add_argument("--tolerances", default="0.03,0.04,0.05,0.06,0.07")
            p.add_argument("--omega-values", default="0.28,0.3,0.32")
    args = parser.parse_args(args_list)
    try:
        if args.command in {None, "gui"}:
            launch_gui(read_config(args.config) if getattr(args, "config", None) else None)
            return 0
        config = read_config(args.config) if args.config else {}
        fp, cols, params = [config.setdefault(k, {}) for k in ("file_paths", "col_cfg", "params")]
        for src, dst in (("tracer", "tracer"), ("targets", "obj"), ("covariance", "covariance")):
            if getattr(args, src) is not None:
                fp[dst] = getattr(args, src)
        for src, dst in (("mode", "mode"), ("sn_type", "sn_type"), ("algorithm", "algo"), ("criterion", "tol_type")):
            if getattr(args, src) is not None:
                config[dst] = getattr(args, src)
        for key in ("tracer_z_col", "z_l_col", "z_s_col", "tracer_id_col", "cov_index_col", "magnitude_col", "mu_col"):
            if getattr(args, key) is not None:
                cols[{"z_l_col": "z1_col", "z_s_col": "z2_col"}.get(key, key)] = getattr(args, key)
        for src, dst in (("output", "output_dir"), ("omega_m", "wm"), ("tolerance", "tol"),
                         ("formulation", "formulation"), ("unique_by", "unique_by"), ("alpha", "alpha"),
                         ("beta", "beta"), ("time_limit", "time_limit"), ("absolute_magnitude", "absolute_magnitude"),
                         ("jla_stat_mode", "jla_stat_mode"), ("tie_break", "tie_break")):
            if getattr(args, src) is not None:
                params[dst] = getattr(args, src)
        if args.no_plots:
            params["plots"] = False
        if args.require_covariance:
            params["require_covariance"] = True
        if args.drop_invalid:
            params["invalid_rows"] = "drop"
        if args.command == "ablation":
            print(run_ablation(config, args.dd_tolerance, args.dz_tolerance).to_string(index=False))
        elif args.command == "sensitivity":
            print(run_sensitivity(config, [float(v) for v in args.tolerances.split(",")],
                                  [float(v) for v in args.omega_values.split(",")]).to_string(index=False))
        else:
            result = run_config(config)
            print(f"Report: {result.output_dir / 'run_summary.json'}")
        return 0
    except (CosmoMatcherError, OSError, ValueError, KeyError) as exc:
        print(f"CosmoMatcher error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
