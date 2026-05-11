"""
Marine Magnetic Levelling Tool
================================
Implements statistically rigorous crossover levelling for marine magnetic data.

Background
----------
Marine magnetic surveys collect Total Magnetic Intensity (TMI) along track lines.
Systematic errors (sensor drift, heading error, diurnal variations not fully removed,
cable layback inconsistencies) cause each line to have a different DC bias.
Levelling removes these biases by exploiting crossover points – locations where two
lines cross so both should record the same true field.

Methods implemented
-------------------
1. Crossover Least-Squares (standard geodetic/geophysical approach)
   - Detect all pairwise crossover points with spatial interpolation
   - Solve: min ||C(line_i) – C(line_j) + d_ij||²  where d_ij = measured crossover diff
   - Option: constant correction per line (standard), linear drift (extended)
   - Option: iterative robust re-weighting (downweight outlier crossovers)

2. Micro-levelling (along-track de-corrugation)
   - High-pass filter residual stripe noise after crossover levelling
   - Directional cosine filter in the frequency domain
   - Savitzky-Golay or Butterworth for smooth along-track trend removal

3. Combined: crossover → micro-levelling (recommended workflow)

Statistical quality metrics
---------------------------
- Mean, Std, RMS, MAD, P90 of crossover differences before and after
- Per-line correction magnitude
- Histogram, Q-Q plot, and convergence plot

References
----------
- Mauring, E. & Kihle, O. (2006) Levelling aerogeophysical data using a
  moving differential median filter.  Geophysics 71(1):L5-L11.
- Leaman, D.E. (1998) Criteria for the recognition of potential field levelling
  errors. Exploration Geophysics 29, 400-407.
- Urquhart, T. (1988) Decorrugation of enhanced magnetic field maps.  SEG Expanded Abstracts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.ticker import AutoMinorLocator
from scipy.spatial import cKDTree
from scipy.optimize import least_squares, lsq_linear
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.stats import probplot, median_abs_deviation, normaltest, shapiro
from math import radians, sin, cos, sqrt, atan2
import warnings
import time
import io

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marine Magnetic Levelling",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧲 Marine Magnetic Levelling Tool")
st.markdown(
    """
    Applies **crossover levelling** and **micro-levelling** to reduce systematic
    errors in marine Total Magnetic Intensity (TMI) data.  Upload the CSV output
    from your processing pipeline to begin.
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY — GEOGRAPHIC
# ─────────────────────────────────────────────────────────────────────────────
EARTH_RADIUS_M = 6_371_000.0
KM_PER_DEG = 111.0


def haversine(lon1, lat1, lon2, lat2):
    """Return distance in metres between two geographic points."""
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return EARTH_RADIUS_M * 2 * atan2(sqrt(a), sqrt(1 - a))


def deg_threshold(metres, mean_lat):
    """Convert a metre threshold to an approximate degree threshold."""
    lon_scale = KM_PER_DEG * abs(cos(radians(mean_lat)))
    lat_scale = KM_PER_DEG
    # conservative: use the smaller scale
    return metres / 1000.0 / min(lon_scale, lat_scale)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY — STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def crossover_stats(diffs):
    """Return a dict of common statistical descriptors for crossover differences."""
    if len(diffs) == 0:
        return {}
    d = np.asarray(diffs)
    return {
        "N": len(d),
        "Mean (nT)": float(np.mean(d)),
        "Std (nT)": float(np.std(d, ddof=1)),
        "RMS (nT)": float(np.sqrt(np.mean(d ** 2))),
        "MAD (nT)": float(median_abs_deviation(d)),
        "Min (nT)": float(np.min(d)),
        "Max (nT)": float(np.max(d)),
        "P10 (nT)": float(np.percentile(d, 10)),
        "P90 (nT)": float(np.percentile(d, 90)),
    }


def format_stats_table(stats_before, stats_after):
    """Return a pandas DataFrame comparing before/after statistics."""
    keys = list(stats_before.keys())
    rows = []
    for k in keys:
        b = stats_before.get(k, np.nan)
        a = stats_after.get(k, np.nan)
        if isinstance(b, float):
            rows.append({"Metric": k,
                         "Before Levelling": f"{b:.3f}",
                         "After Levelling": f"{a:.3f}",
                         "Improvement": f"{b - a:.3f}" if isinstance(b, float) and isinstance(a, float) else ""})
        else:
            rows.append({"Metric": k,
                         "Before Levelling": str(b),
                         "After Levelling": str(a),
                         "Improvement": ""})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CROSSOVER DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def segment_intersection_1d(p1, p2, q1, q2, tol=1e-12):
    """
    Find the parametric intersection of two line segments in 2-D.
    p(t) = p1 + t*(p2-p1),  q(s) = q1 + s*(q2-q1)
    Returns (t, s) with 0<=t<=1, 0<=s<=1, or None if no intersection.
    """
    dx1 = p2[0] - p1[0];  dy1 = p2[1] - p1[1]
    dx2 = q2[0] - q1[0];  dy2 = q2[1] - q1[1]
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < tol:
        return None  # parallel
    dpx = q1[0] - p1[0];  dpy = q1[1] - p1[1]
    t = (dpx * dy2 - dpy * dx2) / denom
    s = (dpx * dy1 - dpy * dx1) / denom
    if -tol <= t <= 1 + tol and -tol <= s <= 1 + tol:
        return np.clip(t, 0, 1), np.clip(s, 0, 1)
    return None


def interpolate_tmi_at_param(tmi_arr, param, seg_idx):
    """
    Linearly interpolate TMI at parametric position 'param' within segment
    [seg_idx, seg_idx+1].
    """
    return tmi_arr[seg_idx] + param * (tmi_arr[seg_idx + 1] - tmi_arr[seg_idx])


def find_crossovers_interpolated(df, line_col, dist_threshold_m=50.0,
                                  tmi_col='TMI', progress_callback=None):
    """
    Detect crossover points between distinct survey lines using exact segment
    intersection.  For each intersection the TMI is linearly interpolated on
    both lines, giving sub-point accuracy.

    Parameters
    ----------
    df : DataFrame with columns Longitude, Latitude, tmi_col, line_col
    dist_threshold_m : float – bounding-box pre-filter in metres
    tmi_col : which TMI column to use

    Returns
    -------
    list of dicts with keys:
        line1, line2, lon, lat,
        tmi1 (interpolated), tmi2 (interpolated),
        diff  (= tmi1 - tmi2),
        seg1_start_idx (global row index), seg2_start_idx
    """
    mean_lat = df['Latitude'].mean()
    thr_deg = deg_threshold(dist_threshold_m, mean_lat)

    # Build per-line arrays
    lines = [l for l in df[line_col].unique()
             if (df[line_col] == l).sum() >= 2]
    line_data = {}
    for ln in lines:
        sub = df[df[line_col] == ln].reset_index()
        line_data[ln] = {
            'lon':  sub['Longitude'].values,
            'lat':  sub['Latitude'].values,
            'tmi':  sub[tmi_col].values,
            'gidx': sub['index'].values,   # global df index
        }

    crossovers = []
    n_lines = len(lines)
    total_pairs = n_lines * (n_lines - 1) // 2
    pair_count = 0

    for i in range(n_lines):
        la = lines[i]
        lon1 = line_data[la]['lon'];  lat1 = line_data[la]['lat']
        tmi1 = line_data[la]['tmi']
        # bounding box of line a
        bb1 = (lon1.min(), lon1.max(), lat1.min(), lat1.max())

        for j in range(i + 1, n_lines):
            pair_count += 1
            if progress_callback and pair_count % max(1, total_pairs // 50) == 0:
                progress_callback(pair_count / total_pairs)

            lb = lines[j]
            lon2 = line_data[lb]['lon'];  lat2 = line_data[lb]['lat']
            tmi2 = line_data[lb]['tmi']
            bb2 = (lon2.min(), lon2.max(), lat2.min(), lat2.max())

            # Bounding box pre-filter (with threshold padding)
            if (bb1[0] - thr_deg > bb2[1] + thr_deg or
                    bb1[1] + thr_deg < bb2[0] - thr_deg or
                    bb1[2] - thr_deg > bb2[3] + thr_deg or
                    bb1[3] + thr_deg < bb2[2] - thr_deg):
                continue

            # Segment-level intersection test
            for s1 in range(len(lon1) - 1):
                p1 = np.array([lon1[s1], lat1[s1]])
                p2 = np.array([lon1[s1 + 1], lat1[s1 + 1]])
                for s2 in range(len(lon2) - 1):
                    q1 = np.array([lon2[s2], lat2[s2]])
                    q2 = np.array([lon2[s2 + 1], lat2[s2 + 1]])
                    res = segment_intersection_1d(p1, p2, q1, q2)
                    if res is None:
                        continue
                    t, s = res
                    xi = p1[0] + t * (p2[0] - p1[0])
                    yi = p1[1] + t * (p2[1] - p1[1])
                    v1 = interpolate_tmi_at_param(tmi1, t, s1)
                    v2 = interpolate_tmi_at_param(tmi2, s, s2)
                    crossovers.append({
                        'line1': la, 'line2': lb,
                        'lon': xi,  'lat': yi,
                        'tmi1': v1, 'tmi2': v2,
                        'diff': v1 - v2,
                        'seg1_start_idx': line_data[la]['gidx'][s1],
                        'seg2_start_idx': line_data[lb]['gidx'][s2],
                    })
    return crossovers


# ─────────────────────────────────────────────────────────────────────────────
# CROSSOVER LEAST-SQUARES LEVELLING
# ─────────────────────────────────────────────────────────────────────────────
def build_lsq_system(crossovers, unique_lines, robust_weights=None):
    """
    Build the design matrix A and observation vector b for:
        A @ corrections = b
    Each row enforces:  c[line1] - c[line2] = -diff_ij

    Parameters
    ----------
    crossovers : list of dicts from find_crossovers_interpolated
    unique_lines : list – all line names (defines correction indices)
    robust_weights : array of shape (N_crossovers,) or None

    Returns
    -------
    A, b, W (weight matrix diagonal)
    """
    n = len(unique_lines)
    line_idx = {ln: i for i, ln in enumerate(unique_lines)}
    N = len(crossovers)

    A = np.zeros((N, n))
    b = np.zeros(N)
    W = np.ones(N) if robust_weights is None else robust_weights.copy()

    for k, co in enumerate(crossovers):
        i1 = line_idx[co['line1']]
        i2 = line_idx[co['line2']]
        A[k, i1] = 1.0
        A[k, i2] = -1.0
        b[k] = -co['diff']

    return A, b, W


def solve_crossover_corrections(crossovers, unique_lines, damping=1e-4,
                                 robust=False, n_iter=5, k_mad=2.5):
    """
    Solve for per-line additive corrections using weighted least squares.

    The system  A c = b  is under-determined (zero-mean ambiguity).
    We regularise with Tikhonov damping (mean-zero constraint).

    Optionally performs IRLS (Iteratively Re-weighted Least Squares) for
    robustness against outlier crossovers (e.g. from acquisition errors).

    Returns
    -------
    corrections : dict {line_name: correction_nT}
    residuals : array of per-crossover residuals after levelling
    weights_final : array of final IRLS weights
    rms_iter : list of per-iteration RMS (for convergence plot)
    """
    n = len(unique_lines)
    W = np.ones(len(crossovers))
    corrections = np.zeros(n)
    rms_iter = []

    for iteration in range(n_iter if robust else 1):
        A, b, Wk = build_lsq_system(crossovers, unique_lines, W)
        # Weighted system: sqrt(W) * A @ c = sqrt(W) * b
        sW = np.sqrt(Wk)
        Aw = A * sW[:, None]
        bw = b * sW

        # Tikhonov regularisation to fix zero-mean
        A_reg = np.vstack([Aw, np.eye(n) * damping])
        b_reg = np.concatenate([bw, np.zeros(n)])

        result = lsq_linear(A_reg, b_reg)
        corrections = result.x

        residuals = A @ corrections - b
        rms = np.sqrt(np.mean(residuals ** 2))
        rms_iter.append(rms)

        if robust:
            mad = median_abs_deviation(residuals, scale='normal')
            if mad < 1e-10:
                break
            W = 1.0 / np.clip(np.abs(residuals) / (k_mad * mad), 1.0, None)

    corr_dict = {ln: corrections[i] for i, ln in enumerate(unique_lines)}
    return corr_dict, residuals, W, rms_iter


def apply_corrections(df, corr_dict, line_col, tmi_col='TMI',
                       out_col='TMI_leveled'):
    """Apply per-line additive corrections and return a new DataFrame."""
    df = df.copy()
    df[out_col] = df[tmi_col].copy()
    for line, corr in corr_dict.items():
        mask = df[line_col] == line
        df.loc[mask, out_col] += corr
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MICRO-LEVELLING (ALONG-TRACK DE-CORRUGATION)
# ─────────────────────────────────────────────────────────────────────────────
def microlevelling_butterworth(df, line_col, tmi_col, cutoff_fraction=0.05,
                                order=4):
    """
    Remove long-wavelength inter-line striping by high-pass Butterworth filter
    applied independently along each survey line.

    The idea (Mauring & Kihle 2006):
        corrected = TMI - low_pass(TMI)  →  residuals only
    But we want to keep the signal, so instead we compute a *correction* that
    minimises the long-wavelength differences between adjacent lines.

    Implementation: for each line, fit and subtract a polynomial (degree 2)
    + Butterworth low-pass, then add back the survey-wide mean.

    Returns a DataFrame with the micro-levelled column added.
    """
    df = df.copy()
    out = df[tmi_col].copy()

    for line in df[line_col].unique():
        mask = df[line_col] == line
        sub = df[mask].copy().sort_values('datetime')
        idx = sub.index
        vals = sub[tmi_col].values

        if len(vals) < 20:
            continue

        # Design a zero-phase Butterworth low-pass
        b_filt, a_filt = butter(order, cutoff_fraction, btype='low',
                                 analog=False)
        try:
            trend = filtfilt(b_filt, a_filt, vals)
        except Exception:
            trend = np.full_like(vals, vals.mean())

        # Correction is the difference between line mean and local trend
        line_mean = vals.mean()
        correction = trend - line_mean
        out.loc[idx] = vals - correction

    df['TMI_microleveled'] = out.values
    return df


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    'before': '#E74C3C',
    'after':  '#2ECC71',
    'neutral': '#3498DB',
    'bg': '#F8F9FA',
}


def fig_crossover_histogram(diffs_before, diffs_after, bins=60):
    """Side-by-side histograms of crossover differences before/after."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=PALETTE['bg'])
    for ax, diffs, label, color in [
        (axes[0], diffs_before, 'Before Levelling', PALETTE['before']),
        (axes[1], diffs_after,  'After Levelling',  PALETTE['after']),
    ]:
        ax.hist(diffs, bins=bins, color=color, alpha=0.85, edgecolor='white',
                linewidth=0.3)
        mu, sigma = np.mean(diffs), np.std(diffs)
        rms = np.sqrt(np.mean(np.array(diffs) ** 2))
        ax.axvline(0, color='k', lw=1.2, ls='--')
        ax.axvline(mu, color='navy', lw=1.5, ls='-', label=f'Mean={mu:.2f}')
        ax.set_title(f"{label}\nRMS={rms:.2f} nT | σ={sigma:.2f} nT | N={len(diffs)}",
                     fontsize=11)
        ax.set_xlabel('Crossover Difference (nT)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(fontsize=9)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_facecolor(PALETTE['bg'])
        ax.grid(True, alpha=0.3, which='both')
    fig.suptitle('Crossover Difference Distributions', fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def fig_qqplot(diffs_before, diffs_after):
    """Q-Q plots to assess normality of crossover differences."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=PALETTE['bg'])
    for ax, diffs, label, color in [
        (axes[0], diffs_before, 'Before', PALETTE['before']),
        (axes[1], diffs_after,  'After',  PALETTE['after']),
    ]:
        (osm, osr), (slope, intercept, r) = probplot(diffs, dist="norm")
        ax.scatter(osm, osr, s=12, alpha=0.6, color=color)
        xlim = np.array([min(osm), max(osm)])
        ax.plot(xlim, slope * xlim + intercept, 'k--', lw=1.5, label=f'R²={r**2:.4f}')
        ax.set_title(f'Q-Q Plot ({label} Levelling)', fontsize=11)
        ax.set_xlabel('Theoretical Quantiles', fontsize=9)
        ax.set_ylabel('Sample Quantiles (nT)', fontsize=9)
        ax.legend(fontsize=9)
        ax.set_facecolor(PALETTE['bg'])
        ax.grid(True, alpha=0.3)
    fig.suptitle('Normality Assessment of Crossover Differences', fontsize=12,
                 fontweight='bold')
    fig.tight_layout()
    return fig


def fig_convergence(rms_iter):
    """IRLS convergence plot."""
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=PALETTE['bg'])
    ax.plot(range(1, len(rms_iter) + 1), rms_iter, 'o-',
            color=PALETTE['neutral'], lw=2, ms=7)
    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel('RMS Crossover Residual (nT)', fontsize=10)
    ax.set_title('IRLS Convergence', fontsize=11, fontweight='bold')
    ax.set_facecolor(PALETTE['bg'])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_correction_map(df, corr_dict, line_col):
    """Map showing the spatial pattern of corrections."""
    rep = (df.groupby(line_col)
             .agg(lon=('Longitude', 'median'), lat=('Latitude', 'median'))
             .reset_index())
    rep['corr'] = rep[line_col].map(corr_dict).fillna(0.0)

    vmax = np.abs(rep['corr']).quantile(0.98)
    if vmax < 1e-6:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=PALETTE['bg'])
    sc = ax.scatter(rep['lon'], rep['lat'],
                    c=rep['corr'], cmap='RdBu_r', norm=norm,
                    s=60, edgecolor='k', linewidths=0.4, zorder=5)
    plt.colorbar(sc, ax=ax, label='Additive Correction (nT)', shrink=0.8)
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title('Spatial Distribution of Per-Line Corrections', fontsize=12,
                 fontweight='bold')
    ax.set_facecolor('#EEF2F6')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_crossover_map(crossovers_before, crossovers_after):
    """Map coloured by crossover difference magnitude, before and after."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=PALETTE['bg'])
    all_diffs = [abs(c['diff']) for c in crossovers_before + crossovers_after]
    vmax = np.percentile(all_diffs, 98) if all_diffs else 1.0
    norm = Normalize(vmin=0, vmax=vmax)

    for ax, cos, label in [
        (axes[0], crossovers_before, 'Before Levelling'),
        (axes[1], crossovers_after,  'After Levelling'),
    ]:
        lons = [c['lon'] for c in cos]
        lats = [c['lat'] for c in cos]
        diffs = [abs(c['diff']) for c in cos]
        sc = ax.scatter(lons, lats, c=diffs, cmap='hot_r', norm=norm,
                        s=25, edgecolor='none', alpha=0.85)
        plt.colorbar(sc, ax=ax, label='|Diff| (nT)', shrink=0.85)
        ax.set_title(f'Crossover Errors – {label}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=9)
        ax.set_ylabel('Latitude', fontsize=9)
        ax.set_facecolor('#EEF2F6')
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_profile_comparison(df, line_name, line_col, tmi_cols, labels, colors):
    """TMI profile comparison for a single survey line."""
    sub = df[df[line_col] == line_name].sort_values('datetime').reset_index(drop=True)
    # Build cumulative along-track distance in km
    dist = [0.0]
    for k in range(1, len(sub)):
        d = haversine(sub.loc[k - 1, 'Longitude'], sub.loc[k - 1, 'Latitude'],
                      sub.loc[k, 'Longitude'], sub.loc[k, 'Latitude'])
        dist.append(dist[-1] + d / 1000.0)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), facecolor=PALETTE['bg'],
                              sharex=True)
    # Top: absolute values
    ax = axes[0]
    for col, lbl, clr in zip(tmi_cols, labels, colors):
        if col in sub.columns:
            ax.plot(dist, sub[col], lw=1.3, color=clr, alpha=0.8, label=lbl)
    ax.set_ylabel('TMI (nT)', fontsize=10)
    ax.set_title(f'Profile: {line_name}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_facecolor(PALETTE['bg'])

    # Bottom: difference (correction applied)
    ax2 = axes[1]
    if tmi_cols[0] in sub.columns and tmi_cols[-1] in sub.columns:
        ax2.plot(dist, sub[tmi_cols[-1]] - sub[tmi_cols[0]],
                 lw=1.2, color='#8E44AD', alpha=0.85)
        ax2.axhline(0, color='k', lw=0.8, ls='--')
        ax2.set_ylabel('Correction (nT)', fontsize=10)
        ax2.set_xlabel('Along-track distance (km)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor(PALETTE['bg'])
    fig.tight_layout()
    return fig


def fig_correction_histogram(corr_dict):
    """Histogram of per-line corrections."""
    vals = list(corr_dict.values())
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=PALETTE['bg'])
    ax.hist(vals, bins=30, color=PALETTE['neutral'], edgecolor='white',
            linewidth=0.4, alpha=0.9)
    ax.axvline(0, color='k', lw=1, ls='--')
    ax.set_xlabel('Additive Correction (nT)', fontsize=10)
    ax.set_ylabel('Number of Lines', fontsize=10)
    ax.set_title(f'Distribution of Per-Line Corrections  (N lines = {len(vals)})',
                 fontsize=11, fontweight='bold')
    ax.set_facecolor(PALETTE['bg'])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_scatter_before_after(diffs_before, diffs_after):
    """Scatter plot: crossover diff before vs after (should cluster near zero)."""
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE['bg'])
    ax.scatter(diffs_before, diffs_after, alpha=0.4, s=12,
               color=PALETTE['neutral'], edgecolor='none')
    lims = [min(min(diffs_before), min(diffs_after)),
            max(max(diffs_before), max(diffs_after))]
    ax.plot(lims, lims, 'k--', lw=1, label='1:1')
    ax.axhline(0, color='green', lw=1, ls='-')
    ax.axvline(0, color='red', lw=1, ls='-')
    ax.set_xlabel('Crossover Diff Before (nT)', fontsize=10)
    ax.set_ylabel('Crossover Diff After (nT)', fontsize=10)
    ax.set_title('Before vs After Crossover Differences', fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_facecolor(PALETTE['bg'])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.sidebar.file_uploader("📂 Upload processed CSV", type=['csv'])

if uploaded_file is None:
    st.info("⬅️  Upload a CSV file (output from your processing pipeline) to begin.")

    # Explanation section
    with st.expander("📖 How does levelling work? (theory & best practice)"):
        st.markdown("""
### The Problem: Systematic Inter-Line Errors

Even after standard corrections (IGRF, diurnal, sensor lag, etc.) each survey
line may carry a **residual DC bias** caused by:
- Incomplete diurnal correction (observatory too far away, interpolation error)
- Sensor drift between calibrations  
- Heading error / magnetisation of vessel hull  
- Cable layback variation  

These biases appear as **stripes** parallel to the survey lines on gridded data.

---

### The Solution: Crossover Levelling

A **crossover point** is where two survey lines physically cross.  At a
crossover the true geomagnetic field is the same for both lines, so any
difference in measured TMI is *entirely due to systematic error*.

We want to find one additive correction per line, **c_i**, such that:

    c_i  –  c_j  ≈  d_ij   for all crossover pairs (i, j)

where d_ij = TMI_i(crossing) – TMI_j(crossing).

This is an **over-determined linear system** solved by weighted least squares.
One degree of freedom is removed by Tikhonov regularisation (zero-mean
constraint), so the solution preserves the original datum.

**Robust variant (IRLS):** crossovers caused by acquisition glitches are
down-weighted automatically using a Huber-like weight based on the residual
relative to MAD.

---

### Survey Design Requirement

Effective levelling requires **tie lines** – lines flown/sailed roughly
perpendicular to the main survey lines.  A good rule of thumb:

- Tie line spacing ≤ 5× main-line spacing  
- At least ~3–5 crossovers per main line  
- Tie lines should cover the full survey extent  

If your survey has **no tie lines**, crossover levelling is impossible; use
micro-levelling (along-track de-corrugation) as a partial substitute.

---

### Quality Metrics

| Metric | Ideal value |
|--------|-------------|
| RMS crossover difference after levelling | < 1–2 nT for modern surveys |
| Mean crossover difference | ≈ 0 nT |
| Reduction in RMS | > 60% |
| Normality of residuals (Shapiro-Wilk p > 0.05) | Desirable |

---

### References
- Mauring, E. & Kihle, O. (2006) *Geophysics* **71**(1):L5-L11.  
- Leaman, D.E. (1998) *Exploration Geophysics* **29**, 400-407.  
- Urquhart, T. (1988) SEG Expanded Abstracts.  
        """)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    return df

with st.spinner("Loading data …"):
    df = load_data(uploaded_file)

st.subheader("📋 Input data preview")
st.dataframe(df.head(10), use_container_width=True)

# ── Column check ────────────────────────────────────────────────────────────
REQUIRED = ['Latitude', 'Longitude', 'TMI', 'datetime', 'Sheet_Name', 'Line_Name']
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df['Full_Line'] = df['Sheet_Name'].astype(str) + '_' + df['Line_Name'].astype(str)

n_lines = df['Full_Line'].nunique()
n_points = len(df)
col1, col2, col3 = st.columns(3)
col1.metric("Survey lines", n_lines)
col2.metric("Data points", f"{n_points:,}")
col3.metric("TMI range (nT)", f"{df['TMI'].min():.1f}  →  {df['TMI'].max():.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️  Levelling Configuration")

method = st.sidebar.selectbox(
    "Levelling method",
    ["Crossover Least-Squares (standard)",
     "Crossover LS + Micro-levelling (recommended)",
     "Micro-levelling only (no tie lines)"],
    index=0,
)

st.sidebar.subheader("Crossover Detection")
dist_thresh = st.sidebar.slider(
    "Intersection search threshold (m)", 5, 200, 30, 5,
    help="Only used for KD-tree pre-filtering; the actual detection uses exact "
         "geometric segment intersection (no threshold bias).")

st.sidebar.subheader("Least-Squares Solver")
robust_ls = st.sidebar.checkbox(
    "Robust IRLS (downweight outlier crossovers)", value=True,
    help="Iteratively Re-weighted Least Squares.  Recommended when data quality "
         "is uneven or crossover errors are non-Gaussian.")
n_iter_irls = st.sidebar.slider("IRLS max iterations", 2, 20, 8, 1) if robust_ls else 1
k_mad_irls  = st.sidebar.slider("IRLS MAD multiplier (k)", 1.0, 5.0, 2.5, 0.5) if robust_ls else 2.5
damping_val = st.sidebar.number_input("Tikhonov regularisation (λ)", value=1e-5,
                                       format="%.1e",
                                       help="Small positive number; prevents "
                                            "over-fitting to sparse crossovers.")

if "only" not in method:
    st.sidebar.subheader("Micro-levelling")
    ml_cutoff = st.sidebar.slider(
        "Butterworth low-pass cutoff (fraction of Nyquist)", 0.01, 0.3, 0.05, 0.01,
        help="Controls which along-track wavelengths are treated as 'long-wavelength "
             "noise'.  0.05 = remove features longer than ~20 point spacings.")

run_button = st.sidebar.button("▶  Run Levelling", type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# RUN LEVELLING
# ─────────────────────────────────────────────────────────────────────────────
if run_button:
    st.header("⚙️  Processing")
    progress_bar = st.progress(0)
    status = st.empty()

    # ── Step 1: Crossover detection ──────────────────────────────────────────
    do_crossover = "Micro-levelling only" not in method

    if do_crossover:
        status.info("🔍 Step 1 / 3 — Detecting crossover points …")
        t0 = time.time()

        def update_progress(p):
            progress_bar.progress(min(int(p * 40), 40))

        crossovers_raw = find_crossovers_interpolated(
            df, 'Full_Line', dist_threshold_m=dist_thresh,
            tmi_col='TMI', progress_callback=update_progress)

        elapsed = time.time() - t0
        n_co = len(crossovers_raw)

        if n_co == 0:
            st.error(
                "❌ No crossover points were detected.  "
                "Check that your survey has tie lines perpendicular to the main "
                "lines and that the intersection threshold is appropriate.  "
                "If there are no tie lines, use 'Micro-levelling only'.")
            st.stop()

        st.success(f"✅ Found **{n_co}** crossover points in {elapsed:.1f} s")
        progress_bar.progress(40)

        # ── Step 2: Solve LS ──────────────────────────────────────────────────
        status.info("🧮 Step 2 / 3 — Solving for per-line corrections …")
        t0 = time.time()
        unique_lines = sorted(df['Full_Line'].unique())

        corr_dict, residuals, weights_final, rms_iter = solve_crossover_corrections(
            crossovers_raw, unique_lines,
            damping=damping_val,
            robust=robust_ls,
            n_iter=n_iter_irls,
            k_mad=k_mad_irls,
        )
        elapsed = time.time() - t0
        n_corr_nonzero = sum(1 for v in corr_dict.values() if abs(v) > 1e-6)
        status.info(f"✅ Corrections solved in {elapsed:.1f} s — "
                    f"{n_corr_nonzero}/{len(corr_dict)} lines received non-trivial corrections")
        progress_bar.progress(65)

        # Apply corrections
        df = apply_corrections(df, corr_dict, 'Full_Line', 'TMI', 'TMI_leveled')

    else:
        # No crossover method – start from raw TMI
        df['TMI_leveled'] = df['TMI'].copy()
        corr_dict = {}
        rms_iter = []
        crossovers_raw = []

    # ── Step 3: Micro-levelling (optional) ───────────────────────────────────
    if "Micro-levelling" in method:
        status.info("🌊 Step 3 / 3 — Applying micro-levelling (Butterworth de-corrugation) …")
        t0 = time.time()
        df = microlevelling_butterworth(
            df, 'Full_Line',
            tmi_col='TMI_leveled' if do_crossover else 'TMI',
            cutoff_fraction=ml_cutoff)
        # Rename for consistent downstream reference
        df['TMI_final'] = df['TMI_microleveled']
        elapsed = time.time() - t0
        status.info(f"✅ Micro-levelling done in {elapsed:.1f} s")
    else:
        df['TMI_final'] = df['TMI_leveled']

    progress_bar.progress(90)

    # ── Compute crossover differences before/after ────────────────────────────
    if do_crossover and n_co > 0:
        diffs_before = [c['diff'] for c in crossovers_raw]
        # Re-compute crossovers on the final levelled TMI
        crossovers_after = find_crossovers_interpolated(
            df, 'Full_Line', dist_threshold_m=dist_thresh, tmi_col='TMI_final')
        diffs_after = [c['diff'] for c in crossovers_after]
        stats_b = crossover_stats(diffs_before)
        stats_a = crossover_stats(diffs_after)
    else:
        diffs_before = diffs_after = []
        crossovers_after = []
        stats_b = stats_a = {}

    progress_bar.progress(100)
    status.success("🎉 Levelling complete!")

    # Store results in session state
    st.session_state['df_result']         = df
    st.session_state['corr_dict']         = corr_dict
    st.session_state['rms_iter']          = rms_iter
    st.session_state['diffs_before']      = diffs_before
    st.session_state['diffs_after']       = diffs_after
    st.session_state['crossovers_raw']    = crossovers_raw
    st.session_state['crossovers_after']  = crossovers_after
    st.session_state['stats_b']           = stats_b
    st.session_state['stats_a']           = stats_a
    st.session_state['method_used']       = method

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS — shown whenever session state is populated
# ─────────────────────────────────────────────────────────────────────────────
if 'df_result' not in st.session_state:
    st.stop()

df_out          = st.session_state['df_result']
corr_dict       = st.session_state['corr_dict']
rms_iter        = st.session_state['rms_iter']
diffs_before    = st.session_state['diffs_before']
diffs_after     = st.session_state['diffs_after']
crossovers_raw  = st.session_state['crossovers_raw']
crossovers_after= st.session_state['crossovers_after']
stats_b         = st.session_state['stats_b']
stats_a         = st.session_state['stats_a']
method_used     = st.session_state['method_used']

st.header("📊 Quality Assessment")

# ── Summary statistics table ─────────────────────────────────────────────────
if stats_b:
    st.subheader("Crossover Difference Statistics")
    st.dataframe(
        format_stats_table(stats_b, stats_a).style.highlight_min(
            subset=['After Levelling'], color='#d4edda'),
        use_container_width=True)

    # Headline KPI metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMS before (nT)",  f"{stats_b.get('RMS (nT)', 0):.3f}")
    m2.metric("RMS after (nT)",   f"{stats_a.get('RMS (nT)', 0):.3f}",
              delta=f"−{stats_b.get('RMS (nT)', 0) - stats_a.get('RMS (nT)', 0):.3f} nT",
              delta_color="inverse")
    rms_imp = (1 - stats_a.get('RMS (nT)', 1) / max(stats_b.get('RMS (nT)', 1), 1e-9)) * 100
    m3.metric("RMS improvement",  f"{rms_imp:.1f} %")
    m4.metric("Crossovers used",  f"{stats_b.get('N', 0)}")

    # Normality test on residuals
    if len(diffs_after) >= 8:
        if len(diffs_after) <= 5000:
            stat_sw, p_sw = shapiro(diffs_after[:5000])
            st.info(f"**Shapiro-Wilk normality test on post-levelling residuals:**  "
                    f"W = {stat_sw:.4f},  p = {p_sw:.4g}  "
                    f"({'✅ Normal' if p_sw > 0.05 else '⚠️ Non-normal — consider IRLS'})")
        else:
            stat_n, p_n = normaltest(diffs_after)
            st.info(f"**D'Agostino-Pearson normality test:**  "
                    f"stat = {stat_n:.4f},  p = {p_n:.4g}  "
                    f"({'✅ Normal' if p_n > 0.05 else '⚠️ Non-normal'})")

# ── Tabs for plots ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Histograms",
    "📈 Q-Q Plots",
    "🗺️ Crossover Map",
    "📍 Correction Map",
    "📉 Profile Viewer",
    "🔄 Convergence",
])

with tab1:
    if diffs_before:
        st.pyplot(fig_crossover_histogram(diffs_before, diffs_after))
        st.pyplot(fig_scatter_before_after(diffs_before, diffs_after))
    else:
        st.info("No crossover data — histograms not available for micro-levelling only.")

with tab2:
    if diffs_before:
        st.pyplot(fig_qqplot(diffs_before, diffs_after))
    else:
        st.info("No crossover data available.")

with tab3:
    if crossovers_raw:
        st.pyplot(fig_crossover_map(crossovers_raw, crossovers_after))
    else:
        st.info("No crossover data available.")

with tab4:
    if corr_dict:
        st.pyplot(fig_correction_map(df_out, corr_dict, 'Full_Line'))
        st.pyplot(fig_correction_histogram(corr_dict))
        # Per-line corrections table
        corr_df = pd.DataFrame(
            [{'Line': k, 'Correction (nT)': round(v, 4)}
             for k, v in sorted(corr_dict.items(), key=lambda x: abs(x[1]),
                                 reverse=True)])
        st.subheader("Per-line corrections (sorted by magnitude)")
        st.dataframe(corr_df, use_container_width=True, height=300)
    else:
        st.info("No per-line corrections (micro-levelling only).")

with tab5:
    lines_available = sorted(df_out['Full_Line'].unique())
    sel_line = st.selectbox("Select a survey line", lines_available, key='profile_sel')
    tmi_cols  = ['TMI']
    col_labels = ['Original TMI']
    col_colors = ['#2980B9']
    if 'TMI_leveled' in df_out.columns:
        tmi_cols.append('TMI_leveled'); col_labels.append('After Crossover LS'); col_colors.append('#E74C3C')
    if 'TMI_microleveled' in df_out.columns:
        tmi_cols.append('TMI_microleveled'); col_labels.append('After Micro-levelling'); col_colors.append('#27AE60')

    st.pyplot(fig_profile_comparison(df_out, sel_line, 'Full_Line',
                                      tmi_cols, col_labels, col_colors))

with tab6:
    if rms_iter:
        st.pyplot(fig_convergence(rms_iter))
        st.dataframe(pd.DataFrame({'Iteration': range(1, len(rms_iter)+1),
                                   'RMS (nT)': rms_iter}),
                     use_container_width=True)
    else:
        st.info("Convergence data only available for IRLS crossover levelling.")

# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
st.header("💾 Download Results")

# Main levelled data
output_cols = [c for c in df_out.columns if c != 'Full_Line']
csv_data = df_out[output_cols].to_csv(index=False).encode('utf-8')
st.download_button("📥 Download levelled TMI (CSV)", csv_data,
                   "levelled_magnetic_data.csv", "text/csv")

# Crossover report
if crossovers_raw:
    co_df = pd.DataFrame(crossovers_raw)[
        ['line1', 'line2', 'lon', 'lat', 'tmi1', 'tmi2', 'diff']]
    co_after_df = pd.DataFrame(crossovers_after)[
        ['line1', 'line2', 'lon', 'lat', 'tmi1', 'tmi2', 'diff']].rename(
        columns={'tmi1': 'tmi1_leveled', 'tmi2': 'tmi2_leveled', 'diff': 'diff_leveled'})
    co_report = co_df.join(co_after_df[['diff_leveled']])
    csv_co = co_report.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download crossover report (CSV)", csv_co,
                       "crossover_report.csv", "text/csv")

# Corrections table
if corr_dict:
    csv_corr = pd.DataFrame(
        [{'Line': k, 'Correction_nT': v} for k, v in corr_dict.items()]
    ).to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download per-line corrections (CSV)", csv_corr,
                       "line_corrections.csv", "text/csv")
