import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.stats import median_abs_deviation
from math import radians, sin, cos, sqrt, atan2
import time

st.set_page_config(page_title="Marine Magnetic Levelling", layout="wide")
st.title("🧲 Marine Magnetic Levelling Tool")
st.markdown("Upload your processed CSV (output from previous processing) to apply levelling and reduce crossover differences.")

# ------------------------------------------------------------------
# Helper functions for levelling (crossover detection & adjustment)
# ------------------------------------------------------------------
def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371000
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def find_crossovers(df, line_col, dist_threshold_m=20.0):
    """
    Find crossover points between distinct lines.
    Returns a list of (line1, line2, idx1, idx2, diff_TMI, dist_m)
    """
    # Build a dict of line -> (coords, indices, values)
    lines = df[line_col].unique()
    line_info = {}
    for line in lines:
        mask = df[line_col] == line
        if mask.sum() < 2:
            continue
        coords = np.column_stack((df.loc[mask, 'Longitude'].values,
                                  df.loc[mask, 'Latitude'].values))
        indices = np.where(mask)[0]
        tmi = df.loc[mask, 'TMI'].values
        line_info[line] = (coords, indices, tmi)
    
    # Convert threshold to degrees (approx)
    km_per_deg = 111.0
    thr_deg = dist_threshold_m / 1000.0 / km_per_deg
    
    crossovers = []
    line_names = list(line_info.keys())
    for i in range(len(line_names)):
        line1 = line_names[i]
        coords1, idx1_global, vals1 = line_info[line1]
        tree = cKDTree(coords1)
        for j in range(i+1, len(line_names)):
            line2 = line_names[j]
            coords2, idx2_global, vals2 = line_info[line2]
            # For each point in line2, find nearest in line1
            dist, idx1_local = tree.query(coords2, distance_upper_bound=thr_deg)
            for k, (d, i1) in enumerate(zip(dist, idx1_local)):
                if d < thr_deg:
                    # crossover found
                    i1_global = idx1_global[i1]
                    i2_global = idx2_global[k]
                    diff = vals1[i1] - vals2[k]
                    # real distance in meters
                    d_m = haversine_distance(coords2[k,0], coords2[k,1],
                                             coords1[i1,0], coords1[i1,1])
                    crossovers.append((line1, line2, i1_global, i2_global, diff, d_m))
    return crossovers

def crossover_levelling(df, line_col, dist_threshold_m=20.0, fix_line=None):
    """
    Perform crossover adjustment: solve for additive corrections per line.
    Returns corrected TMI array and correction per line (dict).
    """
    crossovers = find_crossovers(df, line_col, dist_threshold_m)
    if len(crossovers) == 0:
        st.warning("No crossovers found. No levelling applied.")
        return df['TMI'].values.copy(), {}
    
    # Build system of equations: correction(line1) - correction(line2) = -diff
    unique_lines = sorted(set([c[0] for c in crossovers] + [c[1] for c in crossovers]))
    line_to_idx = {line: i for i, line in enumerate(unique_lines)}
    n_lines = len(unique_lines)
    A = []
    b = []
    for line1, line2, _, _, diff, _ in crossovers:
        row = np.zeros(n_lines)
        row[line_to_idx[line1]] = 1.0
        row[line_to_idx[line2]] = -1.0
        A.append(row)
        b.append(-diff)
    A = np.array(A)
    b = np.array(b)
    
    # Add regularization to fix the zero‑mean ambiguity (small damping)
    A_reg = np.vstack([A, np.eye(n_lines) * 1e-6])
    b_reg = np.concatenate([b, np.zeros(n_lines)])
    res = least_squares(lambda x: A_reg @ x - b_reg, np.zeros(n_lines), method='lm')
    corrections = res.x
    corr_dict = {line: corr for line, corr in zip(unique_lines, corrections)}
    
    # Apply corrections
    corrected = df['TMI'].values.copy()
    for line, corr in corr_dict.items():
        mask = df[line_col] == line
        corrected[mask] += corr
    return corrected, corr_dict

def ishihara_levelling(df, d0_km, t0_hours, d1_km=15.0, max_iter=5, tol=0.1):
    """
    Simplified Ishihara (2015) levelling – spatial + temporal filtering.
    Only for small to medium datasets (< 50000 points) due to O(N^2) complexity.
    """
    df = df.sort_values('datetime').copy().reset_index(drop=True)
    n = len(df)
    if n > 50000:
        st.warning(f"Ishihara levelling with {n} points may be very slow. Consider crossover levelling instead.")
        return df['TMI'].values.copy()
    
    times = df['datetime']
    t_seconds = (times - times.min()).dt.total_seconds().values
    
    # Approximate distances in meters
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(df['Latitude'].mean()))
    x = df['Longitude'].values * km_per_deg_lon
    y = df['Latitude'].values * km_per_deg_lat
    coords_m = np.column_stack((x, y))
    tree = cKDTree(coords_m)
    d1_m = d1_km * 1000.0
    
    neighbor_indices = []
    for i in range(n):
        idx = tree.query_ball_point(coords_m[i], d1_m)
        neighbor_indices.append([j for j in idx if j != i])
    
    def weight_func(d_m, d0_m):
        r = d_m / d0_m
        return 1.0 / ((1.0 + r*r)**2)
    
    def T_func(dt_sec, t1_sec):
        at = np.abs(dt_sec)
        if at <= t1_sec:
            return 0.0
        elif at <= 2*t1_sec:
            return at/t1_sec - 1.0
        return 1.0
    
    def G_func(dt_sec, t0_sec):
        at = np.abs(dt_sec)
        if at >= t0_sec:
            return 0.0
        return np.exp(-4.5 * (at/t0_sec)**2)
    
    d0_m = d0_km * 1000.0
    t0_sec = t0_hours * 3600.0
    t1_sec = t0_sec
    f1, f2 = 0.2, 0.05
    
    a = df['TMI'].values.copy()
    c = np.zeros(n)
    
    for _ in range(max_iter):
        numer = np.zeros(n)
        denom = np.zeros(n)
        for i in range(n):
            sum_w = 0.0
            sum_w_val = 0.0
            for j in neighbor_indices[i]:
                d_m = np.sqrt((coords_m[i][0]-coords_m[j][0])**2 + (coords_m[i][1]-coords_m[j][1])**2)
                if d_m > d1_m:
                    continue
                w = weight_func(d_m, d0_m)
                dt_sec = abs(t_seconds[i] - t_seconds[j])
                T = T_func(dt_sec, t1_sec)
                if T == 0:
                    continue
                weight = T * w
                sum_w += weight
                sum_w_val += weight * (a[j] + c[j] - a[i])
            if sum_w > 0:
                numer[i] = sum_w_val
                denom[i] = sum_w
        delta = np.divide(numer, denom, out=np.zeros_like(numer), where=denom>0)
        
        new_c = np.zeros(n)
        for i in range(n):
            sum_g = 0.0
            sum_g_val = 0.0
            for k in range(n):
                dt_sec = abs(t_seconds[i] - t_seconds[k])
                g = G_func(dt_sec, t0_sec)
                if g == 0:
                    continue
                y_k = denom[k] if denom[k] > 0 else 1.0
                sum_g += g * y_k
                sum_g_val += g * y_k * delta[k]
            if sum_g > 0:
                new_c[i] = sum_g_val / sum_g
        
        # damping
        for i in range(n):
            fi = denom[i]
            if fi > f1:
                pass
            elif fi > f2:
                new_c[i] *= (fi / f1)
            elif fi > 0:
                new_c[i] *= (fi / f1) * (fi / f2)
            else:
                new_c[i] = 0.0
        
        if np.max(np.abs(new_c - c)) < tol:
            c = new_c
            break
        c = new_c
    return a + c

# ------------------------------------------------------------------
# Main Streamlit UI
# ------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Input data preview (first 10 rows)")
    st.dataframe(df.head(10))
    
    # Required columns check
    required = ['Latitude', 'Longitude', 'TMI', 'datetime', 'Sheet_Name', 'Line_Name']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}. Please upload the correct CSV output from your processing pipeline.")
        st.stop()
    
    # Create a combined line identifier (Sheet_Name + Line_Name) for uniqueness
    df['Full_Line'] = df['Sheet_Name'].astype(str) + '_' + df['Line_Name'].astype(str)
    
    st.sidebar.header("Levelling Options")
    method = st.sidebar.selectbox("Method", ["Crossover adjustment (profile tying)", "Ishihara (spatial-temporal)"])
    
    if method == "Crossover adjustment (profile tying)":
        dist_thresh = st.sidebar.number_input("Crossover distance threshold (meters)", value=20.0, step=5.0, help="Points within this distance are considered crossovers.")
        if st.sidebar.button("Run Crossover Levelling"):
            with st.spinner("Detecting crossovers and solving for corrections..."):
                start = time.time()
                corrected, corr_dict = crossover_levelling(df, line_col='Full_Line', dist_threshold_m=dist_thresh)
                df['TMI_leveled'] = corrected
                st.session_state['df_leveled'] = df
                st.session_state['corrections'] = corr_dict
                st.session_state['method'] = method
                elapsed = time.time() - start
                st.success(f"Levelling completed in {elapsed:.2f} seconds. Found {len(corr_dict)} lines with corrections.")
    else:
        d0 = st.sidebar.number_input("Weight distance d0 (km)", value=0.5, step=0.1, format="%.1f")
        t0 = st.sidebar.number_input("Filter half-width t0 (hours)", value=3.0, step=0.5, help="Gaussian filter full width = 2*t0 hours.")
        if st.sidebar.button("Run Ishihara Levelling"):
            with st.spinner("Applying Ishihara spatial-temporal levelling..."):
                start = time.time()
                corrected = ishihara_levelling(df, d0_km=d0, t0_hours=t0)
                df['TMI_leveled'] = corrected
                st.session_state['df_leveled'] = df
                st.session_state['method'] = method
                elapsed = time.time() - start
                st.success(f"Levelling completed in {elapsed:.2f} seconds.")
    
    if 'df_leveled' in st.session_state:
        df_out = st.session_state['df_leveled']
        method_used = st.session_state['method']
        
        st.subheader("📊 Results after Levelling")
        dff = df_out[['Sheet_Name', 'Line_Name', 'datetime', 'Latitude', 'Longitude', 'TMI', 'TMI_leveled']].head(10)
        st.dataframe(dff)
        
        # --- Visualisations ---
        st.header("📈 Levelling Quality Assessment")
        
        # 1. Compute crossover differences before and after levelling
        if method_used == "Crossover adjustment (profile tying)":
            crossovers_before = find_crossovers(df_out, 'Full_Line', dist_threshold_m=20.0)
            crossovers_after = find_crossovers(df_out.assign(TMI=df_out['TMI_leveled']), 'Full_Line', dist_threshold_m=20.0)
            diffs_before = [c[4] for c in crossovers_before]
            diffs_after = [c[4] for c in crossovers_after]
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(diffs_before, bins=50, alpha=0.7, label='Before', color='red')
            axes[0].set_title(f"Crossover differences (nT)\nStd = {np.std(diffs_before):.2f}")
            axes[0].set_xlabel('Difference (nT)')
            axes[0].legend()
            axes[1].hist(diffs_after, bins=50, alpha=0.7, label='After', color='green')
            axes[1].set_title(f"Crossover differences (nT)\nStd = {np.std(diffs_after):.2f}")
            axes[1].set_xlabel('Difference (nT)')
            axes[1].legend()
            st.pyplot(fig)
            plt.close(fig)
            
            st.metric("Standard deviation of crossover differences", 
                      f"{np.std(diffs_before):.2f} → {np.std(diffs_after):.2f} nT",
                      delta=f"-{np.std(diffs_before)-np.std(diffs_after):.2f} nT")
        
        # 2. Map showing corrections per line (only for crossover method)
        if method_used == "Crossover adjustment (profile tying)" and 'corrections' in st.session_state:
            corr_dict = st.session_state['corrections']
            # Prepare data for map
            map_df = df_out.drop_duplicates(subset=['Full_Line'])[['Full_Line', 'Latitude', 'Longitude']].copy()
            map_df['Correction'] = map_df['Full_Line'].map(corr_dict).fillna(0.0)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sc = ax.scatter(map_df['Longitude'], map_df['Latitude'], c=map_df['Correction'], s=30, cmap='RdBu', edgecolor='k')
            plt.colorbar(sc, ax=ax, label='Correction (nT)')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('Line corrections from crossover levelling')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
        # 3. Optional: TMI profile comparison (select a line)
        st.subheader("📉 TMI Profile Comparison (select a line)")
        lines = df_out['Full_Line'].unique()
        sel_line = st.selectbox("Choose a line", lines)
        if sel_line:
            line_df = df_out[df_out['Full_Line'] == sel_line].sort_values('datetime')
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(line_df['datetime'], line_df['TMI'], 'b-', alpha=0.7, label='Original TMI')
            ax.plot(line_df['datetime'], line_df['TMI_leveled'], 'r-', alpha=0.7, label='Levelled TMI')
            ax.set_xlabel('Time')
            ax.set_ylabel('TMI (nT)')
            ax.set_title(f'Line {sel_line}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
        # Download
        st.header("💾 Download Levelled Data")
        output_cols = [c for c in df_out.columns if c not in ['Full_Line']]
        output_df = df_out[output_cols]
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "levelled_magnetic_data.csv", "text/csv")
else:
    st.info("⬅️ Upload a CSV file (output from your previous processing) to begin.")
