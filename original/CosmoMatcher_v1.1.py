# -*- coding: utf-8 -*-
"""
Created on Thu May  14 01:42:32 2026

@author: Hu Jian
Email：dg1626002@smail.nju.edu.cn
v2 update
Multi-probe Cosmological Data Matching Tool (V6.3: Full V5.7 Inheritance + 2x2 Ablation Study Edition)

Modifications:
1. Perfectly retained all V5.7 features: Independent extraction of 6 JLA components, SGLS cross-covariance extraction, smart CSV reading, error catching, custom result column selection.
2. Added ablation study dimensions:
   - Algorithm dimension: MILP Global Optimum vs Greedy Algorithm
   - Tolerance dimension: Δd/d Comoving distance tolerance vs Δz Absolute redshift tolerance
3. Dynamic export naming: Generated files will automatically include `_milp_dd_` or `_greedy_dz_` tags to facilitate paper data comparison and prevent overwriting.
"""

import sys
import traceback
import os

def show_error_and_wait():
    print("\n" + "!"*60)
    print("!!! FATAL ERROR: Program failed to start !!!")
    print("!"*60 + "\n")
    traceback.print_exc()
    input(">>> Please take a screenshot of this window and press Enter to exit...")
    sys.exit(1)

print("--- [1/4] Initializing Python interpreter... ---")

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import pandas as pd
    import numpy as np
    import time
    import threading
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.integrate import quad
    from scipy import sparse
    
    print("--- [2/4] All libraries loaded successfully ---")

    # ==========================================
    # Utility Functions
    # ==========================================
    def smart_read_csv(path, nrows=None, file_label="Data"):
        if not path: return None
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'latin1']
        if path.lower().endswith('.txt') or path.lower().endswith('.dat'):
            separators = [r'\s+', '\t', ',', ';']
        else:
            separators = [',', '\t', r'\s+', ';']
        last_err = ""
        for enc in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(path, sep=sep, nrows=nrows, encoding=enc, 
                                   engine='python', on_bad_lines='skip')
                    if len(df.columns) > 0:
                        if len(df.columns) >= 2 or nrows is not None:
                            return df
                except Exception as e:
                    last_err = str(e)
                    continue
        error_msg = f"Cannot read [{file_label}] file: {os.path.basename(path)}\nLast error: {last_err}"
        raise ValueError(error_msg)

    def load_jla_matrix(path):
        try:
            with open(path, 'r') as f:
                line = f.readline().strip()
                if not line: raise ValueError("File is empty")
                n_rows = int(line)
            data = np.loadtxt(path, skiprows=1)
            if data.size == n_rows * n_rows:
                return data.reshape((n_rows, n_rows))
            else:
                if data.shape == (n_rows, n_rows):
                    return data
                raise ValueError(f"Data size {data.size} does not match declared dimensions {n_rows}x{n_rows}")
        except Exception as e:
            raise ValueError(f"Failed to read JLA matrix {os.path.basename(path)}: {e}")

    def load_pantheon_matrix(path):
        try:
            with open(path, 'r') as f:
                try:
                    n = int(f.readline().strip())
                    skip = 1
                except:
                    n = 0
                    skip = 0
            data = np.loadtxt(path, skiprows=skip)
            if data.ndim == 2 and data.shape[0] == data.shape[1]:
                return data
            if n > 0:
                return data.reshape((n, n))
            else:
                sz = int(np.sqrt(data.size))
                return data.reshape((sz, sz))
        except Exception as e:
            raise ValueError(f"Failed to read matrix: {e}")

    # ==========================================
    # Logical Backend
    # ==========================================
    class CosmoBackend:
        @staticmethod
        def integrand(z, wm):
            return (wm * (1 + z)**3 + (1 - wm))**-0.5

        @staticmethod
        def comoving_distance(z_val, wm):
            if z_val <= 0: return 0.0
            result, _ = quad(CosmoBackend.integrand, 0, z_val, args=(wm,))
            return result

        @staticmethod
        def run_main_logic(mode, sn_type, algo, tol_type, file_paths, col_cfg, params, log_func):
            try:
                # --- 1. Data Loading ---
                try:
                    tracer_df = smart_read_csv(file_paths['tracer'], file_label="Photometric Data (SN)")
                except Exception as e: log_func(f"❌ {e}"); return

                tracer_df['original_index'] = tracer_df.index
                tracer_df = tracer_df.dropna(subset=[col_cfg['tracer_z_col']])
                log_func(f"Photometric data loaded: {len(tracer_df)} rows (from {os.path.basename(file_paths['tracer'])})")

                try:
                    obj_df = smart_read_csv(file_paths['obj'], file_label="Target Data (Lens/Cluster)")
                except Exception as e: log_func(f"❌ {e}"); return

                obj_df['obj_index'] = obj_df.index
                for c in [col_cfg['z1_col'], col_cfg.get('z2_col')]:
                    if c: obj_df[c] = pd.to_numeric(obj_df[c], errors='coerce')
                obj_df.dropna(subset=[col_cfg['z1_col']], inplace=True)
                if mode == 'sgls': obj_df.dropna(subset=[col_cfg['z2_col']], inplace=True)
                log_func(f"Target data loaded: {len(obj_df)} rows (from {os.path.basename(file_paths['obj'])})")

                # --- 2. Covariance Loading ---
                cov_data = {} 
                if sn_type == 'pantheon':
                    p = file_paths.get('cov_pantheon')
                    if p and p.strip():
                        log_func(f"Loading Pantheon+ covariance...")
                        cov_data['full'] = load_pantheon_matrix(p)
                elif sn_type == 'jla':
                    jla_keys = ['v0', 'va', 'vb', 'v0a', 'v0b', 'vab']
                    log_func("Loading the 6 JLA component matrices...")
                    for k in jla_keys:
                        p = file_paths.get(f'cov_jla_{k}')
                        if p and p.strip():
                            try:
                                cov_data[k] = load_jla_matrix(p)
                                log_func(f"  -> Loaded {k}")
                            except Exception as e:
                                log_func(f"❌ Failed to load {k}: {e}")

                # --- 3. Parameter and Condition Calculation ---
                wm = params['wm']
                tol = params['tol']
                
                tracer_indices = tracer_df['original_index'].values
                tracer_zs = tracer_df[col_cfg['tracer_z_col']].values
                
                if tol_type == 'dd':
                    log_func(">>> Calculating comoving distances for photometric data (Δd/d mode)...")
                    tracer_df['d_c'] = tracer_df[col_cfg['tracer_z_col']].apply(lambda z: CosmoBackend.comoving_distance(z, wm))
                    tracer_ds = tracer_df['d_c'].values
                else:
                    log_func(">>> Using raw redshift for absolute deviation matching (Δz mode)...")
                    tracer_ds = None

                final_results = []
                sn_roles = []

                # Core validation function: Unify handling of DD and DZ tolerances
                def evaluate_match(z_target, d_target, t_zs, t_ds):
                    if tol_type == 'dz':
                        err = np.abs(t_zs - z_target)
                        return err <= tol, err  # Returns: mask, error (for optimization)
                    else: # 'dd'
                        abs_diff = np.abs(t_ds - d_target)
                        rel_err = abs_diff / d_target
                        return rel_err <= tol, abs_diff

                # ==========================================================
                # Branch A: Cluster Mode - Always uses greedy
                # ==========================================================
                if mode == 'cluster':
                    log_func(f"\n>>> [Mode: CLUSTER] Enabling greedy global matching (Criterion: {tol_type.upper()}, Tolerance: {tol})...")
                    sn_roles = ['sn_matched']
                    z_col = col_cfg['z1_col']
                    potential_matches = []
                    z_vals = obj_df[z_col].values
                    obj_idxs = obj_df['obj_index'].values

                    for i in range(len(obj_df)):
                        oid = obj_idxs[i]
                        z_val = z_vals[i]
                        d_obj = CosmoBackend.comoving_distance(z_val, wm) if tol_type == 'dd' else None
                        
                        mask, cost = evaluate_match(z_val, d_obj, tracer_zs, tracer_ds)
                        valid_idxs = np.where(mask)[0]
                        
                        for idx in valid_idxs:
                            potential_matches.append({
                                'obj_index': oid, 'sn_matched_idx': tracer_indices[idx], 'error': cost[idx]
                            })
                    
                    log_func(f"Initial screening: {len(potential_matches)} potential pairs...")
                    potential_matches.sort(key=lambda x: x['error'])
                    used_sn = set(); used_obj = set()
                    for m in potential_matches:
                        if m['obj_index'] in used_obj or m['sn_matched_idx'] in used_sn: continue
                        final_results.append(m)
                        used_obj.add(m['obj_index']); used_sn.add(m['sn_matched_idx'])
                    log_func(f"✅ Matching complete! Matched {len(final_results)} pairs in total.")

                # ==========================================================
                # Branch B: Strong Lensing Mode (SGLS) - Core Ablation Area
                # ==========================================================
                else:
                    log_func(f"\n>>> [Mode: SGLS] Algorithm: {algo.upper()}, Physical Criterion: {tol_type.upper()}, Tolerance: {tol}")
                    sn_roles = ['sn_lens', 'sn_source']
                    zl_arr, zs_arr = obj_df[col_cfg['z1_col']].values, obj_df[col_cfg['z2_col']].values
                    obj_idxs = obj_df['obj_index'].values
                    
                    if tol_type == 'dd':
                        dl_arr = np.array([CosmoBackend.comoving_distance(z, wm) for z in zl_arr])
                        ds_arr = np.array([CosmoBackend.comoving_distance(z, wm) for z in zs_arr])
                    else:
                        dl_arr = ds_arr = [None] * len(zl_arr)

                    # --- Greedy Algorithm ---
                    if algo == 'greedy':
                        used_sn = set()
                        for i in range(len(obj_df)):
                            oid, zl, zs, dl, ds = obj_idxs[i], zl_arr[i], zs_arr[i], dl_arr[i], ds_arr[i]
                            if tol_type == 'dd' and (dl <= 0 or ds <= 0): continue
                            
                            # 1. Find the nearest Lens SN
                            mask_l, cost_l = evaluate_match(zl, dl, tracer_zs, tracer_ds)
                            valid_l = np.where(mask_l & (~np.isin(tracer_indices, list(used_sn))))[0]
                            if len(valid_l) == 0: continue
                            best_l_idx = valid_l[np.argmin(cost_l[valid_l])]
                            sn_l_id = tracer_indices[best_l_idx]
                            
                            # 2. Find the nearest Source SN
                            used_temp = used_sn.copy(); used_temp.add(sn_l_id)
                            mask_s, cost_s = evaluate_match(zs, ds, tracer_zs, tracer_ds)
                            valid_s = np.where(mask_s & (~np.isin(tracer_indices, list(used_temp))))[0]
                            if len(valid_s) == 0: continue
                            best_s_idx = valid_s[np.argmin(cost_s[valid_s])]
                            sn_s_id = tracer_indices[best_s_idx]
                            
                            # 3. Lock and record
                            used_sn.update([sn_l_id, sn_s_id])
                            final_results.append({'obj_index': oid, 'sn_lens_idx': sn_l_id, 'sn_source_idx': sn_s_id})
                            
                        log_func(f"✅ Greedy matching complete! Extracted {len(final_results)} pairs.")

                    # --- MILP Global Optimization ---
                    else:
                        candidates = [] 
                        for i in range(len(obj_df)):
                            oid, zl, zs, dl, ds = obj_idxs[i], zl_arr[i], zs_arr[i], dl_arr[i], ds_arr[i]
                            if tol_type == 'dd' and (dl <= 0 or ds <= 0): continue
                            
                            mask_l, cost_l = evaluate_match(zl, dl, tracer_zs, tracer_ds)
                            mask_s, cost_s = evaluate_match(zs, ds, tracer_zs, tracer_ds)
                            
                            idxs_l, idxs_s = tracer_indices[mask_l], tracer_indices[mask_s]
                            c_l, c_s = cost_l[mask_l], cost_s[mask_s]
                            
                            if len(idxs_l) == 0 or len(idxs_s) == 0: continue
                            
                            for k, sn1 in enumerate(idxs_l):
                                for m, sn2 in enumerate(idxs_s):
                                    if sn1 != sn2:
                                        # Use the sum of squared lens and source errors as cost
                                        candidates.append((oid, sn1, sn2, c_l[k]**2 + c_s[m]**2))
                                        
                        if not candidates: log_func("❌ MILP: No candidate combinations found."); return

                        all_obj_ids = obj_df['obj_index'].values
                        map_obj = {uid: i for i, uid in enumerate(all_obj_ids)}
                        unique_sn = set(g[1] for g in candidates) | set(g[2] for g in candidates)
                        map_sn = {uid: (i + len(all_obj_ids)) for i, uid in enumerate(unique_sn)}
                        
                        rows, cols, data = [], [], []
                        c_vec = -1.0 + (1e-6 * np.array([g[3] for g in candidates]))
                        
                        for j, (oid, sn1, sn2, _) in enumerate(candidates):
                            if oid in map_obj:
                                rows.extend([map_obj[oid], map_sn[sn1], map_sn[sn2]])
                                cols.extend([j, j, j])
                                data.extend([1, 1, 1])

                        A = sparse.coo_matrix((data, (rows, cols)), shape=(len(map_obj)+len(map_sn), len(candidates)))
                        res = milp(c=c_vec, constraints=LinearConstraint(A, -np.inf, 1), 
                                   integrality=np.ones_like(c_vec), bounds=Bounds(0, 1))
                        
                        if res.success:
                            sel_idx = np.where(res.x > 0.5)[0]
                            for idx in sel_idx:
                                g = candidates[idx]
                                final_results.append({'obj_index': g[0], 'sn_lens_idx': g[1], 'sn_source_idx': g[2]})
                            log_func(f"✅ MILP optimization complete! Extracted {len(final_results)} pairs.")
                        else:
                            log_func(f"❌ Solver failed: {res.message}"); return

                # --- 4. Export CSV Results ---
                if not final_results: log_func("⚠️ Results are empty."); return
                df_res = pd.DataFrame(final_results)
                merged = df_res.merge(obj_df, on='obj_index')
                
                out_cols = list(set(col_cfg['output_cols'] + ['original_index']))
                tracer_sub = tracer_df[out_cols].copy()
                
                merged = merged.merge(tracer_sub.add_prefix(sn_roles[0]+'_'), left_on=sn_roles[0]+'_idx', right_on=sn_roles[0]+'_original_index')
                if len(sn_roles) > 1:
                    merged = merged.merge(tracer_sub.add_prefix(sn_roles[1]+'_'), left_on=sn_roles[1]+'_idx', right_on=sn_roles[1]+'_original_index')
                
                file_tag = f"{algo}_{tol_type}_{sn_type}" #update 2026.05.14
                csv_name = f"matched_{mode}_{file_tag}.csv"
                merged.to_csv(csv_name, index=False)
                log_func(f"Table saved: {csv_name}")

                # --- 5. Covariance Extraction ---
                if cov_data:
                    log_func(">>> Extracting and saving sub-covariance matrices...")
                    
                    tasks = []
                    if mode == 'cluster':
                        idx = merged['sn_matched_idx'].values.astype(int)
                        tasks.append(("lens", idx, idx)) 
                    else:
                        idx_l = merged['sn_lens_idx'].values.astype(int)
                        idx_s = merged['sn_source_idx'].values.astype(int)
                        tasks.append(("lens", idx_l, idx_l))     
                        tasks.append(("source", idx_s, idx_s))   
                        tasks.append(("cross", idx_l, idx_s))    

                    for suffix, rows, cols in tasks:
                        for key, mat in cov_data.items():
                            if mat is None: continue
                            sub_cov = mat[np.ix_(rows, cols)]
                            fname = f"cov_{mode}_{file_tag}_{suffix}_{key}.txt" if key != 'full' else f"cov_{mode}_{file_tag}_{suffix}.txt"
                            np.savetxt(fname, sub_cov, fmt='%.8e')
                            log_func(f"  -> Saved: {fname}")

                log_func("=== All tasks completed ===")

            except Exception as e:
                log_func(f"❌ Unknown error: {e}")
                traceback.print_exc()

    # ==========================================
    # GUI Frontend 
    # ==========================================
    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("CosmoMatcher V1.1 (Full Version with Ablation Engine)")
            self.geometry("950x980")
            
            self.mode_var = tk.StringVar(value="sgls") 
            self.sn_type_var = tk.StringVar(value="jla")
            
            self.algo_var = tk.StringVar(value="milp")
            self.tol_type_var = tk.StringVar(value="dd")
            
            self.path_tracer = tk.StringVar()
            self.path_obj = tk.StringVar()
            self.path_cov_pantheon = tk.StringVar()
            self.paths_cov_jla = {k: tk.StringVar() for k in ['v0', 'va', 'vb', 'v0a', 'v0b', 'vab']}
            self.tracer_cols = []
            self.obj_cols = []
            self.setup_ui()
            
        def setup_ui(self):
            f_top = ttk.Frame(self); f_top.pack(fill="x", padx=10, pady=5)
            
            f_mode = ttk.LabelFrame(f_top, text="Matching Mode", padding=5)
            f_mode.grid(row=0, column=0, padx=5, pady=2, sticky="ew")
            ttk.Radiobutton(f_mode, text="Strong Gravitational Lenses (2 SNe)", variable=self.mode_var, value="sgls", command=self.refresh_ui).pack(side="left", padx=5)
            ttk.Radiobutton(f_mode, text="Galaxy Clusters (1 SN)", variable=self.mode_var, value="cluster", command=self.refresh_ui).pack(side="left", padx=5)

            f_sn = ttk.LabelFrame(f_top, text="Supernova Sample Type", padding=5)
            f_sn.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
            ttk.Radiobutton(f_sn, text="Pantheon+ (Single Matrix)", variable=self.sn_type_var, value="pantheon", command=self.refresh_cov_ui).pack(side="left", padx=5)
            ttk.Radiobutton(f_sn, text="JLA (6-Component Matrices)", variable=self.sn_type_var, value="jla", command=self.refresh_cov_ui).pack(side="left", padx=5)

            f_ablation = ttk.LabelFrame(f_top, text="Ablation Study Settings", padding=5)
            f_ablation.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
            
            ttk.Label(f_ablation, text="Algorithm:").grid(row=0, column=0, padx=5)
            ttk.Radiobutton(f_ablation, text="MILP Global Optimization", variable=self.algo_var, value="milp").grid(row=0, column=1, padx=5)
            ttk.Radiobutton(f_ablation, text="Greedy Algorithm", variable=self.algo_var, value="greedy").grid(row=0, column=2, padx=5)
            
            ttk.Label(f_ablation, text="| Tolerance Criterion:").grid(row=0, column=3, padx=5)
            ttk.Radiobutton(f_ablation, text="Δd/d (Comoving Distance)", variable=self.tol_type_var, value="dd").grid(row=0, column=4, padx=5)
            ttk.Radiobutton(f_ablation, text="Δz (Absolute Redshift)", variable=self.tol_type_var, value="dz").grid(row=0, column=5, padx=5)

            f_files = ttk.LabelFrame(self, text="1. Data Loading", padding=10)
            f_files.pack(fill="x", padx=10, pady=2)
            self.create_file_row(f_files, "Photometric Data (SN):", self.path_tracer, self.on_tracer_loaded)
            self.lbl_obj = self.create_file_row(f_files, "Lenses/Clusters:", self.path_obj, self.on_obj_loaded)
            
            self.f_cov = ttk.LabelFrame(self, text="2. Covariance Matrix Settings (Required for Joint Constraints)", padding=10)
            self.f_cov.pack(fill="x", padx=10, pady=2)
            self.refresh_cov_ui() 

            f_cols = ttk.LabelFrame(self, text="3. Redshift Column Mapping", padding=10)
            f_cols.pack(fill="x", padx=10, pady=2)
            self.lbl_z1 = ttk.Label(f_cols, text="Lens Redshift (zl):"); self.cb_z1 = ttk.Combobox(f_cols, width=10)
            self.lbl_z2 = ttk.Label(f_cols, text="Source Redshift (zs):"); self.cb_z2 = ttk.Combobox(f_cols, width=10)
            self.lbl_tracer_z = ttk.Label(f_cols, text="SN Data Redshift (z):"); self.cb_tracer_z = ttk.Combobox(f_cols, width=10)

            f_out = ttk.LabelFrame(self, text="4. Output Columns to Keep", padding=10)
            f_out.pack(fill="both", expand=True, padx=10)
            f_list = ttk.Frame(f_out); f_list.pack(fill="both", expand=True)
            sb = ttk.Scrollbar(f_list, orient="vertical")
            self.lb_cols = tk.Listbox(f_list, selectmode="multiple", height=5, yscrollcommand=sb.set)
            sb.config(command=self.lb_cols.yview)
            sb.pack(side="right", fill="y"); self.lb_cols.pack(side="left", fill="both", expand=True)
            ttk.Button(f_out, text="Select Recommended Columns", command=self.select_defaults).pack()

            f_run = ttk.LabelFrame(self, text="5. Execution Parameters", padding=10)
            f_run.pack(fill="x", padx=10)
            ttk.Label(f_run, text="Omega_m:").pack(side="left")
            self.en_wm = ttk.Entry(f_run, width=8); self.en_wm.insert(0, "0.300"); self.en_wm.pack(side="left", padx=5)
            ttk.Label(f_run, text="Tolerance Value (e.g., 0.05 or 0.005):").pack(side="left", padx=(10,0))
            self.en_tol = ttk.Entry(f_run, width=8); self.en_tol.insert(0, "0.05"); self.en_tol.pack(side="left", padx=5)
            self.btn_run = ttk.Button(f_run, text="Start Matching and Extract Covariances", command=self.start)
            self.btn_run.pack(side="right")

            self.txt_log = tk.Text(self, height=8); self.txt_log.pack(fill="x", padx=10, pady=5)
            self.refresh_ui()

        def refresh_cov_ui(self):
            for widget in self.f_cov.winfo_children(): widget.destroy()
            stype = self.sn_type_var.get()
            if stype == 'pantheon':
                self.create_file_row(self.f_cov, "Full Covariance Matrix:", self.path_cov_pantheon)
            else:
                tk.Label(self.f_cov, text="Please import the 6 JLA component matrices (.dat):").grid(row=0, column=0, columnspan=3, sticky="w")
                labels = {'v0': 'Mag Cov (v0)', 'va': 'Stretch Cov (va)', 'vb': 'Colour Cov (vb)',
                          'v0a': 'Mag-Stretch (v0a)', 'v0b': 'Mag-Colour (v0b)', 'vab': 'Stretch-Colour (vab)'}
                keys = list(labels.keys())
                for i, k in enumerate(keys):
                    row = (i // 2) + 1; col = (i % 2) * 3 
                    ttk.Label(self.f_cov, text=labels[k]).grid(row=row, column=col, sticky="e", padx=5, pady=2)
                    ttk.Entry(self.f_cov, textvariable=self.paths_cov_jla[k], width=25).grid(row=row, column=col+1, padx=2)
                    ttk.Button(self.f_cov, text="...", width=3, command=lambda v=self.paths_cov_jla[k]: self.browse(v, None)).grid(row=row, column=col+2, padx=5)

        def refresh_ui(self):
            mode = self.mode_var.get()
            self.lbl_z1.pack_forget(); self.cb_z1.pack_forget()
            self.lbl_z2.pack_forget(); self.cb_z2.pack_forget()
            self.lbl_tracer_z.pack_forget(); self.cb_tracer_z.pack_forget()
            
            self.lbl_z1.pack(side="left"); self.cb_z1.pack(side="left", padx=5)
            if mode == "cluster":
                self.lbl_obj.config(text="Cluster/Single Object Data:")
                self.lbl_z1.config(text="Cluster Redshift (z):")
            else:
                self.lbl_obj.config(text="Strong Lens Data:")
                self.lbl_z1.config(text="Lens Redshift (zl):")
                self.lbl_z2.pack(side="left", padx=(10,0)); self.cb_z2.pack(side="left", padx=5)
            self.lbl_tracer_z.pack(side="left", padx=(10,0)); self.cb_tracer_z.pack(side="left", padx=5)

        def create_file_row(self, p, txt, var, cb=None):
            f = ttk.Frame(p); f.pack(fill="x", pady=2)
            l = ttk.Label(f, text=txt, width=20, anchor="w"); l.pack(side="left")
            ttk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True, padx=5)
            ttk.Button(f, text="Browse", command=lambda: self.browse(var, cb)).pack(side="right")
            return l

        def browse(self, var, cb):
            path = filedialog.askopenfilename()
            if path: var.set(path); cb(path) if cb else None

        def on_tracer_loaded(self, path):
            try:
                df = smart_read_csv(path, nrows=1)
                self.tracer_cols = list(df.columns)
                self.cb_tracer_z['values'] = self.tracer_cols
                for c in self.tracer_cols:
                    if c.lower() in ['zhd', 'z', 'redshift', 'zcmb']: self.cb_tracer_z.set(c)
                self.lb_cols.delete(0, tk.END)
                for c in self.tracer_cols: self.lb_cols.insert(tk.END, c)
                self.log(f"Tracer Columns: {len(self.tracer_cols)}")
            except Exception as e: self.log(f"Err: {e}")

        def on_obj_loaded(self, path):
            try:
                df = smart_read_csv(path, nrows=1)
                self.obj_cols = list(df.columns)
                self.cb_z1['values'] = self.obj_cols
                self.cb_z2['values'] = self.obj_cols
                for c in self.obj_cols:
                    cl = c.lower()
                    if cl in ['z', 'zl', 'z_lens']: self.cb_z1.set(c)
                    if cl in ['zs', 'z_source']: self.cb_z2.set(c)
                self.log(f"Object Columns: {len(self.obj_cols)}")
            except Exception as e: self.log(f"Err: {e}")

        def select_defaults(self):
            defaults = ['CID', 'zHD', 'MU_SH0ES', 'CEPH_DIST', 'IS_CALIBRATOR', 'Name', 'z', 'dist_mod', 
                        'mb', 'x1', 'c', 'dmb', 'dx1', 'dc', 'zcmb']
            self.lb_cols.selection_clear(0, tk.END)
            for i, c in enumerate(self.tracer_cols):
                if c in defaults or c.lower() in defaults: self.lb_cols.selection_set(i)

        def log(self, m): self.txt_log.insert(tk.END, m+"\n"); self.txt_log.see(tk.END)

        def start(self):
            mode = self.mode_var.get()
            sn_type = self.sn_type_var.get()
            algo = self.algo_var.get()
            tol_type = self.tol_type_var.get()
            
            if not self.path_tracer.get() or not self.path_obj.get(): return messagebox.showwarning("Warning", "Please load main data files")
            out_cols = [self.lb_cols.get(i) for i in self.lb_cols.curselection()]
            
            cfg = {'tracer_z_col': self.cb_tracer_z.get(), 'z1_col': self.cb_z1.get(),
                   'z2_col': self.cb_z2.get() if mode == 'sgls' else None, 'output_cols': out_cols}
            
            if not cfg['tracer_z_col'] or not cfg['z1_col']: return messagebox.showwarning("Warning", "Please specify redshift columns")
            
            file_paths = {
                'tracer': self.path_tracer.get(),
                'obj': self.path_obj.get(),
                'cov_pantheon': self.path_cov_pantheon.get()
            }
            if sn_type == 'jla':
                for k, v in self.paths_cov_jla.items():
                    val = v.get()
                    if val: file_paths[f'cov_jla_{k}'] = val

            try: params = {'wm': float(self.en_wm.get()), 'tol': float(self.en_tol.get())}
            except: return messagebox.showerror("Error", "Parameter Error")

            self.btn_run.config(state="disabled")
            t = threading.Thread(target=self.run_bg, args=(mode, sn_type, algo, tol_type, file_paths, cfg, params))
            t.daemon = True; t.start()

        def run_bg(self, mode, sn_type, algo, tol_type, fpaths, cfg, params):
            def log_cb(m): self.after(0, lambda: self.log(m))
            CosmoBackend.run_main_logic(mode, sn_type, algo, tol_type, fpaths, cfg, params, log_cb)
            self.after(0, lambda: self.btn_run.config(state="normal"))

    print("--- [3/4] Class definitions complete ---")

except Exception as e:
    show_error_and_wait()

if __name__ == "__main__":
    print("--- [4/4] Starting UI... ---")
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        show_error_and_wait()