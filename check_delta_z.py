# -*- coding: utf-8 -*-
"""
Created on Thu May 14 22:19:51 2026

@author: HU JIAN
"""

import pandas as pd
import numpy as np

# 1. 读取你用来跑 MCMC 的配对 CSV 文件
csv_file = 'matched_cluster_milp_dd_pantheon.csv'
df = pd.read_csv(csv_file)

try:
    # 2. 提取超新星 (SN) 和致密射电源 (CRS) 的实际红移
    z_crs = df['z'].values
    z_sn = df['sn_matched_zHD'].values
    
    # 3. 计算红移差的绝对值
    delta_z = np.abs(z_sn - z_crs)
    
    # 4. 计算中位数和最大值
    median_dz = np.median(delta_z)
    max_dz = np.max(delta_z)
    
    print("\n========== 计算结果 ==========")
    print(f"总配对数量 (N)   : {len(delta_z)}")
    print(f"中位红移差 (Median) : {median_dz:.4f}")
    print(f"最大红移差 (Maximum): {max_dz:.4f}")
    print("==============================")
    
except KeyError as e:
    print(f"\n[错误] 找不到列名 {e}！请检查列名拼写。")