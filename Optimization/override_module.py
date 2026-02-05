import math
import pandas as pd

ACC_LINEAR = 100.0

# ==========================================================
# OVERRIDE CNC
# ==========================================================
def apply_override(df_base: pd.DataFrame, k: float) -> pd.DataFrame:
    df = df_base.copy()
    df["k"] = k
    df["F"] = df["F"] * k
    df["S"] = df["S"] * k
    return df

# ==========================================================
# RICALCOLO TEMPI
# ==========================================================
def recompute_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ideal_seg_time"] = 0.0

    for _, block in df.groupby("esecuzione_n", sort=False):
        idx = block.index
        seg_times = []

        for d, f in zip(block["dist"], block["F"]):
            if d > 0 and f > 0:
                v = f / 60.0
                a = ACC_LINEAR
                s_acc = v * v / a
                t = (
                    2 * v / a + (d - s_acc) / v
                    if d >= s_acc
                    else 2 * math.sqrt(d / a)
                )
            else:
                t = 0.0
            seg_times.append(t)

        df.loc[idx, "ideal_seg_time"] = seg_times

    df["t_ideal"] = df.groupby("esecuzione_n")["ideal_seg_time"].cumsum()
    df["t_ideal_cumul"] = df["ideal_seg_time"].cumsum()
    return df

# ==========================================================
# RICALCOLO MRR
# ==========================================================
def recompute_MRR(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ideal_MRR"] = 0.0

    for _, block in df.groupby("esecuzione_n", sort=False):
        idx = block.index
        vol = block["VOL_TOT"].dropna()
        if vol.empty:
            continue
        vol = vol.iloc[0]

        denom = (block["ideal_seg_time"] * block["F"]).sum()
        if denom > 0:
            df.loc[idx, "ideal_MRR"] = (vol / denom) * block["F"]

    return df