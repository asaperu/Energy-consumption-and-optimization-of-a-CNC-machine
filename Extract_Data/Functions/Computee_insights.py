import os
import numpy as np
import pandas as pd


def compute_insights(merged_path: str, output_path: str):
    df = pd.read_excel(merged_path)

    # --- Parametri/colonne attese ---
    ESEC_COL = "esecuzione_n"
    FEED_COL = "F"
    VOL_COL  = "VOL_TOT"
    TS_COL   = "Timestamp"
    TIME_COL = "Time computed"
    REP_COL  = "repetition"

    # --- Controlli colonne di base ---
    for col in (ESEC_COL, FEED_COL, VOL_COL):
        if col not in df.columns:
            raise ValueError(
                f"La colonna '{col}' non è presente nel file merged. "
                f"Colonne disponibili: {list(df.columns)}"
            )

    # --- Tipi/normalizzazioni ---
    df[FEED_COL] = pd.to_numeric(df[FEED_COL], errors="coerce")
    df[VOL_COL]  = pd.to_numeric(df[VOL_COL],  errors="coerce")
    df[REP_COL]  = pd.to_numeric(df.get(REP_COL, 0.0), errors="coerce").fillna(0.0)

    # --- Calcolo TIME_COL se manca, usando Timestamp ---
    if TIME_COL not in df.columns:
        if TS_COL not in df.columns:
            raise ValueError(f"Manca sia '{TIME_COL}' che '{TS_COL}': non posso calcolare il tempo.")
        df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")

        time_comp = np.full(len(df), np.nan, dtype=float)
        for _, idx_block in df.groupby(ESEC_COL).groups.items():
            ts_block = df.loc[idx_block, TS_COL]
            valid = ts_block.dropna()
            if valid.empty:
                continue
            t0 = valid.iloc[0]
            secs = (ts_block - t0).dt.total_seconds()
            time_comp[idx_block] = secs.to_numpy()
        df[TIME_COL] = time_comp

    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")

    # --- Vettore dei feed presenti ---
    unique_F = sorted(df[FEED_COL].dropna().unique().tolist())

    # --- Costruzione colonne count_block_F* ---
    for idx_f, _ in enumerate(unique_F):
        df[f"count_block_F{idx_f}"] = 0.0

    for _, block in df.groupby(ESEC_COL, sort=False):
        rep = block[REP_COL]
        for idx_f, f_val in enumerate(unique_F):
            col = f"count_block_F{idx_f}"
            cond = (block[FEED_COL] == f_val)
            incr = cond.astype(int) * (1.0 + rep)
            df.loc[block.index, col] = incr.cumsum()

    # --- Colonne T* ---
    for idx_f, _ in enumerate(unique_F):
        df[f"T{idx_f}"] = 0.0

    for _, block in df.groupby(ESEC_COL, sort=False):
        idx = block.index
        time_vals = block[TIME_COL].dropna()
        T_tot = float(time_vals.iloc[-1]) if not time_vals.empty else 0.0

        final_counts = [
            float(block[f"count_block_F{idx_f}"].iloc[-1])
            for idx_f in range(len(unique_F))
        ]
        count_tot = sum(final_counts)

        if T_tot <= 0 or count_tot <= 0:
            for idx_f in range(len(unique_F)):
                df.loc[idx, f"T{idx_f}"] = 0.0
        else:
            for idx_f in range(len(unique_F)):
                Ti = T_tot * (final_counts[idx_f] / count_tot)
                df.loc[idx, f"T{idx_f}"] = Ti

    # --- MRR* per feed attivi (F > 0) ---
    active_indices = [i for i, f in enumerate(unique_F) if f > 0]
    for idx_f in range(len(unique_F)):
        df[f"MRR{idx_f}"] = 0.0

    for _, block in df.groupby(ESEC_COL, sort=False):
        idx = block.index
        vol_vals = block[VOL_COL].dropna()
        vol = float(vol_vals.iloc[0]) if not vol_vals.empty else 0.0

        T_vec = [float(block[f"T{idx_f}"].iloc[0]) for idx_f in range(len(unique_F))]
        denom = sum(T_vec[i] * unique_F[i] for i in active_indices)

        k = vol / denom if denom > 0 and vol > 0 else 0.0

        for idx_f, f_val in enumerate(unique_F):
            mrr_val = k * f_val if (f_val > 0 and k > 0) else 0.0
            df.loc[idx, f"MRR{idx_f}"] = mrr_val

    # --- MRR della riga corrente ---
    df["MRR"] = 0.0
    feed_to_idx = {f: i for i, f in enumerate(unique_F)}

    for _, block in df.groupby(ESEC_COL, sort=False):
        for i in block.index:
            f_val = df.at[i, FEED_COL]
            if f_val in feed_to_idx:
                idx_f = feed_to_idx[f_val]
                df.at[i, "MRR"] = df.at[i, f"MRR{idx_f}"]
            else:
                df.at[i, "MRR"] = 0.0

    # --- Potenza totale P (somma colonne *_OutPower / outpower) ---
    outpower_cols = [c for c in df.columns if "outpower" in c.lower()]
    if outpower_cols:
        df[outpower_cols] = df[outpower_cols].apply(pd.to_numeric, errors="coerce")
        df["P"] = df[outpower_cols].sum(axis=1)
    else:
        df["P"] = 0.0

    # --- Creazione directory di output se serve ---
    dirpath = os.path.dirname(output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # --- Fogli G0/G1/G2 con Timestamp non nullo ---
    if "movimento" not in df.columns:
        raise ValueError("Manca la colonna 'movimento' necessaria per creare i fogli G0/G1/G2.")
    if TS_COL not in df.columns:
        raise ValueError("Manca la colonna 'Timestamp' necessaria per filtrare i movimenti con Timestamp presente.")

    g0 = df[(df["movimento"] == "G0") & (df[TS_COL].notna())].copy()
    g1 = df[(df["movimento"] == "G1") & (df[TS_COL].notna())].copy()
    g2 = df[(df["movimento"] == "G2") & (df[TS_COL].notna())].copy()

    # --- Scrittura su 4 fogli nell'Excel di output (All, G0, G1, G2) ---
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All", index=False)
        g0.to_excel(writer, sheet_name="G0", index=False)
        g1.to_excel(writer, sheet_name="G1", index=False)
        g2.to_excel(writer, sheet_name="G2", index=False)

    return output_path
