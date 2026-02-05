import pandas as pd
from openpyxl import load_workbook


def process_telemetries(input_path, output_filtered, sheet_compressed="Compressed"):
    
    """
    Pipeline:
    1. Filtra le colonne rilevanti
    2. Comprimi blocchi consecutivi con stesso GCode_GCode
    3. Salva file filtrato + foglio compresso nello stesso Excel
    """

    df = pd.read_excel(input_path)
    cols = list(df.columns)

    # Colonne sempre tenute
    key_tokens = ("Timestamp", "GCode_Line", "GCode_GCode", "GCode_IsoFile")
    always_keep = [c for c in cols if any(k in c for k in key_tokens)]

    axis_prefixes = ("RX", "RX1", "RA", "RY", "RZ")
    axis_suffixes = ("_OutPower", "_Rpm", "_Speed")

    selected = {
        c for c in cols
        if c in always_keep
        or ("Inverter" in c and c.endswith("_OutPower"))
        or (c.startswith(axis_prefixes) and c.endswith(axis_suffixes))
    }

    df_filtered = df[[c for c in cols if c in selected]]

    # --- COMPRESSIONE ---
    gcode_col = "GCode_GCode"
    df_filtered["group_id"] = (df_filtered[gcode_col] != df_filtered[gcode_col].shift()).cumsum()

    group_sizes = df_filtered.groupby("group_id").size()
    repetition_map = (group_sizes - 1).clip(lower=0)

    meta_cols = [c for c in ("Timestamp", "GCode_GCode", "GCode_Line", "GCode_IsoFile")
                 if c in df_filtered.columns]

    agg_dict = {
        c: (
            "first" if c in meta_cols
            else "mean" if pd.api.types.is_numeric_dtype(df_filtered[c])
            else "first"
        )
        for c in df_filtered.columns
        if c != "group_id"
    }

    compressed = (
        df_filtered
        .groupby("group_id", sort=False)
        .agg(agg_dict)
        .reset_index()
    )
    compressed["repetition"] = compressed["group_id"].map(repetition_map)
    compressed.drop(columns=["group_id"], inplace=True)

    # --- SALVATAGGIO ---
    df_filtered.to_excel(output_filtered, index=False)
    with pd.ExcelWriter(output_filtered, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        compressed.to_excel(writer, sheet_name=sheet_compressed, index=False)