import pandas as pd


def merge_gcode_telemetry(
    file_parsed,
    file_telemetry,
    output_path,
    max_steps=15,
):
    """
    Merge tra:
      - file_parsed: EXCEL con almeno ['esecuzione_n', 'N', 'code', ...]
      - file_telemetry: EXCEL con almeno ['GCode_Line', 'GCode_GCode', ...]
        (usa il foglio 'Compressed')

    Per ogni esecuzione_n:
      - si limita la telemetria all'intervallo di N
      - si scorre il blocco parsed e si cerca match 1-a-1
        con GCode_GCode entro max_steps righe a partire da seg_pos
    """

    df1 = pd.read_excel(file_parsed)
    df2 = pd.read_excel(file_telemetry, sheet_name="Compressed")
    print(f"[INFO] Telemetria letta dal foglio 'Compressed' di: {file_telemetry}")

    df1["N"] = pd.to_numeric(df1["N"], errors="coerce")
    df2["GCode_Line"] = pd.to_numeric(df2["GCode_Line"], errors="coerce")

    merged_rows = []

    for _, group in df1.groupby("esecuzione_n", sort=True):
        sort_cols = ["N"]
        if "timestamp" in group.columns:
            sort_cols.append("timestamp")
        block = group.sort_values(sort_cols, kind="mergesort")

        N_min, N_max = block["N"].min(), block["N"].max()

        if pd.isna(N_min) or pd.isna(N_max):
            segment = df2.reset_index(drop=True)
        else:
            line_min, line_max = N_min / 10.0, N_max / 10.0
            segment = (
                df2[
                    (df2["GCode_Line"] >= line_min)
                    & (df2["GCode_Line"] <= line_max)
                ]
                .copy()
                .reset_index(drop=True)
            )

        seg_pos = 0

        for _, row1 in block.iterrows():
            code_val = row1["code"]
            match_row2 = None

            if not segment.empty:
                start_idx = seg_pos
                end_idx = min(seg_pos + max_steps, len(segment))

                for i in range(start_idx, end_idx):
                    if segment.at[i, "GCode_GCode"] == code_val:
                        match_row2 = segment.iloc[i]
                        seg_pos = i + 1  # riga consumata
                        break

            if match_row2 is not None:
                merged_row = pd.concat([row1, match_row2], axis=0)
            else:
                empty_data = {col: None for col in df2.columns}
                merged_row = pd.concat([row1, pd.Series(empty_data)], axis=0)

            merged_rows.append(merged_row)

    merged_df = pd.DataFrame(merged_rows)
    merged_df.to_excel(output_path, index=False)