import re
import math
import pandas as pd


ACC_LINEAR = 100.0


def parse_lavorazioni_regex(file_path: str) -> pd.DataFrame:
    start_re = re.compile(r'\*{3}\s*Esecuzione lavorazione')
    end_re = re.compile(r'\*{7}\s*FINE INFO VOLUMI ASPORTATI')

    line_re = re.compile(
        r'(?:\bN(?P<N>\d+))?'
        r'(?:.*?\bG(?P<G>[0123]))?'
        r'(?:.*?X(?P<X>-?\d+(?:[.,]\d+)?))?'
        r'(?:.*?Y(?P<Y>-?\d+(?:[.,]\d+)?))?'
        r'(?:.*?Z(?P<Z>-?\d+(?:[.,]\d+)?))?'
        r'(?:.*?I(?P<I>-?\d+(?:[.,]\d+)?))?'
        r'(?:.*?J(?P<J>-?\d+(?:[.,]\d+)?))?'
        r'(?:.*?\bF(?P<F>\d+(?:[.,]\d+)?))?'
    )

    def to_float(val):
        return float(val.replace(',', '.')) if val is not None else None

    def versor(value: float) -> int:
        if value > 0:
            return 1
        elif value < 0:
            return -1
        else:
            return 0

    records = []
    in_block = False

    last_x = last_y = last_z = None
    prev_x = prev_y = prev_z = None

    esecuzione_idx = 0
    esecuzione_nome = None

    current_diametro = None
    block_diametro = None
    current_vol_tot = None
    block_start_idx = None
    current_s = None
    cum_t_ideal = 0.0

    with open(file_path, 'r', encoding='latin-1') as f:
        for raw_line in f:
            original_line = raw_line.rstrip('\n')
            line = original_line.strip()

            # Lettura S
            m_s = re.search(r'\bS([0-9.,]+)', line)
            if m_s:
                current_s = to_float(m_s.group(1))

            # Lettura diametro utensile
            m_diam = re.compile(
                r'DIAMETRO\s+UTENSILE:\s*([0-9.,]+)', re.IGNORECASE
            ).search(line)
            if m_diam:
                current_diametro = to_float(m_diam.group(1))

            # Lettura VOL_TOT (solo dentro al blocco)
            m_vol = re.search(r"VOL_TOT:\s*([0-9.,]+)", line, re.IGNORECASE)
            if m_vol and in_block:
                current_vol_tot = to_float(m_vol.group(1))
                if block_start_idx is not None:
                    for rec in records[block_start_idx:]:
                        rec['VOL_TOT'] = current_vol_tot

            # Inizio blocco lavorazione
            if start_re.search(line):
                in_block = True
                esecuzione_idx += 1

                last_x = last_y = last_z = None
                prev_x = prev_y = prev_z = None

                block_diametro = current_diametro
                current_vol_tot = None
                block_start_idx = len(records)
                cum_t_ideal = 0.0

                m_nome = re.search(r'\*{3}\s*Esecuzione lavorazione\s*(.*)', line)
                esecuzione_nome = (
                    m_nome.group(1).strip()
                    if m_nome and m_nome.group(1).strip()
                    else f"Lavorazione_{esecuzione_idx}"
                )
                continue

            # Fine blocco lavorazione
            if end_re.search(line):
                in_block = False
                esecuzione_nome = None
                block_diametro = None
                current_vol_tot = None
                block_start_idx = None
                continue

            if not in_block:
                continue

            # Salta commenti / righe vuote
            if not line or line.startswith(('(', ';')):
                continue

            m = line_re.search(line)
            if not m:
                continue

            g_str = m.group('G')
            if g_str is None:
                continue

            movimento = f"G{g_str}"

            if movimento in ('G2', 'G3'):
                i_val = to_float(m.group('I'))
                j_val = to_float(m.group('J'))
            else:
                i_val = j_val = None

            n_str = m.group('N')
            x_val = to_float(m.group('X'))
            y_val = to_float(m.group('Y'))
            z_val = to_float(m.group('Z'))
            f_val = to_float(m.group('F')) if m.group('F') else 0.0

            # Mantieni l'ultimo valore noto di X,Y,Z se omessi
            x_val = last_x if x_val is None and last_x is not None else x_val
            y_val = last_y if y_val is None and last_y is not None else y_val
            z_val = last_z if z_val is None and last_z is not None else z_val

            if x_val is not None:
                last_x = x_val
            if y_val is not None:
                last_y = y_val
            if z_val is not None:
                last_z = z_val

            cur_x = x_val or 0.0
            cur_y = y_val or 0.0
            cur_z = z_val or 0.0

            # Delta rispetto alla riga precedente
            if prev_x is None:
                delta_x = delta_y = delta_z = 0.0
            else:
                delta_x = cur_x - prev_x
                delta_y = cur_y - prev_y
                delta_z = cur_z - prev_z

            xv = versor(delta_x)
            yv = versor(delta_y)
            zv = versor(delta_z)

            # Raggio per G2/G3
            if movimento in ('G2', 'G3') and i_val is not None and j_val is not None:
                R_val = math.sqrt((cur_x - i_val) ** 2 + (cur_y - j_val) ** 2)
            else:
                R_val = 0.0

            theta_val = None
            dist_lin = None
            dist_circ = None
            dist = None

            # Distanza (lineare o circolare)
            if (
                movimento in ('G2', 'G3')
                and i_val is not None and j_val is not None
                and prev_x is not None and prev_y is not None
            ):
                vx_prev = prev_x - i_val
                vy_prev = prev_y - j_val
                vx_curr = cur_x - i_val
                vy_curr = cur_y - j_val

                num = vx_prev * vx_curr + vy_prev * vy_curr
                den = math.sqrt(vx_prev ** 2 + vy_prev ** 2) * math.sqrt(vx_curr ** 2 + vy_curr ** 2)

                if den != 0:
                    cos_theta = max(-1.0, min(1.0, num / den))
                    theta_val = math.acos(cos_theta)
                    dist_circ = R_val * theta_val
                    dist = dist_circ
            else:
                if prev_x is not None and prev_y is not None and prev_z is not None:
                    dist_lin = math.sqrt(
                        (cur_x - prev_x) ** 2 +
                        (cur_y - prev_y) ** 2 +
                        (cur_z - prev_z) ** 2
                    )
                else:
                    dist_lin = 0.0
                dist = dist_lin

            # Tempo ideale del segmento con profilo trapezoidale (ACC_LINEAR)
            if dist is not None and dist > 0 and f_val not in (None, 0):
                v_cmd = f_val / 60.0  # mm/min -> mm/s
                a = ACC_LINEAR
                t_acc = v_cmd / a
                s_acc = 0.5 * v_cmd * v_cmd / a
                if dist >= 2.0 * s_acc:
                    seg_t_ideal = 2.0 * t_acc + (dist - 2.0 * s_acc) / v_cmd
                else:
                    seg_t_ideal = 2.0 * math.sqrt(dist / a)
            else:
                seg_t_ideal = 0.0

            cum_t_ideal += seg_t_ideal
            t_ideal = cum_t_ideal

            # Estraggo la linea di codice G completa (per debug)
            m_code = re.search(r'\bG[0123].*', original_line)
            code_line = m_code.group(0).strip() if m_code else None

            record = {
                'esecuzione_n': esecuzione_idx,
                'esecuzione_nome': esecuzione_nome,
                'N': int(n_str) if n_str else None,
                'movimento': movimento,
                'X': x_val,
                'Y': y_val,
                'Z': z_val,
                'delta_x': delta_x,
                'delta_y': delta_y,
                'delta_z': delta_z,
                'xv': xv,
                'yv': yv,
                'zv': zv,
                'I': i_val,
                'J': j_val,
                'F': f_val,
                'S': current_s,
                'diametro_utensile': block_diametro,
                'VOL_TOT': current_vol_tot,
                'code': code_line,
                'R': R_val,
                'theta': theta_val,
                'dist_lineare': dist_lin,
                'dist_circolare': dist_circ,
                'dist': dist,
                't_ideal': t_ideal,
            }

            records.append(record)
            prev_x, prev_y, prev_z = cur_x, cur_y, cur_z

    df = pd.DataFrame(
        records,
        columns=[
            'esecuzione_n', 'esecuzione_nome',
            'N', 'movimento',
            'X', 'Y', 'Z', 'I', 'J',
            'delta_x', 'delta_y', 'delta_z',
            'xv', 'yv', 'zv',
            'R',
            'F', 'S',
            'diametro_utensile', 'VOL_TOT',
            'code',
            'theta', 'dist_lineare', 'dist_circolare', 'dist',
            't_ideal',
        ],
    )

       # ------------------------------------------------------------------
    # ESTENSIONE IN STILE compute_insights (VERSIONE IDEALE):
    # - usa t_ideal come tempo
    # - NON usa "repetition"
    # - tutti i campi hanno prefisso ideal_ per non scontrarsi con i reali
    # ------------------------------------------------------------------
    ESEC_COL = "esecuzione_n"
    FEED_COL = "F"
    VOL_COL = "VOL_TOT"
    TIME_COL = "t_ideal"

    # Controllo colonne minime
    for col in (ESEC_COL, FEED_COL, VOL_COL, TIME_COL):
        if col not in df.columns:
            raise ValueError(f"Manca la colonna '{col}' nel DataFrame del parser.")

    df[FEED_COL] = pd.to_numeric(df[FEED_COL], errors="coerce")
    df[VOL_COL] = pd.to_numeric(df[VOL_COL], errors="coerce")
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")

    # Feed distinti
    unique_F = sorted(df[FEED_COL].dropna().unique().tolist())

    # ideal_count_block_F*
    for idx_f, _ in enumerate(unique_F):
        df[f"ideal_count_block_F{idx_f}"] = 0.0

    for _, block in df.groupby(ESEC_COL, sort=False):
        for idx_f, f_val in enumerate(unique_F):
            col = f"ideal_count_block_F{idx_f}"
            cond = (block[FEED_COL] == f_val)

            # niente repetition: solo 0/1
            incr = cond.astype(int)
            df.loc[block.index, col] = incr.cumsum()

    # ideal_T*
    for idx_f, _ in enumerate(unique_F):
        df[f"ideal_T{idx_f}"] = 0.0

    for _, block in df.groupby(ESEC_COL, sort=False):
        idx = block.index
        time_vals = block[TIME_COL].dropna()
        T_tot = float(time_vals.iloc[-1]) if not time_vals.empty else 0.0

        final_counts = [
            float(block[f"ideal_count_block_F{idx_f}"].iloc[-1])
            for idx_f in range(len(unique_F))
        ]
        count_tot = sum(final_counts)

        if T_tot <= 0 or count_tot <= 0:
            for idx_f in range(len(unique_F)):
                df.loc[idx, f"ideal_T{idx_f}"] = 0.0
        else:
            for idx_f in range(len(unique_F)):
                Ti = T_tot * (final_counts[idx_f] / count_tot)
                df.loc[idx, f"ideal_T{idx_f}"] = Ti

    # ideal_MRR* per feed > 0
    active_indices = [i for i, f in enumerate(unique_F) if f > 0]
    for idx_f in range(len(unique_F)):
        df[f"ideal_MRR{idx_f}"] = 0.0

    for _, block in df.groupby(ESEC_COL, sort=False):
        idx = block.index
        vol_vals = block[VOL_COL].dropna()
        vol = float(vol_vals.iloc[0]) if not vol_vals.empty else 0.0

        T_vec = [float(block[f"ideal_T{idx_f}"].iloc[0]) for idx_f in range(len(unique_F))]
        denom = sum(T_vec[i] * unique_F[i] for i in active_indices)

        k = vol / denom if denom > 0 and vol > 0 else 0.0

        for idx_f, f_val in enumerate(unique_F):
            mrr_val = k * f_val if (f_val > 0 and k > 0) else 0.0
            df.loc[idx, f"ideal_MRR{idx_f}"] = mrr_val

    # ideal_MRR “attivo” sulla riga corrente in base al feed
    df["ideal_MRR"] = 0.0
    feed_to_idx = {f: i for i, f in enumerate(unique_F)}

    for _, block in df.groupby(ESEC_COL, sort=False):
        for i in block.index:
            f_val = df.at[i, FEED_COL]
            if f_val in feed_to_idx:
                idx_f = feed_to_idx[f_val]
                df.at[i, "ideal_MRR"] = df.at[i, f"ideal_MRR{idx_f}"]
            else:
                df.at[i, "ideal_MRR"] = 0.0

     # ----------------------------------------------------------
    # Ricostruisco il tempo di segmento da t_ideal (cumulata per blocco)
    # e poi faccio la cumulata TOTALE su tutto il file
    # ----------------------------------------------------------
    df["ideal_seg_time"] = 0.0

    for _, block in df.groupby(ESEC_COL, sort=False):
        idx = block.index
        t_vals = block[TIME_COL].astype(float)
        # tempo del segmento = differenza tra t_ideal di righe consecutive
        # la prima riga del blocco ha seg_time = t_ideal stessa
        seg = t_vals.diff().fillna(t_vals)
        df.loc[idx, "ideal_seg_time"] = seg

    # Cumulata globale (non per blocco)
    df["t_ideal_cumul"] = df["ideal_seg_time"].cumsum()

    return df
