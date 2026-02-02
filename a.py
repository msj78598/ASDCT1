# -*- coding: utf-8 -*-
"""
Agricultural Load-Based Loss Detection System (Streamlit)

Purpose:
Analyze voltage/current readings for meters assumed to be connected to active agricultural fields.
Any illogical electrical behavior is treated as a potential non-technical loss indicator.

Classification:
1) Confirmed Loss:
   - Current present with near-zero voltage on the same phase
   - Zero/near-zero current on one phase while other phases carry significant load
   - Severe current imbalance between phases (i_imb high)

2) Suspected High Loss:
   - Zero load while voltage is present (field assumed active, but may be fed by another meter)

3) Suspected Medium Loss:
   - Very low load while voltage is present (seasonal/limited operation possible)

4) Normal:
   - No abnormal electrical behavior detected

Run:
  pip install -r requirements.txt
  streamlit run app.py
"""

from pathlib import Path
from functools import lru_cache
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ===================== Paths (optional models) =====================
PROJECT_DIR = Path(__file__).resolve().parent
PATH_SCALER = PROJECT_DIR / "scaler.joblib"
PATH_IF     = PROJECT_DIR / "isolation_forest.joblib"
PATH_SVM    = PROJECT_DIR / "ocsvm.joblib"  # optional

# ===================== Data columns =====================
ID_COL = "Meter Number"
FEATURE_COLS = ["V1", "V2", "V3", "A1", "A2", "A3"]

# Optional consumption column (if present, used as a supporting indicator only)
POSSIBLE_CONS_COLS = ["consumption", "Consumption", "kwh", "KWH", "kWh", "Energy", "energy"]

# ===================== Helpers =====================
def validate_columns(df: pd.DataFrame, required_cols):
    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        raise ValueError(f"الأعمدة الناقصة: {miss}")

def safe_to_numeric(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def excel_bytes(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        df.to_excel(w, sheet_name="Sheet1", index=False)
    return out.getvalue()

def render_table(df, title, download_name, height=450):
    st.subheader(title)
    st.dataframe(df, use_container_width=True, height=height)
    st.download_button(
        "⬇️ تنزيل (Excel)",
        excel_bytes(df),
        file_name=download_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

def make_template_excel() -> bytes:
    tmp = pd.DataFrame(columns=[ID_COL] + FEATURE_COLS)
    return excel_bytes(tmp)

def clip01(x):
    return np.clip(x, 0, 1)

def find_consumption_col(df: pd.DataFrame):
    for c in POSSIBLE_CONS_COLS:
        if c in df.columns:
            return c
    return None

# ===================== Load models (optional) =====================
@lru_cache(maxsize=1)
def load_models_if_available():
    """
    Optional: Load scaler + IF + (optional) SVM if present.
    If not present, return None and system still works using rules-only classification.
    """
    if not PATH_SCALER.exists() or not PATH_IF.exists():
        return None

    models = {}
    models["scaler"] = joblib.load(PATH_SCALER)
    models["if"] = joblib.load(PATH_IF)

    if PATH_SVM.exists():
        try:
            models["svm"] = joblib.load(PATH_SVM)
        except Exception:
            models["svm"] = None
    else:
        models["svm"] = None

    return models

def model_decision_flags(Xs, model_if, model_svm=None, thr_if=0.0, thr_svm=0.0, use_or=True):
    """
    Optional only.
    decision_function >= 0 غالبًا طبيعي
    decision_function < 0 غالبًا شاذ
    """
    df_if = model_if.decision_function(Xs)
    anom_if = (df_if < thr_if).astype(int)

    df_svm = None
    anom_svm = None
    if model_svm is not None:
        df_svm = model_svm.decision_function(Xs)
        anom_svm = (df_svm < thr_svm).astype(int)
        if use_or:
            flags = ((anom_if == 1) | (anom_svm == 1)).astype(int)
        else:
            flags = ((anom_if == 1) & (anom_svm == 1)).astype(int)
    else:
        flags = anom_if

    if df_svm is not None:
        score_ens = (-0.5 * df_if) + (-0.5 * df_svm)
    else:
        score_ens = -df_if

    return df_if, df_svm, score_ens, flags, anom_if, anom_svm

# ===================== Feature engineering =====================
def compute_signal_features(
    df: pd.DataFrame,
    v_eps: float = 1e-6,
    i_eps: float = 1e-6,
    r_eps: float = 1e-6,
    i_near_zero_thr: float = 0.05,
) -> pd.DataFrame:
    """
    Stable electrical indicators:
    - V_mean, V_min, V_max
    - I_sum, I_max
    - v_imb, i_imb
    - weak_ratio
    - r_spread
    - near_zero_phase_present (any phase current near zero)
    """
    V = df[["V1", "V2", "V3"]].astype(float)
    A = df[["A1", "A2", "A3"]].astype(float)

    out = df.copy()

    out["V_mean"] = V.mean(axis=1)
    out["V_min"] = V.min(axis=1)
    out["V_max"] = V.max(axis=1)

    out["I_mean"] = A.mean(axis=1)
    out["I_sum"] = A.sum(axis=1)
    out["I_max"] = A.abs().max(axis=1)

    out["v_imb"] = (V.max(axis=1) - V.min(axis=1)) / out["V_mean"].abs().clip(lower=v_eps)
    out["i_imb"] = (A.max(axis=1) - A.min(axis=1)) / out["I_mean"].abs().clip(lower=i_eps)

    out["weak_ratio"] = (A.min(axis=1) / A.max(axis=1).abs().clip(lower=i_eps)).clip(0, 1)

    R = pd.DataFrame({
        "r1": A["A1"] / V["V1"].abs().clip(lower=v_eps),
        "r2": A["A2"] / V["V2"].abs().clip(lower=v_eps),
        "r3": A["A3"] / V["V3"].abs().clip(lower=v_eps),
    }).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    r_min = R.min(axis=1).clip(lower=r_eps)
    out["r_spread"] = (R.max(axis=1) / r_min).clip(lower=1.0)

    out["near_zero_phase_present"] = (A.abs().le(i_near_zero_thr).sum(axis=1) >= 1).astype(int)

    return out

# ===================== Agricultural Load-Based Classification =====================
def apply_agri_load_rules(
    df: pd.DataFrame,
    # Voltage presence / near-zero definition
    v_present_min: float = 50.0,        # consider "voltage present" if V_mean >= this
    v_zero_pct: float = 0.10,           # near-zero threshold = v_zero_pct * V_mean
    v_zero_abs_max: float = 15.0,       # OR absolute near-zero ceiling (extra safety)

    # Current thresholds
    i_significant: float = 2.0,         # current indicating load
    i_near_zero_thr: float = 0.05,      # current considered (near) zero

    # Confirmed imbalance thresholds
    i_imb_confirm_thr: float = 1.80,    # severe i_imb

    # No-Load / Low-Load thresholds for agricultural list
    no_load_sum_thr: float = 0.20,      # I_sum <= this => No-Load
    no_load_max_thr: float = 0.10,      # I_max <= this => No-Load
    low_load_sum_thr: float = 1.00,     # I_sum <= this (but > no_load) => Low-Load
    low_load_max_thr: float = 0.50,     # I_max <= this (but > no_load) => Low-Load
):
    out = df.copy()
    V = out[["V1", "V2", "V3"]].astype(float)
    A = out[["A1", "A2", "A3"]].astype(float)

    # Voltage present (system assumption: field is active, so V should exist)
    v_present = (out["V_mean"].fillna(0) >= v_present_min)

    # Near-zero voltage threshold row-wise
    v_zero_thr_row = (v_zero_pct * out["V_mean"].abs()).fillna(0.0)
    v_zero_thr_row = np.minimum(v_zero_thr_row, v_zero_abs_max)  # cap it

    # ----- Confirmed Loss rules -----

    # C1: I present with near-zero V on same phase
    c1_v0_with_i = (
        ((out["V1"] <= v_zero_thr_row) & (out["A1"].abs() >= i_significant)) |
        ((out["V2"] <= v_zero_thr_row) & (out["A2"].abs() >= i_significant)) |
        ((out["V3"] <= v_zero_thr_row) & (out["A3"].abs() >= i_significant))
    ) & v_present

    # C2: one phase current ~0 while other phases carry significant load
    near_zero_phase = (A.abs().le(i_near_zero_thr)).sum(axis=1) >= 1
    other_has_load = (A.abs().ge(i_significant)).sum(axis=1) >= 1
    c2_i0_one_phase_others_load = v_present & near_zero_phase & other_has_load

    # C3: severe current imbalance
    c3_severe_iimb = v_present & (out["i_imb"].fillna(0) >= i_imb_confirm_thr) & (out["I_max"].fillna(0) >= i_significant)

    confirmed = (c1_v0_with_i | c2_i0_one_phase_others_load | c3_severe_iimb)

    out["reason_confirm_v0_with_i"] = c1_v0_with_i.astype(int)
    out["reason_confirm_i_near_zero"] = c2_i0_one_phase_others_load.astype(int)
    out["reason_confirm_extreme_iimb"] = c3_severe_iimb.astype(int)

    # ----- Suspected High / Medium -----

    # No-Load (Suspected High): voltage present but essentially no current
    no_load = v_present & (out["I_sum"].fillna(0) <= no_load_sum_thr) & (out["I_max"].fillna(0) <= no_load_max_thr)

    # Low-Load (Suspected Medium): voltage present but very low current (above no-load)
    low_load = (
        v_present &
        (~no_load) &
        (out["I_sum"].fillna(0) <= low_load_sum_thr) &
        (out["I_max"].fillna(0) <= low_load_max_thr)
    )

    # If consumption column exists, add supportive reasons (does not override)
    if "consumption_value" in out.columns:
        # zero consumption support only if value is numeric
        cons_zero = out["consumption_value"].fillna(0) <= 0
    else:
        cons_zero = pd.Series(False, index=out.index)

    # Suspected High: No-load while voltage present (strong suspicion)
    suspected_high = (~confirmed) & no_load
    # Suspected Medium: low-load while voltage present (medium suspicion)
    suspected_medium = (~confirmed) & (~suspected_high) & low_load

    out["reason_suspected_high_no_load"] = suspected_high.astype(int)
    out["reason_suspected_medium_low_load"] = suspected_medium.astype(int)
    out["support_consumption_zero"] = cons_zero.astype(int)

    # ----- Normal -----
    normal = (~confirmed) & (~suspected_high) & (~suspected_medium)

    # Final label
    out["final_label"] = np.select(
        [confirmed, suspected_high, suspected_medium, normal],
        ["Confirmed Loss", "Suspected High Loss", "Suspected Medium Loss", "Normal"],
        default="Normal"
    )

    # Primary reason (single text)
    def primary_reason_row(r):
        if r["final_label"] == "Confirmed Loss":
            if r["reason_confirm_v0_with_i"] == 1:
                return "Current present with near-zero voltage on same phase"
            if r["reason_confirm_i_near_zero"] == 1:
                return "Zero/near-zero current on one phase while other phases carry load"
            if r["reason_confirm_extreme_iimb"] == 1:
                return "Severe current imbalance between phases"
            return "Confirmed electrical contradiction"
        if r["final_label"] == "Suspected High Loss":
            if r["support_consumption_zero"] == 1:
                return "No-load with voltage present + zero consumption (support)"
            return "No-load while voltage is present"
        if r["final_label"] == "Suspected Medium Loss":
            return "Very low load while voltage is present"
        return "No abnormal electrical behavior"

    out["primary_reason"] = out.apply(primary_reason_row, axis=1)

    # Severity score for sorting (higher = more priority)
    sev = (
        0.60 * out["reason_confirm_v0_with_i"].fillna(0) +
        0.45 * out["reason_confirm_i_near_zero"].fillna(0) +
        0.40 * out["reason_confirm_extreme_iimb"].fillna(0) +
        0.30 * out["reason_suspected_high_no_load"].fillna(0) +
        0.20 * out["reason_suspected_medium_low_load"].fillna(0) +
        0.10 * out["support_consumption_zero"].fillna(0)
    )
    out["severity_score"] = sev

    return out

# ===================== UI config =====================
st.set_page_config(page_title="Agricultural Load-Based Loss Detection", layout="wide")

st.title("Agricultural Load-Based Loss Detection System")
st.caption("تحليل كهربائي بحت (V/I) لعدادات يفترض أنها تخدم حقول زراعية نشطة — 4 مستويات تصنيف.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    preset = st.selectbox(
        "Preset",
        ["Balanced (Recommended)", "Sensitive (Catch more)", "Strict (Reduce alerts)"],
        index=0
    )

    if preset == "Sensitive (Catch more)":
        defaults = dict(
            v_present_min=30.0,
            v_zero_pct=0.12,
            v_zero_abs_max=20.0,
            i_significant=1.5,
            i_near_zero_thr=0.08,
            i_imb_confirm_thr=1.50,
            no_load_sum_thr=0.30,
            no_load_max_thr=0.15,
            low_load_sum_thr=1.50,
            low_load_max_thr=0.70,
            use_models=False,
            combine_or=True,
            thr_if=0.0,
            thr_svm=0.0,
        )
    elif preset == "Strict (Reduce alerts)":
        defaults = dict(
            v_present_min=60.0,
            v_zero_pct=0.08,
            v_zero_abs_max=12.0,
            i_significant=2.5,
            i_near_zero_thr=0.05,
            i_imb_confirm_thr=2.00,
            no_load_sum_thr=0.10,
            no_load_max_thr=0.05,
            low_load_sum_thr=0.80,
            low_load_max_thr=0.40,
            use_models=False,
            combine_or=True,
            thr_if=0.0,
            thr_svm=0.0,
        )
    else:
        defaults = dict(
            v_present_min=50.0,
            v_zero_pct=0.10,
            v_zero_abs_max=15.0,
            i_significant=2.0,
            i_near_zero_thr=0.05,
            i_imb_confirm_thr=1.80,
            no_load_sum_thr=0.20,
            no_load_max_thr=0.10,
            low_load_sum_thr=1.00,
            low_load_max_thr=0.50,
            use_models=False,
            combine_or=True,
            thr_if=0.0,
            thr_svm=0.0,
        )

    with st.expander("🔌 Electrical thresholds", expanded=True):
        v_present_min = st.number_input("Voltage present if V_mean ≥", value=float(defaults["v_present_min"]), step=5.0)
        v_zero_pct = st.slider("Near-zero voltage ratio (V ≤ pct * V_mean)", 0.02, 0.30, float(defaults["v_zero_pct"]), 0.01)
        v_zero_abs_max = st.number_input("Near-zero voltage absolute cap (V ≤)", value=float(defaults["v_zero_abs_max"]), step=1.0)

        i_significant = st.number_input("Significant current threshold (A)", value=float(defaults["i_significant"]), step=0.5)
        i_near_zero_thr = st.number_input("Near-zero current threshold (A)", value=float(defaults["i_near_zero_thr"]), step=0.01)

        i_imb_confirm_thr = st.number_input("Severe current imbalance i_imb ≥", value=float(defaults["i_imb_confirm_thr"]), step=0.1)

    with st.expander("🟧 Suspected High / Medium thresholds", expanded=True):
        no_load_sum_thr = st.number_input("No-load if I_sum ≤", value=float(defaults["no_load_sum_thr"]), step=0.05)
        no_load_max_thr = st.number_input("No-load if I_max ≤", value=float(defaults["no_load_max_thr"]), step=0.05)

        low_load_sum_thr = st.number_input("Low-load if I_sum ≤", value=float(defaults["low_load_sum_thr"]), step=0.10)
        low_load_max_thr = st.number_input("Low-load if I_max ≤", value=float(defaults["low_load_max_thr"]), step=0.10)

    with st.expander("🧠 Optional models (IF/OCSVM)", expanded=False):
        use_models = st.toggle("Enable models (optional)", value=bool(defaults["use_models"]))
        combine_or = st.toggle("Combine IF and OCSVM using OR", value=bool(defaults["combine_or"]))
        thr_if = st.number_input("IF decision threshold (df < thr => anomaly)", value=float(defaults["thr_if"]), step=0.05)
        thr_svm = st.number_input("OCSVM decision threshold (df < thr => anomaly)", value=float(defaults["thr_svm"]), step=0.05)
        st.caption("Models are optional and do not override Confirmed/Suspected rules in this agricultural methodology.")

    st.markdown("---")
    st.download_button(
        "⬇️ Download Excel Template",
        make_template_excel(),
        file_name="agri_load_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ---------------- Main ----------------
tab_infer, tab_help = st.tabs(["📈 Analyze File", "ℹ️ Help"])

with tab_infer:
    st.subheader("Upload Excel for Analysis")
    uploaded = st.file_uploader(
        "Required columns: Meter Number, V1,V2,V3,A1,A2,A3 (consumption optional)",
        type=["xlsx"]
    )

    if uploaded is None:
        st.info("Upload an Excel file to start.")
    else:
        try:
            df = pd.read_excel(uploaded)
            df.columns = [c.strip() for c in df.columns]

            validate_columns(df, [ID_COL] + FEATURE_COLS)

            # Detect optional consumption column
            cons_col = find_consumption_col(df)
            if cons_col is not None:
                df["consumption_value"] = pd.to_numeric(df[cons_col], errors="coerce")
            else:
                df["consumption_value"] = np.nan

            df = safe_to_numeric(df, FEATURE_COLS + ["consumption_value"])
            df_infer = df.dropna(subset=FEATURE_COLS).copy().reset_index(drop=True)

            if df_infer.empty:
                st.warning("No valid rows after cleaning (missing/non-numeric V/I).")
                st.stop()

            # Optional models inference
            if use_models:
                models = load_models_if_available()
                if models is None:
                    st.warning("Models not found (scaler.joblib / isolation_forest.joblib). Continuing rules-only.")
                    use_models_runtime = False
                else:
                    use_models_runtime = True
            else:
                use_models_runtime = False

            detailed = df_infer[[ID_COL] + FEATURE_COLS].copy()
            detailed["consumption_value"] = df_infer["consumption_value"].copy()

            # If models enabled and available: compute flags (for reference only)
            if use_models_runtime:
                scaler = models["scaler"]
                model_if = models["if"]
                model_svm = models.get("svm", None)

                X = df_infer[FEATURE_COLS].values
                Xs = scaler.transform(X)

                df_if, df_svm, score_ens, flags, anom_if, anom_svm = model_decision_flags(
                    Xs, model_if, model_svm=model_svm,
                    thr_if=float(thr_if), thr_svm=float(thr_svm),
                    use_or=bool(combine_or)
                )

                detailed["df_if"] = df_if
                detailed["anom_if"] = anom_if
                if df_svm is not None:
                    detailed["df_svm"] = df_svm
                    detailed["anom_svm"] = anom_svm
                detailed["score_ensemble"] = score_ens
                detailed["model_flag"] = flags
            else:
                detailed["model_flag"] = 0

            # Electrical features
            detailed = compute_signal_features(detailed, i_near_zero_thr=float(i_near_zero_thr))

            # Apply agricultural methodology
            detailed = apply_agri_load_rules(
                detailed,
                v_present_min=float(v_present_min),
                v_zero_pct=float(v_zero_pct),
                v_zero_abs_max=float(v_zero_abs_max),
                i_significant=float(i_significant),
                i_near_zero_thr=float(i_near_zero_thr),
                i_imb_confirm_thr=float(i_imb_confirm_thr),
                no_load_sum_thr=float(no_load_sum_thr),
                no_load_max_thr=float(no_load_max_thr),
                low_load_sum_thr=float(low_load_sum_thr),
                low_load_max_thr=float(low_load_max_thr),
            )

            # KPIs
            c_conf = int((detailed["final_label"] == "Confirmed Loss").sum())
            c_high = int((detailed["final_label"] == "Suspected High Loss").sum())
            c_med  = int((detailed["final_label"] == "Suspected Medium Loss").sum())
            c_norm = int((detailed["final_label"] == "Normal").sum())

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Confirmed Loss", c_conf)
            k2.metric("Suspected High", c_high)
            k3.metric("Suspected Medium", c_med)
            k4.metric("Normal", c_norm)

            st.markdown("---")

            # Summary per meter
            summary = detailed.groupby(ID_COL, as_index=False).agg(
                rows=("final_label", "size"),
                confirmed=("final_label", lambda s: int((s == "Confirmed Loss").sum())),
                suspected_high=("final_label", lambda s: int((s == "Suspected High Loss").sum())),
                suspected_medium=("final_label", lambda s: int((s == "Suspected Medium Loss").sum())),
                normal=("final_label", lambda s: int((s == "Normal").sum())),
                max_Imax=("I_max", "max"),
                max_Isum=("I_sum", "max"),
                max_Vmean=("V_mean", "max"),
                max_iimb=("i_imb", "max"),
                max_severity=("severity_score", "max"),
            )

            def meter_label(row):
                if row["confirmed"] > 0:
                    return "Confirmed Loss"
                if row["suspected_high"] > 0:
                    return "Suspected High Loss"
                if row["suspected_medium"] > 0:
                    return "Suspected Medium Loss"
                return "Normal"

            summary["meter_final_label"] = summary.apply(meter_label, axis=1)

            pr = {
                "Confirmed Loss": 0,
                "Suspected High Loss": 1,
                "Suspected Medium Loss": 2,
                "Normal": 3
            }
            summary["prio"] = summary["meter_final_label"].map(pr).fillna(9).astype(int)

            summary = summary.sort_values(
                by=["prio", "max_severity", "confirmed", "suspected_high", "suspected_medium", "max_iimb", "max_Imax"],
                ascending=[True, False, False, False, False, False, False]
            ).drop(columns=["prio"]).reset_index(drop=True)

            # Tabs
            t_all, t_conf, t_high, t_med, t_norm = st.tabs(
                ["📄 All", "✅ Confirmed", "🟧 Suspected High", "🟨 Suspected Medium", "🟩 Normal"]
            )

            with t_all:
                render_table(summary, "Meter Summary (Prioritized)", "meter_summary_agri_load.xlsx", height=360)
                render_table(detailed, "Detailed Results", "detailed_agri_load.xlsx", height=520)

            with t_conf:
                dfc = detailed[detailed["final_label"] == "Confirmed Loss"].copy()
                render_table(dfc, "Confirmed Loss - Detailed", "confirmed_agri_load.xlsx", height=520)

            with t_high:
                dfh = detailed[detailed["final_label"] == "Suspected High Loss"].copy()
                render_table(dfh, "Suspected High Loss - Detailed", "suspected_high_agri_load.xlsx", height=520)

            with t_med:
                dfm = detailed[detailed["final_label"] == "Suspected Medium Loss"].copy()
                render_table(dfm, "Suspected Medium Loss - Detailed", "suspected_medium_agri_load.xlsx", height=520)

            with t_norm:
                dfn = detailed[detailed["final_label"] == "Normal"].copy()
                render_table(dfn, "Normal - Detailed", "normal_agri_load.xlsx", height=520)

            # Reasons dashboard
            st.markdown("---")
            st.subheader("Reasons Dashboard")
            reason_cols = [
                "reason_confirm_v0_with_i",
                "reason_confirm_i_near_zero",
                "reason_confirm_extreme_iimb",
                "reason_suspected_high_no_load",
                "reason_suspected_medium_low_load",
                "support_consumption_zero",
            ]
            reasons_sum = detailed[reason_cols].sum().sort_values(ascending=False).reset_index()
            reasons_sum.columns = ["Reason", "Count"]
            st.dataframe(reasons_sum, use_container_width=True)

        except Exception as e:
            st.exception(e)

with tab_help:
    st.markdown("""
## Agricultural Load-Based Loss Detection – Methodology

### Principle
This system assumes the meter is associated with an active agricultural field.
Therefore, **illogical electrical behavior** is treated as a potential non-technical loss indicator.

### Classes
- **Confirmed Loss**: clear electrical contradiction (e.g., I with V≈0).
- **Suspected High Loss**: voltage present but no load (could be fed by another source).
- **Suspected Medium Loss**: voltage present but very low load (seasonal/limited operation).
- **Normal**: no abnormal behavior.

### Output
- Final class label
- Primary reason
- Supporting indicators (V_mean, I_sum, I_max, i_imb, …)
- Prioritized list for field inspections
""")

st.markdown("---")
st.markdown("👨‍💻 Developed by: Mashhour Alabbas | 2026")
