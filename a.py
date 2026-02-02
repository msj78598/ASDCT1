# -*- coding: utf-8 -*-
"""
Agricultural Load-Based Loss Detection System (Streamlit)
تحليل أحمال/سلوك كهربائي لعدادات يُفترض أنها مرتبطة بحقول زراعية فعّالة.

منهجية التصنيف (بدون علاقة بالبريكر):
1) Confirmed Loss:
   - وجود جهد قريب من الصفر على أي فازة (V≈0) مع وجود جهد معتبر في فازة أخرى (حتى لو بدون تيار).
   - وجود تيار مع جهد قريب من الصفر على نفس الفازة.
   - فازة تيارها ≈0 بينما فازات أخرى عليها حمل واضح.
   - عدم اتزان تيار شديد بين الفازات (i_imb).

2) Suspected High Loss:
   - جهد موجود + حمل/تيار = صفر (No-Load)
   - (اختياري) consumption = 0 إذا كان موجودًا بالملف (كمؤشر داعم)

3) Suspected Medium Loss:
   - جهد موجود + حمل منخفض جدًا (Very Low Load)

4) Normal:
   - لا توجد مؤشرات غير منطقية ضمن الحدود.

تشغيل:
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

# Optional consumption column (support only)
POSSIBLE_CONS_COLS = ["consumption", "Consumption", "consumption_value", "kwh", "KWH", "kWh", "Energy", "energy"]

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

def find_consumption_col(df: pd.DataFrame):
    for c in POSSIBLE_CONS_COLS:
        if c in df.columns:
            return c
    return None

# ===================== Load models (optional) =====================
@lru_cache(maxsize=1)
def load_models():
    models = {}
    if PATH_SCALER.exists() and PATH_IF.exists():
        models["scaler"] = joblib.load(PATH_SCALER)
        models["if"] = joblib.load(PATH_IF)
        if PATH_SVM.exists():
            try:
                models["svm"] = joblib.load(PATH_SVM)
            except Exception:
                models["svm"] = None
        else:
            models["svm"] = None
    else:
        models["scaler"] = None
        models["if"] = None
        models["svm"] = None
    return models

def model_decision_flags(Xs, model_if, model_svm=None, thr_if=0.0, thr_svm=0.0, use_or=True):
    """اختياري: model_flag للمراجعة فقط."""
    df_if = model_if.decision_function(Xs)
    anom_if = (df_if < thr_if).astype(int)

    df_svm = None
    anom_svm = None
    if model_svm is not None:
        df_svm = model_svm.decision_function(Xs)
        anom_svm = (df_svm < thr_svm).astype(int)
        flags = ((anom_if == 1) | (anom_svm == 1)).astype(int) if use_or else ((anom_if == 1) & (anom_svm == 1)).astype(int)
        score_ens = (-0.5 * df_if) + (-0.5 * df_svm)
    else:
        flags = anom_if
        score_ens = -df_if

    return df_if, df_svm, score_ens, flags, anom_if, anom_svm

# ===================== Feature engineering =====================
def compute_signal_features(df: pd.DataFrame, v_eps: float = 1e-6, i_eps: float = 1e-6, r_eps: float = 1e-6,
                            i_near_zero_thr: float = 0.05) -> pd.DataFrame:
    out = df.copy()
    V = out[["V1", "V2", "V3"]].astype(float)
    A = out[["A1", "A2", "A3"]].astype(float)

    out["V_mean"] = V.mean(axis=1)
    out["V_min"]  = V.min(axis=1)
    out["V_max"]  = V.max(axis=1)

    out["I_mean"] = A.mean(axis=1)
    out["I_sum"]  = A.sum(axis=1)
    out["I_max"]  = A.abs().max(axis=1)

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

# ===================== Rules-based classification =====================
def apply_load_based_rules(
    df: pd.DataFrame,
    # Voltage
    v_zero_pct: float = 0.10,
    v_zero_abs_max: float = 15.0,       # سقف مطلق لتحديد V≈0
    v_present_abs: float = 50.0,        # اعتبر الجهد "موجود" إذا V_max >= هذا
    v_other_present_abs: float = 50.0,  # لتأكيد "فازة أخرى فيها جهد"

    # Current
    i_near_zero_thr: float = 0.05,
    i_phase_load_thr: float = 1.0,
    i_sum_no_load_thr: float = 0.15,
    i_sum_low_load_thr: float = 0.80,

    # Imbalance
    i_imb_confirm_thr: float = 1.50,

    # Consumption support (optional)
    use_consumption_if_available: bool = True,
    consumption_zero_thr: float = 0.0,
):
    out = df.copy()
    V = out[["V1", "V2", "V3"]].astype(float)
    A = out[["A1", "A2", "A3"]].astype(float)

    # Optional consumption
    cons_col = find_consumption_col(out) if use_consumption_if_available else None
    if cons_col is not None:
        out[cons_col] = pd.to_numeric(out[cons_col], errors="coerce")
        out["consumption_value"] = out[cons_col]
    else:
        out["consumption_value"] = np.nan

    # Voltage present?
    voltage_present = (out["V_max"].abs().fillna(0) >= v_present_abs)

    # Near-zero voltage threshold per-row: min(pct * |V_mean|, abs_cap)
    v_mean_abs = out["V_mean"].abs().fillna(0.0)
    v_zero_thr = np.minimum(v_zero_pct * v_mean_abs, float(v_zero_abs_max))

    v1_zero = out["V1"].abs().fillna(0) <= v_zero_thr
    v2_zero = out["V2"].abs().fillna(0) <= v_zero_thr
    v3_zero = out["V3"].abs().fillna(0) <= v_zero_thr

    v1_other_present = (out["V2"].abs().fillna(0) >= v_other_present_abs) | (out["V3"].abs().fillna(0) >= v_other_present_abs)
    v2_other_present = (out["V1"].abs().fillna(0) >= v_other_present_abs) | (out["V3"].abs().fillna(0) >= v_other_present_abs)
    v3_other_present = (out["V1"].abs().fillna(0) >= v_other_present_abs) | (out["V2"].abs().fillna(0) >= v_other_present_abs)

    # Current levels
    no_load = (out["I_sum"].abs().fillna(0) <= i_sum_no_load_thr) & (out["I_max"].abs().fillna(0) <= max(i_sum_no_load_thr, i_near_zero_thr))
    low_load = (~no_load) & (out["I_sum"].abs().fillna(0) <= i_sum_low_load_thr)

    # ---------------- Confirmed ----------------
    # C0: V≈0 on any phase + another phase has voltage (even if current is zero)  ✅ key change
    c0_v_zero_any = voltage_present & (
        (v1_zero & v1_other_present) |
        (v2_zero & v2_other_present) |
        (v3_zero & v3_other_present)
    )

    # C1: V≈0 with current on same phase
    c1_v0_with_i = (
        (v1_zero & (out["A1"].abs().fillna(0) >= i_phase_load_thr)) |
        (v2_zero & (out["A2"].abs().fillna(0) >= i_phase_load_thr)) |
        (v3_zero & (out["A3"].abs().fillna(0) >= i_phase_load_thr))
    ) & voltage_present

    # C2: one phase current ~0 while another is significantly loaded
    c2_i_near_zero_with_load = voltage_present & (
        ((out["A1"].abs().fillna(0) <= i_near_zero_thr) & ((out["A2"].abs().fillna(0) >= i_phase_load_thr) | (out["A3"].abs().fillna(0) >= i_phase_load_thr))) |
        ((out["A2"].abs().fillna(0) <= i_near_zero_thr) & ((out["A1"].abs().fillna(0) >= i_phase_load_thr) | (out["A3"].abs().fillna(0) >= i_phase_load_thr))) |
        ((out["A3"].abs().fillna(0) <= i_near_zero_thr) & ((out["A1"].abs().fillna(0) >= i_phase_load_thr) | (out["A2"].abs().fillna(0) >= i_phase_load_thr)))
    )

    # C3: severe current imbalance
    c3_extreme_iimb = voltage_present & (out["i_imb"].fillna(0) >= i_imb_confirm_thr) & (out["I_max"].fillna(0) >= i_phase_load_thr)

    confirmed = c0_v_zero_any | c1_v0_with_i | c2_i_near_zero_with_load | c3_extreme_iimb

    out["reason_confirm_v0_any_phase"] = c0_v_zero_any.astype(int)
    out["reason_confirm_v0_with_i"] = c1_v0_with_i.astype(int)
    out["reason_confirm_i_near_zero"] = c2_i_near_zero_with_load.astype(int)
    out["reason_confirm_extreme_iimb"] = c3_extreme_iimb.astype(int)

    # ---------------- Suspected High / Medium ----------------
    cons_zero = pd.Series(False, index=out.index)
    if cons_col is not None:
        cons_zero = out["consumption_value"].fillna(np.inf) <= consumption_zero_thr

    suspected_high = (~confirmed) & voltage_present & (no_load | cons_zero)
    out["reason_suspected_high_no_load"] = ((~confirmed) & voltage_present & no_load).astype(int)
    out["reason_suspected_high_no_consumption"] = ((~confirmed) & voltage_present & cons_zero).astype(int)

    suspected_medium = (~confirmed) & (~suspected_high) & voltage_present & low_load
    out["reason_suspected_medium_low_load"] = suspected_medium.astype(int)

    # ---------------- Final label ----------------
    out["final_label"] = np.select(
        [confirmed, suspected_high, suspected_medium],
        ["Confirmed Loss", "Suspected High Loss", "Suspected Medium Loss"],
        default="Normal"
    )

    # Primary reason
    def primary_reason_row(r):
        if r["final_label"] == "Confirmed Loss":
            if r.get("reason_confirm_v0_any_phase", 0) == 1:
                return "Voltage near-zero on a phase while another phase has voltage (illogical)"
            if r.get("reason_confirm_v0_with_i", 0) == 1:
                return "Current present with near-zero voltage on same phase"
            if r.get("reason_confirm_i_near_zero", 0) == 1:
                return "Zero/near-zero current on one phase while other phases carry load"
            if r.get("reason_confirm_extreme_iimb", 0) == 1:
                return "Severe current imbalance between phases"
            return "Confirmed electrical contradiction"
        if r["final_label"] == "Suspected High Loss":
            if r.get("reason_suspected_high_no_consumption", 0) == 1:
                return "No-load with voltage present + zero consumption (support)"
            return "No-load while voltage is present"
        if r["final_label"] == "Suspected Medium Loss":
            return "Very low load while voltage is present"
        return "No abnormal electrical behavior"

    out["primary_reason"] = out.apply(primary_reason_row, axis=1)

    # Priority sort key
    pr = {"Confirmed Loss": 0, "Suspected High Loss": 1, "Suspected Medium Loss": 2, "Normal": 3}
    out["prio"] = out["final_label"].map(pr).fillna(9).astype(int)

    return out

# ===================== UI config =====================
st.set_page_config(page_title="Agricultural Load-Based Loss Detection", layout="wide")

st.title("Agricultural Load-Based Loss Detection System")
st.caption("تحليل جهد/تيار لعدادات زراعية مفترض أنها تخدم حقول فعّالة — كشف سلوك غير منطقي (بدون علاقة بالبريكر).")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ الإعدادات")

    preset = st.selectbox(
        "Preset (جاهز)",
        ["متوازن (Recommended)", "حساس (لا يفوّت)", "صارم (تقليل الإنذارات)"],
        index=0
    )

    if preset == "حساس (لا يفوّت)":
        defaults = dict(
            v_zero_pct=0.12, v_zero_abs_max=20.0, v_present_abs=40.0, v_other_present_abs=40.0,
            i_near_zero_thr=0.05, i_phase_load_thr=0.8, i_sum_no_load_thr=0.20, i_sum_low_load_thr=1.00,
            i_imb_confirm_thr=1.20, use_consumption_if_available=True,
        )
    elif preset == "صارم (تقليل الإنذارات)":
        defaults = dict(
            v_zero_pct=0.08, v_zero_abs_max=12.0, v_present_abs=60.0, v_other_present_abs=60.0,
            i_near_zero_thr=0.05, i_phase_load_thr=1.5, i_sum_no_load_thr=0.12, i_sum_low_load_thr=0.60,
            i_imb_confirm_thr=1.80, use_consumption_if_available=True,
        )
    else:
        defaults = dict(
            v_zero_pct=0.10, v_zero_abs_max=15.0, v_present_abs=50.0, v_other_present_abs=50.0,
            i_near_zero_thr=0.05, i_phase_load_thr=1.0, i_sum_no_load_thr=0.15, i_sum_low_load_thr=0.80,
            i_imb_confirm_thr=1.50, use_consumption_if_available=True,
        )

    with st.expander("⚡ إعدادات الجهد", expanded=True):
        v_zero_pct = st.slider("نسبة V لتحديد V≈0 (Vphase < pct*V_mean)", 0.02, 0.30, float(defaults["v_zero_pct"]), 0.01)
        v_zero_abs_max = st.slider("سقف مطلق لـ V≈0 (V ≤)", 1.0, 50.0, float(defaults["v_zero_abs_max"]), 1.0)
        v_present_abs = st.slider("حد أدنى لاعتبار الجهد موجود (V_max ≥)", 10.0, 200.0, float(defaults["v_present_abs"]), 5.0)
        v_other_present_abs = st.slider("حد أدنى لفازة أخرى لتأكيد V≈0 (V ≥)", 10.0, 200.0, float(defaults["v_other_present_abs"]), 5.0)

    with st.expander("🔌 إعدادات التيار", expanded=True):
        i_near_zero_thr = st.slider("تيار يعتبر ≈0 (A)", 0.0, 1.0, float(defaults["i_near_zero_thr"]), 0.01)
        i_phase_load_thr = st.slider("تيار يدل على حمل واضح (A)", 0.1, 20.0, float(defaults["i_phase_load_thr"]), 0.1)
        i_sum_no_load_thr = st.slider("حد No-Load لمجموع التيار (A)", 0.0, 5.0, float(defaults["i_sum_no_load_thr"]), 0.01)
        i_sum_low_load_thr = st.slider("حد Very Low Load لمجموع التيار (A)", 0.1, 10.0, float(defaults["i_sum_low_load_thr"]), 0.05)

    with st.expander("📐 عدم الاتزان", expanded=True):
        i_imb_confirm_thr = st.slider("عدم اتزان تيار شديد للتأكيد (i_imb ≥)", 0.50, 3.00, float(defaults["i_imb_confirm_thr"]), 0.05)

    with st.expander("🧾 الاستهلاك (اختياري)", expanded=False):
        use_consumption_if_available = st.toggle("استخدم عمود الاستهلاك إذا كان موجودًا", value=bool(defaults["use_consumption_if_available"]))
        st.caption("إذا كان ملفك يحتوي consumption/Consumption/... فسيُستخدم لدعم حالات No-Consumption ضمن Suspected High.")

    st.markdown("---")
    st.download_button(
        "⬇️ تنزيل قالب Excel (Template)",
        make_template_excel(),
        file_name="agri_load_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ---------------- Main tabs ----------------
tab_infer, tab_help = st.tabs(["📈 تحليل ملف", "ℹ️ المساعدة"])

with tab_infer:
    st.subheader("رفع ملف Excel للتحليل")
    uploaded = st.file_uploader(
        "الأعمدة المطلوبة: Meter Number, V1, V2, V3, A1, A2, A3 (اختياري: consumption)",
        type=["xlsx"]
    )

    if uploaded is None:
        st.info("ارفع ملف Excel للبدء.")
    else:
        try:
            df = pd.read_excel(uploaded)
            df.columns = [c.strip() for c in df.columns]
            validate_columns(df, [ID_COL] + FEATURE_COLS)

            cons_col = find_consumption_col(df)
            cols_to_num = FEATURE_COLS + ([cons_col] if cons_col is not None else [])
            df = safe_to_numeric(df, cols_to_num)

            df_infer = df.dropna(subset=FEATURE_COLS).copy().reset_index(drop=True)
            if df_infer.empty:
                st.warning("لا يوجد صفوف صالحة بعد التنظيف (قيم ناقصة/غير رقمية).")
                st.stop()

            detailed = df_infer[[ID_COL] + FEATURE_COLS].copy()
            if cons_col is not None:
                detailed[cons_col] = df_infer[cons_col].values

            # Optional model flag (reference only)
            models = load_models()
            scaler = models.get("scaler", None)
            model_if = models.get("if", None)
            model_svm = models.get("svm", None)

            if scaler is not None and model_if is not None:
                X = df_infer[FEATURE_COLS].values
                Xs = scaler.transform(X)
                df_if, df_svm, score_ens, flags, anom_if, anom_svm = model_decision_flags(Xs, model_if, model_svm=model_svm)
                detailed["model_flag"] = flags
                detailed["df_if"] = df_if
                detailed["score_ensemble"] = score_ens
            else:
                detailed["model_flag"] = 0

            detailed = compute_signal_features(detailed, i_near_zero_thr=float(i_near_zero_thr))

            detailed = apply_load_based_rules(
                detailed,
                v_zero_pct=float(v_zero_pct),
                v_zero_abs_max=float(v_zero_abs_max),
                v_present_abs=float(v_present_abs),
                v_other_present_abs=float(v_other_present_abs),
                i_near_zero_thr=float(i_near_zero_thr),
                i_phase_load_thr=float(i_phase_load_thr),
                i_sum_no_load_thr=float(i_sum_no_load_thr),
                i_sum_low_load_thr=float(i_sum_low_load_thr),
                i_imb_confirm_thr=float(i_imb_confirm_thr),
                use_consumption_if_available=bool(use_consumption_if_available),
            )

            # KPIs
            c_conf = int((detailed["final_label"] == "Confirmed Loss").sum())
            c_high = int((detailed["final_label"] == "Suspected High Loss").sum())
            c_med  = int((detailed["final_label"] == "Suspected Medium Loss").sum())
            c_norm = int((detailed["final_label"] == "Normal").sum())

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Confirmed", c_conf)
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
                max_Vmax=("V_max", "max"),
                min_Vmin=("V_min", "min"),
                max_Imax=("I_max", "max"),
                max_Isum=("I_sum", "max"),
                max_iimb=("i_imb", "max"),
                max_vimb=("v_imb", "max"),
                model_flags=("model_flag", "sum"),
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
            pr = {"Confirmed Loss": 0, "Suspected High Loss": 1, "Suspected Medium Loss": 2, "Normal": 3}
            summary["prio"] = summary["meter_final_label"].map(pr).fillna(9).astype(int)

            summary = summary.sort_values(
                by=["prio", "confirmed", "suspected_high", "suspected_medium", "max_iimb", "max_Imax"],
                ascending=[True, False, False, False, False, False]
            ).drop(columns=["prio"]).reset_index(drop=True)

            # Tabs
            t_all, t_conf, t_high, t_med, t_norm = st.tabs(
                ["📄 الكل", "✅ Confirmed", "🟥 Suspected High", "🟧 Suspected Medium", "🟩 Normal"]
            )

            with t_all:
                render_table(summary, "ملخص العدادات (Meter Summary)", "summary_agri_load.xlsx", height=380)
                render_table(detailed.drop(columns=["prio"], errors="ignore"), "النتائج التفصيلية (Detailed)", "detailed_agri_load.xlsx", height=450)

            with t_conf:
                render_table(detailed[detailed["final_label"] == "Confirmed Loss"].copy(),
                             "Confirmed Loss - Detailed", "confirmed_agri_load.xlsx", height=520)

            with t_high:
                render_table(detailed[detailed["final_label"] == "Suspected High Loss"].copy(),
                             "Suspected High Loss - Detailed", "suspected_high_agri_load.xlsx", height=520)

            with t_med:
                render_table(detailed[detailed["final_label"] == "Suspected Medium Loss"].copy(),
                             "Suspected Medium Loss - Detailed", "suspected_medium_agri_load.xlsx", height=520)

            with t_norm:
                render_table(detailed[detailed["final_label"] == "Normal"].copy(),
                             "Normal - Detailed", "normal_agri_load.xlsx", height=520)

            # Reasons dashboard
            st.markdown("---")
            st.subheader("لوحة أسباب القرار (Reasons)")
            reason_cols = [
                "reason_confirm_v0_any_phase",
                "reason_confirm_v0_with_i",
                "reason_confirm_i_near_zero",
                "reason_confirm_extreme_iimb",
                "reason_suspected_high_no_load",
                "reason_suspected_high_no_consumption",
                "reason_suspected_medium_low_load",
            ]
            existing = [c for c in reason_cols if c in detailed.columns]
            reasons_sum = detailed[existing].sum().sort_values(ascending=False).reset_index()
            reasons_sum.columns = ["Reason", "Count"]
            st.dataframe(reasons_sum, use_container_width=True)

        except Exception as e:
            st.exception(e)

with tab_help:
    st.markdown("""
### ✅ لماذا حالة (V=0 على فازة) تعتبر Confirmed حتى لو بدون تيار؟
لأننا نفترض أن القائمة تخص **حقول زراعية فعّالة**.
وجود فازة جهدها **قريب من الصفر** مع وجود جهد معتبر في فازة أخرى يدل غالبًا على:
- فصل/قص VT أو خلل توصيل
- عبث/تلاعب
- خلل قياس أو توصيلات غير منطقية

### التصنيفات
- **Confirmed Loss:** تناقض كهربائي واضح (ومنها V≈0 على أي فازة).
- **Suspected High Loss:** جهد موجود + No-Load / No-Consumption (قد تكون تغذية بديلة).
- **Suspected Medium Loss:** جهد موجود + Very Low Load (يحتاج تحقق زمني/سياقي).
- **Normal:** لا توجد مؤشرات ضمن العتبات.
""")

st.markdown("---")
st.markdown("👨‍💻 Developed by: Mashhour Alabbas | 2026")
