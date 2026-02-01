# -*- coding: utf-8 -*-
"""
ASDCT - Streamlit (IF + optional OCSVM) + Smart Fusion (Rules + Models)
- بدون إجبار نسبة شذوذ (NO target_rate)
- يعتمد على decision_function threshold (افتراضي 0) للنماذج
- Confirmed rules تغلب دائمًا
- Suspected rules + Model promotion
- Model Anomaly فقط إذا مدعوم بمؤشرات كهربائية (لتقليل الضجيج)

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

# ===================== Paths =====================
PROJECT_DIR = Path(__file__).resolve().parent
PATH_SCALER = PROJECT_DIR / "scaler.joblib"
PATH_IF     = PROJECT_DIR / "isolation_forest.joblib"
PATH_SVM    = PROJECT_DIR / "ocsvm.joblib"  # optional

# ===================== Data columns =====================
ID_COL = "Meter Number"
FEATURE_COLS = ["V1", "V2", "V3", "A1", "A2", "A3"]

# ===================== Helpers =====================
def validate_columns(df: pd.DataFrame, required_cols):
    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        raise ValueError(f"الأعمدة الناقصة: {miss}")

def safe_to_numeric(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
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

# ===================== Load models =====================
@lru_cache(maxsize=1)
def load_models():
    models = {}
    if not PATH_SCALER.exists():
        raise FileNotFoundError("scaler.joblib غير موجود.")
    if not PATH_IF.exists():
        raise FileNotFoundError("isolation_forest.joblib غير موجود.")

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

# ===================== Feature engineering (robust) =====================
def compute_signal_features(
    df: pd.DataFrame,
    v_eps: float = 1e-6,
    i_eps: float = 1e-6,
    r_eps: float = 1e-6,
    i_near_zero_thr: float = 0.05,
) -> pd.DataFrame:
    """
    مشتقات كهربائية مستقرة:
    - v_imb, i_imb
    - weak_ratio
    - r_spread (I/V spread)
    - near_zero_phase_present
    """
    V = df[["V1", "V2", "V3"]].astype(float)
    A = df[["A1", "A2", "A3"]].astype(float)

    out = df.copy()
    out["V_mean"] = V.mean(axis=1)
    out["I_mean"] = A.mean(axis=1)
    out["I_sum"] = A.sum(axis=1)
    out["I_max"] = A.abs().max(axis=1)

    out["v_imb"] = (V.max(axis=1) - V.min(axis=1)) / out["V_mean"].abs().clip(lower=v_eps)
    out["i_imb"] = (A.max(axis=1) - A.min(axis=1)) / out["I_mean"].abs().clip(lower=i_eps)

    out["weak_ratio"] = (A.min(axis=1) / A.max(axis=1).abs().clip(lower=i_eps)).clip(0, 1)

    # ratios I/V per phase with protection
    R = pd.DataFrame({
        "r1": A["A1"] / V["V1"].abs().clip(lower=v_eps),
        "r2": A["A2"] / V["V2"].abs().clip(lower=v_eps),
        "r3": A["A3"] / V["V3"].abs().clip(lower=v_eps),
    }).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    r_min = R.min(axis=1).clip(lower=r_eps)
    out["r_spread"] = (R.max(axis=1) / r_min).clip(lower=1.0)

    out["near_zero_phase_present"] = (A.abs().le(i_near_zero_thr).sum(axis=1) >= 1).astype(int)

    return out

# ===================== Model inference (NO forced anomaly rate) =====================
def model_decision_flags(Xs, model_if, model_svm=None, thr_if=0.0, thr_svm=0.0, use_or=True):
    """
    قرار النموذج الحقيقي:
    decision_function >= 0 غالبًا طبيعي
    decision_function < 0 غالبًا شاذ
    نسمح بعتبات قابلة للضبط (thr_if, thr_svm) لتقليل/زيادة الحساسية.
    """
    df_if = model_if.decision_function(Xs)  # higher => more normal
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

    # score for ranking (not for forcing threshold)
    # convert to "anomaly score": lower decision => higher anomaly
    if df_svm is not None:
        score_ens = (-0.5 * df_if) + (-0.5 * df_svm)
    else:
        score_ens = -df_if

    return df_if, df_svm, score_ens, flags, anom_if, anom_svm

# ===================== Rules + Fusion =====================
def apply_rules_and_fusion(
    df: pd.DataFrame,
    # Confirmed (حاسم)
    v_zero_pct: float = 0.10,
    i_min_for_loss: float = 1.0,
    i_near_zero_thr: float = 0.05,
    i_sum_min: float = 0.5,
    low_v_phases_for_confirm: int = 2,
    i_imb_confirm_thr: float = 1.50,

    # Suspected (bypass)
    v_imb_max_for_bypass: float = 0.08,
    i_imb_min_for_bypass: float = 0.50,
    weak_phase_ratio: float = 0.40,
    r_spread_factor: float = 1.8,

    # Model promotion -> suspected
    bypass_promote_thr: float = 0.60,

    # Model anomaly gating (to avoid "normal-looking" records flagged)
    model_support_iimb: float = 0.25,
    model_support_vimb: float = 0.05,
    model_support_rspread: float = 1.30,
    model_support_weakratio: float = 0.80,
    model_support_bypass: float = 0.35,
):
    out = df.copy()
    V = out[["V1", "V2", "V3"]].astype(float)
    A = out[["A1", "A2", "A3"]].astype(float)

    # load presence (avoid noise)
    load_ok = (out["I_sum"] >= i_sum_min) | (out["I_max"] >= i_min_for_loss)

    # v≈0 threshold per row
    v_zero_thr = v_zero_pct * out["V_mean"].abs().replace(0, np.nan)

    # Confirmed conditions
    # C1: V≈0 + I on same phase
    c1_v0_i = (
        ((out["V1"] < v_zero_thr) & (out["A1"] >= i_min_for_loss)) |
        ((out["V2"] < v_zero_thr) & (out["A2"] >= i_min_for_loss)) |
        ((out["V3"] < v_zero_thr) & (out["A3"] >= i_min_for_loss))
    )

    # C2: multiple low V phases with load (VT cut / screws / etc.)
    low_v_count = V.lt(v_zero_thr, axis=0).sum(axis=1)
    c2_many_v_low = (low_v_count >= int(low_v_phases_for_confirm))

    # C3: phase current near zero while other phase has significant current (CT/bridge/tamper)
    c3_i_near_zero_with_load = (out["I_max"] >= i_min_for_loss) & (A.abs().le(i_near_zero_thr).sum(axis=1) >= 1)

    # C4: extreme current imbalance
    c4_extreme_iimb = out["i_imb"].fillna(0) >= i_imb_confirm_thr

    confirmed = load_ok & (c1_v0_i | c2_many_v_low | c3_i_near_zero_with_load | c4_extreme_iimb)

    out["reason_confirm_v0_with_i"] = (load_ok & c1_v0_i).astype(int)
    out["reason_confirm_many_v_low"] = (load_ok & c2_many_v_low).astype(int)
    out["reason_confirm_i_near_zero"] = (load_ok & c3_i_near_zero_with_load).astype(int)
    out["reason_confirm_extreme_iimb"] = (load_ok & c4_extreme_iimb).astype(int)

    # Suspected (bypass pattern)
    v_balanced = out["v_imb"] <= v_imb_max_for_bypass
    i_unbalanced = out["i_imb"] >= i_imb_min_for_bypass
    weak_phase = out["weak_ratio"] <= weak_phase_ratio
    r_spread_big = out["r_spread"] >= r_spread_factor

    suspected_pattern = load_ok & v_balanced & i_unbalanced & weak_phase & r_spread_big

    # bypass_score (0..1)
    bypass_score = clip01(
        0.55 * clip01(out["i_imb"].fillna(0) / 2.0) +
        0.25 * clip01(1.0 - out["weak_ratio"].fillna(1)) +
        0.20 * clip01(np.log1p(out["r_spread"].fillna(1)) / np.log(6))
    )
    out["bypass_score"] = bypass_score

    # Model promote to suspected (only if model flagged + load + some support)
    if "is_anomaly" in out.columns:
        model_flag = (out["is_anomaly"] == 1)
    else:
        model_flag = pd.Series(False, index=out.index)

    model_promote_to_suspected = model_flag & load_ok & (
        (out["bypass_score"] >= bypass_promote_thr) |
        (out["i_imb"].fillna(0) >= i_imb_min_for_bypass) |
        (out["r_spread"].fillna(1) >= r_spread_factor) |
        (out["weak_ratio"].fillna(1) <= weak_phase_ratio)
    )

    suspected = (~confirmed) & (suspected_pattern | model_promote_to_suspected)

    out["reason_suspected_pattern"] = suspected_pattern.astype(int)
    out["reason_suspected_model_promote"] = model_promote_to_suspected.astype(int)

    # Model anomaly gating: show ONLY if supported by some electrical sign
    model_support = (
        (out["i_imb"].fillna(0) >= model_support_iimb) |
        (out["v_imb"].fillna(0) >= model_support_vimb) |
        (out["r_spread"].fillna(1) >= model_support_rspread) |
        (out["weak_ratio"].fillna(1) <= model_support_weakratio) |
        (out["bypass_score"].fillna(0) >= model_support_bypass)
    )
    model_only = model_flag & (~confirmed) & (~suspected) & model_support

    out["reason_model_anomaly_only"] = model_only.astype(int)

    out["confirmed_loss"] = confirmed.astype(int)
    out["suspected_loss"] = suspected.astype(int)
    out["model_anomaly"] = model_only.astype(int)

    out["final_label"] = np.select(
        [out["confirmed_loss"] == 1, out["suspected_loss"] == 1, out["model_anomaly"] == 1],
        ["Confirmed Loss", "Suspected Loss", "Model Anomaly"],
        default="Normal"
    )

    return out

# ===================== UI config =====================
st.set_page_config(page_title="ASDCT • CT Loss Detection (Pro)", layout="wide")

st.title("نظام اكتشاف حالات الفاقد المحتملة — عدادات CT")
st.caption("نسخة احترافية: دمج ذكي بين القواعد الحاسمة والنماذج بدون إجبار نسبة شذوذ.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ الإعدادات")

    # Presets
    preset = st.selectbox(
        "Preset (جاهز)",
        ["متوازن (Recommended)", "حساس (لا يفوّت)", "صارم (تقليل الإنذارات)"],
        index=0
    )

    # Default values by preset
    if preset == "حساس (لا يفوّت)":
        defaults = dict(
            v_zero_pct=0.12, i_min_for_loss=1.0, i_near_zero_thr=0.08, i_sum_min=0.5,
            low_v_phases_for_confirm=2, i_imb_confirm_thr=1.20,
            v_imb_max_for_bypass=0.10, i_imb_min_for_bypass=0.45, weak_phase_ratio=0.45, r_spread_factor=1.7,
            bypass_promote_thr=0.55,
            thr_if=0.0, thr_svm=0.0, combine_or=True
        )
    elif preset == "صارم (تقليل الإنذارات)":
        defaults = dict(
            v_zero_pct=0.08, i_min_for_loss=2.0, i_near_zero_thr=0.05, i_sum_min=1.0,
            low_v_phases_for_confirm=3, i_imb_confirm_thr=1.80,
            v_imb_max_for_bypass=0.06, i_imb_min_for_bypass=0.65, weak_phase_ratio=0.35, r_spread_factor=2.2,
            bypass_promote_thr=0.70,
            thr_if=0.0, thr_svm=0.0, combine_or=True
        )
    else:  # balanced
        defaults = dict(
            v_zero_pct=0.10, i_min_for_loss=1.0, i_near_zero_thr=0.05, i_sum_min=0.5,
            low_v_phases_for_confirm=2, i_imb_confirm_thr=1.50,
            v_imb_max_for_bypass=0.08, i_imb_min_for_bypass=0.50, weak_phase_ratio=0.40, r_spread_factor=1.8,
            bypass_promote_thr=0.60,
            thr_if=0.0, thr_svm=0.0, combine_or=True
        )

    with st.expander("🧠 إعدادات النماذج (بدون إجبار نسبة)", expanded=True):
        combine_or = st.toggle("دمج IF و OCSVM بطريقة OR (أي واحد يشير = شذوذ)", value=defaults["combine_or"])
        thr_if = st.number_input("Threshold لـ IF (decision_function < thr => anomaly)", value=float(defaults["thr_if"]), step=0.05)
        thr_svm = st.number_input("Threshold لـ OCSVM (decision_function < thr => anomaly)", value=float(defaults["thr_svm"]), step=0.05)
        st.caption("الافتراضي 0 عادةً صحيح. قلّل العتبة (مثلاً -0.2) لتقليل الشذوذ، وارفعها (مثلاً +0.2) لزيادة الحساسية.")

    with st.expander("✅ قواعد Confirmed Loss (حاسمة)", expanded=True):
        v_zero_pct = st.slider("نسبة الجهد لاعتباره ≈ صفر", 0.02, 0.30, float(defaults["v_zero_pct"]), 0.01)
        i_min_for_loss = st.slider("تيار يدل على وجود حمل (A)", 0.1, 50.0, float(defaults["i_min_for_loss"]), 0.1)
        i_near_zero_thr = st.slider("تيار يعتبر ≈ صفر (A)", 0.0, 5.0, float(defaults["i_near_zero_thr"]), 0.01)
        i_sum_min = st.slider("حد مجموع التيار لإلغاء الضجيج (A)", 0.0, 50.0, float(defaults["i_sum_min"]), 0.1)
        low_v_phases_for_confirm = st.selectbox("Confirmed إذا عدد فازات الجهد المنخفض ≥", [1, 2, 3], index=[1,2,3].index(int(defaults["low_v_phases_for_confirm"])))
        i_imb_confirm_thr = st.slider("عدم اتزان تيار شديد لتأكيد الفاقد (i_imb ≥)", 0.50, 3.00, float(defaults["i_imb_confirm_thr"]), 0.05)

    with st.expander("⚠️ قواعد Suspected Loss (جنابر/Bypass)", expanded=True):
        v_imb_max_for_bypass = st.slider("اتزان الجهد (v_imb ≤)", 0.01, 0.25, float(defaults["v_imb_max_for_bypass"]), 0.01)
        i_imb_min_for_bypass = st.slider("عدم اتزان التيار (i_imb ≥)", 0.10, 2.50, float(defaults["i_imb_min_for_bypass"]), 0.05)
        weak_phase_ratio = st.slider("طور ضعيف: min/max ≤", 0.05, 0.90, float(defaults["weak_phase_ratio"]), 0.05)
        r_spread_factor = st.slider("تشتيت نسب I/V (≥)", 1.1, 8.0, float(defaults["r_spread_factor"]), 0.1)
        bypass_promote_thr = st.slider("ترقية شذوذ النموذج إلى Suspected إذا bypass_score ≥", 0.10, 0.95, float(defaults["bypass_promote_thr"]), 0.05)

    with st.expander("🧯 فلترة Model Anomaly (تقليل الإنذارات الكاذبة)", expanded=False):
        model_support_iimb = st.slider("ادعم شذوذ النموذج إذا i_imb ≥", 0.0, 1.0, 0.25, 0.05)
        model_support_vimb = st.slider("ادعم شذوذ النموذج إذا v_imb ≥", 0.0, 0.2, 0.05, 0.01)
        model_support_rspread = st.slider("ادعم شذوذ النموذج إذا r_spread ≥", 1.0, 3.0, 1.30, 0.05)
        model_support_weakratio = st.slider("ادعم شذوذ النموذج إذا weak_ratio ≤", 0.2, 1.0, 0.80, 0.05)
        model_support_bypass = st.slider("ادعم شذوذ النموذج إذا bypass_score ≥", 0.0, 1.0, 0.35, 0.05)

    st.markdown("---")
    st.download_button(
        "⬇️ تنزيل قالب Excel (Template)",
        make_template_excel(),
        file_name="asdct_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("---")
    st.caption("📦 مسارات النماذج:")
    st.code(str(PATH_SCALER))
    st.code(str(PATH_IF))
    st.code(str(PATH_SVM))

# ---------------- Main tabs ----------------
tab_infer, tab_help = st.tabs(["📈 تحليل ملف", "ℹ️ المساعدة"])

with tab_infer:
    st.subheader("رفع ملف Excel للتحليل")
    uploaded = st.file_uploader(
        "الأعمدة المطلوبة: Meter Number, V1, V2, V3, A1, A2, A3",
        type=["xlsx"]
    )

    if not PATH_SCALER.exists() or not PATH_IF.exists():
        st.error("يجب توفر scaler.joblib و isolation_forest.joblib في نفس مجلد app.py.")
    else:
        if uploaded is None:
            st.info("ارفع ملف Excel للبدء.")
        else:
            try:
                # Read and validate
                df = pd.read_excel(uploaded)
                df.columns = [c.strip() for c in df.columns]
                validate_columns(df, [ID_COL] + FEATURE_COLS)

                df = safe_to_numeric(df, FEATURE_COLS)
                df_infer = df.dropna(subset=FEATURE_COLS).copy().reset_index(drop=True)

                if df_infer.empty:
                    st.warning("لا يوجد صفوف صالحة بعد التنظيف (قيم ناقصة/غير رقمية).")
                    st.stop()

                # Prepare X
                X = df_infer[FEATURE_COLS].copy()
                models = load_models()
                scaler = models["scaler"]
                model_if = models["if"]
                model_svm = models.get("svm", None)

                Xs = scaler.transform(X.values)

                # Model flags without forcing rate
                df_if, df_svm, score_ens, flags, anom_if, anom_svm = model_decision_flags(
                    Xs, model_if, model_svm=model_svm,
                    thr_if=float(thr_if), thr_svm=float(thr_svm),
                    use_or=bool(combine_or)
                )

                used_models = "IF + OCSVM" if model_svm is not None else "IF فقط"
                st.success(f"تم تحليل {df_infer.shape[0]} سجل باستخدام {used_models} بدون إجبار نسبة شذوذ.")

                # Build detailed results
                detailed = df_infer[[ID_COL] + FEATURE_COLS].copy()
                detailed["df_if"] = df_if
                detailed["anom_if"] = anom_if
                if df_svm is not None:
                    detailed["df_svm"] = df_svm
                    detailed["anom_svm"] = anom_svm
                detailed["score_ensemble"] = score_ens
                detailed["is_anomaly"] = flags

                # Electrical features
                detailed = compute_signal_features(
                    detailed,
                    i_near_zero_thr=float(i_near_zero_thr)
                )

                # Fusion
                detailed = apply_rules_and_fusion(
                    detailed,
                    v_zero_pct=float(v_zero_pct),
                    i_min_for_loss=float(i_min_for_loss),
                    i_near_zero_thr=float(i_near_zero_thr),
                    i_sum_min=float(i_sum_min),
                    low_v_phases_for_confirm=int(low_v_phases_for_confirm),
                    i_imb_confirm_thr=float(i_imb_confirm_thr),
                    v_imb_max_for_bypass=float(v_imb_max_for_bypass),
                    i_imb_min_for_bypass=float(i_imb_min_for_bypass),
                    weak_phase_ratio=float(weak_phase_ratio),
                    r_spread_factor=float(r_spread_factor),
                    bypass_promote_thr=float(bypass_promote_thr),
                    model_support_iimb=float(model_support_iimb),
                    model_support_vimb=float(model_support_vimb),
                    model_support_rspread=float(model_support_rspread),
                    model_support_weakratio=float(model_support_weakratio),
                    model_support_bypass=float(model_support_bypass),
                )

                # KPIs
                c_confirmed = int((detailed["final_label"] == "Confirmed Loss").sum())
                c_suspected = int((detailed["final_label"] == "Suspected Loss").sum())
                c_modelanom = int((detailed["final_label"] == "Model Anomaly").sum())
                c_normal = int((detailed["final_label"] == "Normal").sum())
                c_anom = int(detailed["is_anomaly"].sum())

                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Confirmed", c_confirmed)
                k2.metric("Suspected", c_suspected)
                k3.metric("Model Anomaly", c_modelanom)
                k4.metric("Normal", c_normal)
                k5.metric("Model anomalies (raw)", c_anom)

                st.markdown("---")

                # Summary per meter
                summary = detailed.groupby(ID_COL, as_index=False).agg(
                    rows=("final_label", "size"),
                    confirmed=("confirmed_loss", "sum"),
                    suspected=("suspected_loss", "sum"),
                    model_anomaly=("model_anomaly", "sum"),
                    model_flags=("is_anomaly", "sum"),
                    max_bypass_score=("bypass_score", "max"),
                    max_score_ensemble=("score_ensemble", "max"),
                    max_i=("I_max", "max"),
                    max_vimb=("v_imb", "max"),
                    max_iimb=("i_imb", "max"),
                    max_rsp=("r_spread", "max"),
                )

                def meter_label(row):
                    if row["confirmed"] > 0:
                        return "Confirmed Loss"
                    if row["suspected"] > 0:
                        return "Suspected Loss"
                    if row["model_anomaly"] > 0:
                        return "Model Anomaly"
                    return "Normal"

                summary["meter_final_label"] = summary.apply(meter_label, axis=1)

                # Sorting priority
                pr = {"Confirmed Loss": 0, "Suspected Loss": 1, "Model Anomaly": 2, "Normal": 3}
                summary["prio"] = summary["meter_final_label"].map(pr).fillna(9).astype(int)
                summary = summary.sort_values(
                    by=["prio", "confirmed", "suspected", "max_bypass_score", "max_score_ensemble"],
                    ascending=[True, False, False, False, False]
                ).drop(columns=["prio"]).reset_index(drop=True)

                # Tabs for outputs
                t_all, t_conf, t_susp, t_model = st.tabs(["📄 الكل", "✅ Confirmed", "⚠️ Suspected", "🧠 Model Anomaly"])

                with t_all:
                    render_table(summary, "ملخص العدادات (Meter Summary)", "summary_asdct_pro.xlsx", height=380)
                    render_table(detailed, "النتائج التفصيلية (Detailed)", "detailed_asdct_pro.xlsx", height=450)

                with t_conf:
                    dfc = detailed[detailed["final_label"] == "Confirmed Loss"].copy()
                    render_table(dfc, "Confirmed Loss - Detailed", "confirmed_asdct_pro.xlsx", height=520)

                with t_susp:
                    dfs = detailed[detailed["final_label"] == "Suspected Loss"].copy()
                    render_table(dfs, "Suspected Loss - Detailed", "suspected_asdct_pro.xlsx", height=520)

                with t_model:
                    dfm = detailed[detailed["final_label"] == "Model Anomaly"].copy()
                    render_table(dfm, "Model Anomaly - Detailed", "model_anomaly_asdct_pro.xlsx", height=520)
                    st.caption("ملاحظة: Model Anomaly هنا لا يظهر إلا إذا كان شذوذ النموذج مدعوم بمؤشر كهربائي (لتقليل الإنذارات الكاذبة).")

                # Reasons dashboard
                st.markdown("---")
                st.subheader("لوحة أسباب القرار (Reasons)")
                reason_cols = [
                    "reason_confirm_v0_with_i",
                    "reason_confirm_many_v_low",
                    "reason_confirm_i_near_zero",
                    "reason_confirm_extreme_iimb",
                    "reason_suspected_pattern",
                    "reason_suspected_model_promote",
                    "reason_model_anomaly_only",
                ]
                reasons_sum = detailed[reason_cols].sum().sort_values(ascending=False).reset_index()
                reasons_sum.columns = ["Reason", "Count"]
                st.dataframe(reasons_sum, use_container_width=True)

            except Exception as e:
                st.exception(e)

with tab_help:
    st.markdown("""
### ما الذي يضمن عدم إهمال حالات الفاقد؟
نستخدم **طبقات قرار**:

#### 1) Confirmed Loss (حاسم – يغلب أي نتيجة نموذج)
- **V≈0 مع وجود I على نفس الفازة**
- **2 أو 3 فازات جهد منخفض جدًا + يوجد حمل** (قص VT/مسامير/…)
- **فازة تيارها ≈0 مع وجود تيار قوي بفازة أخرى** (غير منطقي لعدادات CT الكبيرة غالبًا)
- **عدم اتزان تيار شديد جدًا (i_imb)**

> هذه القواعد هدفها عدم تفويت أي فاقد مؤكد.

#### 2) Suspected Loss (جنابر/Bypass)
- نمط bypass: **V متزن + I غير متزن + طور ضعيف + r_spread كبير + حمل**
- أو ترقية من النموذج عند وجود حمل ودعم من المؤشرات أو bypass_score.

#### 3) Model Anomaly (للمراجعة)
- شذوذ نموذج **مدعوم بمؤشر كهربائي** فقط، لتقليل الإنذارات على الحالات الطبيعية.

#### 4) Normal

### لماذا أزلنا target_rate؟
لأنه كان يجبر النظام يطلع نسبة شذوذ ثابتة حتى لو البيانات سليمة.
الآن نستخدم **decision_function threshold** (افتراضي 0) وهو القرار الحقيقي للنموذج.
""")

st.markdown("---")
st.markdown("👨‍💻 **تطوير :** مشهور العباس 2026 | نسخة محسّنة: Pro Fusion بدون إجبار نسبة شذوذ")
