# -*- coding: utf-8 -*-
"""
ASDCT - Streamlit (IF + OCSVM) + Rules:
- يعمل من جذر المستودع:
  app.py
  isolation_forest.joblib
  ocsvm.joblib          (اختياري - إن لم يوجد يعمل IF فقط)
  scaler.joblib
  requirements.txt

تشغيل محلي:
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

# ===================== مسارات الملفات =====================
PROJECT_DIR = Path(__file__).resolve().parent
PATH_SCALER = PROJECT_DIR / "scaler.joblib"
PATH_IF     = PROJECT_DIR / "isolation_forest.joblib"
PATH_SVM    = PROJECT_DIR / "ocsvm.joblib"  # اختياري

# ===================== أعمدة البيانات =====================
ID_COL = "Meter Number"
FEATURE_COLS = ["V1","V2","V3","A1","A2","A3"]

# ===================== أدوات مساعدة =====================
def validate_columns(df: pd.DataFrame, required_cols):
    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        raise ValueError(f"الأعمدة الناقصة: {miss}")

def normalize_score(raw: np.ndarray, invert=False) -> np.ndarray:
    s = np.array(raw, dtype=float)
    if invert: s = -s
    lo, hi = np.nanpercentile(s, [5,95])
    if hi - lo < 1e-9: return np.clip(np.full_like(s, 0.5), 0, 1)
    return np.clip((s - lo) / (hi - lo), 0, 1)

def percentile_threshold(scores: np.ndarray, target_rate: float) -> float:
    target_rate = float(np.clip(target_rate, 0.001, 0.3))
    return np.nanpercentile(scores, 100 * (1 - target_rate))

def excel_bytes(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        df.to_excel(w, sheet_name="Sheet1", index=False)
    return out.getvalue()

def render_table(df, title, download_name):
    st.subheader(title)
    try:
        st.dataframe(df, use_container_width=True, height=450)
    except Exception as e:
        st.warning(f"تعذر عرض الجدول بمظهر Streamlit، سيتم عرض HTML بسيط. السبب: {e}")
        st.markdown(df.to_html(index=False), unsafe_allow_html=True)
    st.download_button("⬇️ تنزيل (Excel)", excel_bytes(df),
                       file_name=download_name,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ===================== تحميل النماذج (مع رسائل واضحة) =====================
@lru_cache(maxsize=1)
def load_models():
    models = {}
    try:
        models["scaler"] = joblib.load(PATH_SCALER)
    except Exception as e:
        st.error(f"تعذر تحميل scaler.joblib: {e}"); raise
    try:
        models["if"] = joblib.load(PATH_IF)
    except Exception as e:
        st.error(f"تعذر تحميل isolation_forest.joblib: {e}"); raise
    if PATH_SVM.exists():
        try:
            models["svm"] = joblib.load(PATH_SVM)
        except Exception as e:
            st.warning(f"تعذر تحميل ocsvm.joblib، سيتم العمل بـ IF فقط. السبب: {e}")
    return models

# ===================== اشتقاقات المؤشرات =====================
def compute_signal_features(df):
    V = df[["V1","V2","V3"]].astype(float)
    A = df[["A1","A2","A3"]].astype(float)
    df = df.copy()
    df["V_mean"] = V.mean(axis=1)
    df["I_mean"] = A.mean(axis=1)
    df["I_sum"]  = A.sum(axis=1)
    df["v_imb"]  = (V.max(axis=1) - V.min(axis=1)) / df["V_mean"].replace(0, np.nan)
    df["i_imb"]  = (A.max(axis=1) - A.min(axis=1)) / df["I_mean"].replace(0, np.nan)
    df["weak_ratio"] = (A.min(axis=1) / A.max(axis=1).replace(0, np.nan)).replace([np.inf,-np.inf], np.nan).fillna(1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        r1 = A["A1"]/V["V1"]; r2 = A["A2"]/V["V2"]; r3 = A["A3"]/V["V3"]
    R = pd.concat([r1.rename("r1"), r2.rename("r2"), r3.rename("r3")], axis=1)
    R = R.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    df["r_spread"] = (R.max(axis=1) / R.replace(0, np.nan).min(axis=1)).replace([np.inf,-np.inf], np.nan).fillna(1.0)
    df["zero_phase_present"] = (A.eq(0).sum(axis=1) >= 1).astype(int)
    return df

def label_confirmed_and_suspected(
    df,
    v_zero_pct=0.10,
    i_min_for_loss=1.0,
    v_imb_max_for_bypass=0.06,
    i_imb_min_for_bypass=0.60,
    weak_phase_ratio=0.35,
    r_spread_factor=2.0,
    i_sum_min=0.5
):
    V = df[["V1","V2","V3"]].astype(float)
    Vmean = df["V_mean"]

    # --- Confirmed ---
    v_zero_thr = v_zero_pct * Vmean.replace(0, np.nan)

    c1 = ((df["V1"] < v_zero_thr) & (df["A1"] >= i_min_for_loss)) | \
         ((df["V2"] < v_zero_thr) & (df["A2"] >= i_min_for_loss)) | \
         ((df["V3"] < v_zero_thr) & (df["A3"] >= i_min_for_loss))
    all_v_low = (V.lt(v_zero_thr, axis=0).sum(axis=1) == 3)

    c2 = (df["v_imb"] >= 0.20) & (df["I_mean"] >= i_min_for_loss)
    c3 = (df["zero_phase_present"] == 1) & (df["I_sum"] >= i_sum_min)

    confirmed = (c1 & (~all_v_low)) | c2 | c3

    # --- Suspected (bypass) ---
    v_balanced   = df["v_imb"] <= v_imb_max_for_bypass
    i_unbalanced = df["i_imb"] >= i_imb_min_for_bypass
    weak_phase   = df["weak_ratio"] <= weak_phase_ratio
    r_spread_big = df["r_spread"]  >= r_spread_factor
    load_ok      = df["I_sum"]     >= i_sum_min

    suspected = (~confirmed) & v_balanced & i_unbalanced & weak_phase & r_spread_big & load_ok

    bypass_score = (
        0.5 * (df["i_imb"]/2.0).clip(0,1) +
        0.3 * (1.0 - df["weak_ratio"]).clip(0,1) +
        0.2 * (np.log1p(df["r_spread"]) / np.log(6)).clip(0,1)
    ).clip(0,1).fillna(0.0)

    out = df.copy()
    out["confirmed_loss"] = confirmed.astype(int)
    out["suspected_loss"] = suspected.astype(int)
    out["bypass_score"]   = bypass_score
    out["final_label"] = np.select(
        [out["confirmed_loss"]==1, out["suspected_loss"]==1],
        ["Confirmed Loss", "Suspected Loss"],
        default="Normal/Model"
    )
    return out

# ===================== واجهة Streamlit =====================
st.set_page_config(page_title="ASDCT • IF + OCSVM + Rules", layout="wide")
st.title("نظام اكتشاف حالات الفاقد المحتملة عدادات CT")

with st.sidebar:
    st.header("إعدادات النموذج")
    target_rate = st.slider("النسبة المستهدفة للشذوذ (%)", 1, 20, 5, 1) / 100.0
    st.caption("تحدد تقريبًا نسبة النقاط المصنفة كشذوذ من النموذج.")

    st.markdown("---")
    st.header("ثوابت الفاقد المؤكّد/المحتمل")
    v_zero_pct = st.slider("نسبة الجهد لاعتباره ≈ صفر", 0.02, 0.30, 0.10, 0.01)
    i_min_for_loss = st.slider("حد التيار للفاقد المؤكد (A)", 0.1, 10.0, 1.0, 0.1)
    v_imb_max_for_bypass = st.slider("اتزان الجهد (v_imb ≤)", 0.01, 0.20, 0.06, 0.01)
    i_imb_min_for_bypass = st.slider("عدم اتزان التيار (i_imb ≥)", 0.10, 2.50, 0.60, 0.05)
    weak_phase_ratio = st.slider("طور ضعيف: min/max ≤", 0.05, 0.80, 0.35, 0.05)
    r_spread_factor = st.slider("تشتيت نسب I/V (≥)", 1.2, 5.0, 2.0, 0.1)
    i_sum_min = st.slider("حد مجموع التيار لإلغاء الضجيج (A)", 0.0, 5.0, 0.5, 0.1)

    st.markdown("---")
    st.subheader("ملفات النماذج (جذر المشروع)")
    st.code(str(PATH_IF))
    st.code(str(PATH_SVM))
    st.code(str(PATH_SCALER))

tab_infer, tab_help = st.tabs(["📈 تحليل ملف", "ℹ️ مساعدة"])

with tab_infer:
    st.subheader("رفع ملف Excel للتحليل")
    uploaded = st.file_uploader("أعمدة مطلوبة: Meter Number, V1, V2, V3, A1, A2, A3", type=["xlsx"])

    if not PATH_SCALER.exists() or not PATH_IF.exists():
        st.error("يجب توفر الملفات: scaler.joblib و isolation_forest.joblib في نفس مجلد app.py.")
    else:
        if uploaded is not None:
            try:
                df = pd.read_excel(uploaded)
                df.columns = [c.strip() for c in df.columns]
                validate_columns(df, [ID_COL] + FEATURE_COLS)

                # تنظيف وتحويل
                df_infer = df.dropna(subset=FEATURE_COLS).copy()
                X = df_infer[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

                # تحميل النماذج وتحويل الميزات
                models = load_models()
                scaler = models["scaler"]; model_if = models["if"]; model_svm = models.get("svm")
                Xs = scaler.transform(X.values)

                # درجات النماذج
                s_if  = normalize_score(-model_if.score_samples(Xs))
                if model_svm is not None:
                    s_svm = normalize_score(-model_svm.decision_function(Xs))
                    s_ens = np.nanmean(np.c_[s_if, s_svm], axis=1)
                    used_models = "IF + OCSVM"
                else:
                    s_svm = None
                    s_ens = s_if
                    used_models = "IF فقط"

                thr = percentile_threshold(s_ens, target_rate)
                flags = (s_ens > thr).astype(int)

                # تجميع النتائج التفصيلية
                detailed = df_infer.loc[X.index, [ID_COL] + FEATURE_COLS].copy()
                detailed["score_if"] = s_if
                if s_svm is not None: detailed["score_svm"] = s_svm
                detailed["score_ensemble"] = s_ens
                detailed["is_anomaly"] = flags

                # اشتق المؤشرات وتطبيق القواعد
                detailed = compute_signal_features(detailed)
                detailed = label_confirmed_and_suspected(
                    detailed,
                    v_zero_pct=v_zero_pct,
                    i_min_for_loss=i_min_for_loss,
                    v_imb_max_for_bypass=v_imb_max_for_bypass,
                    i_imb_min_for_bypass=i_imb_min_for_bypass,
                    weak_phase_ratio=weak_phase_ratio,
                    r_spread_factor=r_spread_factor,
                    i_sum_min=i_sum_min
                )

                # ملخص لكل عداد
                summary = detailed.groupby(ID_COL, as_index=False).agg(
                    rows=("is_anomaly", "size"),
                    anomalies=("is_anomaly", "sum"),
                    confirmed=("confirmed_loss", "sum"),
                    suspected=("suspected_loss", "sum"),
                    max_bypass_score=("bypass_score", "max"),
                    max_model_score=("score_ensemble", "max")
                ).sort_values(
                    by=["confirmed","suspected","max_bypass_score","max_model_score"],
                    ascending=False
                ).reset_index(drop=True)

                st.success(
                    f"تم تحليل {detailed.shape[0]} سجلًّا باستخدام {used_models}. "
                    f"الشذوذ (النموذج): {int(flags.sum())} (≈ {100*flags.mean():.2f}%)."
                )

                # عرض وتنزيل
                render_table(summary, "ملخّص العدادات", "summary_asdct.xlsx")
                render_table(detailed, "النتائج التفصيلية", "detailed_asdct.xlsx")

                # KPIs
                st.markdown("---")
                k1,k2,k3 = st.columns(3)
                k1.metric("عدد العدادات", detailed[ID_COL].nunique())
                k2.metric("إجمالي السجلات", int(detailed.shape[0]))
                k3.metric("نسبة الشذوذ (النموذج)", f"{100*flags.mean():.2f}%")
                st.caption(f"العتبة (من التجميعي): {thr:.4f}")

            except Exception as e:
                st.exception(e)

with tab_help:
    st.markdown("""
### كيف تُتخذ القرارات؟
- **النموذج (IF + OCSVM)** يعطي درجة شذوذ ويحدد العتبة تلقائيًا حسب النسبة المختارة.
- **فاقد مؤكّد**: قواعد كهربائية حاسمة (تيار مع جهد ≈ صفر، أو اختلال جهد شديد مع حمل، أو طور صفر مع حمل).
- **فاقد محتمل**: «جنابر» — جهود متزنة، تيارات غير متزنة بوضوح، طور ضعيف جدًا، وتشتت كبير في نسب I/V.
- يمكن تعديل العتبات من الشريط الجانبي لضبط الحساسية حسب الواقع الميداني.

### شكل ملف الإدخال
- Meter Number, V1, V2, V3, A1, A2, A3
""")

# فاصل وتعريف المطوّر
st.markdown("---")
st.markdown("👨‍💻 **تطوير :** مشهور العباس | 00966553339838 | ")
