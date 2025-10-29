# -*- coding: utf-8 -*-
"""
ASDCT - Streamlit Inference (IF + OCSVM)
يستخدم الملفات الموجودة في جذر المشروع:
- isolation_forest.joblib
- ocsvm.joblib
- scaler.joblib

تشغيل:
    streamlit run app.py
"""

from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ---------------------------
# مسارات الملفات (جذر المستودع)
# ---------------------------
PROJECT_DIR = Path(__file__).resolve().parent
PATH_SCALER = PROJECT_DIR / "scaler.joblib"
PATH_IF = PROJECT_DIR / "isolation_forest.joblib"
PATH_SVM = PROJECT_DIR / "ocsvm.joblib"      # اختياري إن لم يتوفر سيعمل IF فقط

# ---------------------------
# أعمدة البيانات
# ---------------------------
ID_COL = "Meter Number"
FEATURE_COLS = ["V1", "V2", "V3", "A1", "A2", "A3"]

# ---------------------------
# دوال مساعدة
# ---------------------------
def validate_columns(df: pd.DataFrame, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"الأعمدة الناقصة في الملف: {missing}")

def normalize_score(raw: np.ndarray, invert=False) -> np.ndarray:
    s = np.array(raw, dtype=float)
    if invert:
        s = -s
    lo, hi = np.nanpercentile(s, [5, 95])
    if hi - lo < 1e-9:
        return np.clip(np.full_like(s, 0.5), 0, 1)
    sn = (s - lo) / (hi - lo)
    return np.clip(sn, 0, 1)

def percentile_threshold(scores: np.ndarray, target_rate: float) -> float:
    target_rate = float(np.clip(target_rate, 0.001, 0.3))
    return np.nanpercentile(scores, 100 * (1 - target_rate))

def compute_repeat_confidence(flags_for_meter: pd.Series) -> float:
    frac = float(flags_for_meter.mean())  # نسبة الشذوذ لهذا العداد
    bonus = 0.15 if flags_for_meter.sum() >= 3 else (0.08 if flags_for_meter.sum() == 2 else 0.0)
    return float(np.clip(frac + bonus, 0.0, 1.0))

def create_download(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
    return out.getvalue()

# ---------------------------
# واجهة Streamlit
# ---------------------------
st.set_page_config(page_title="ASDCT - IF + OCSVM", layout="wide")
st.title("كشف حالات الفاقد في عدادات CT — (Isolation Forest + One-Class SVM)")

with st.sidebar:
    st.header("إعدادات الكشف")
    target_rate = st.slider("النسبة المستهدفة للشذوذ (%)", 1, 20, 5, 1) / 100.0
    st.caption("تحدد تقريبًا نسبة النقاط التي ستصنّف كشذوذ.")
    st.markdown("---")
    st.subheader("مسارات النماذج")
    st.code(str(PATH_IF))
    st.code(str(PATH_SVM))
    st.code(str(PATH_SCALER))

tab_infer, tab_help = st.tabs(["📈 تحليل ملف", "ℹ️ مساعدة"])

with tab_infer:
    st.subheader("رفع ملف Excel للتحليل")
    uploaded = st.file_uploader(
        "ارفع ملفًا يحتوي الأعمدة: Meter Number, V1, V2, V3, A1, A2, A3",
        type=["xlsx"]
    )

    # تحقق من توفر الملفات الأساسية
    if not PATH_SCALER.exists() or not PATH_IF.exists():
        st.error("يجب توفر الملفات: scaler.joblib و isolation_forest.joblib في نفس مجلد app.py.")
    else:
        if uploaded is not None:
            try:
                # قراءة الملف وتنظيف الأعمدة
                df = pd.read_excel(uploaded)
                df.columns = [c.strip() for c in df.columns]
                validate_columns(df, [ID_COL] + FEATURE_COLS)

                # تنظيف/تحويل القيم
                df_infer = df.dropna(subset=FEATURE_COLS).copy()
                X = df_infer[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

                # تحميل النماذج
                scaler = joblib.load(PATH_SCALER)
                model_if = joblib.load(PATH_IF)

                # قد لا يتوفر OCSVM (لو حذفته)، عندها نشتغل بـ IF فقط
                model_svm = joblib.load(PATH_SVM) if PATH_SVM.exists() else None

                # تحويل الميزات
                Xs = scaler.transform(X.values)

                # درجات IF (أكبر = أشد شذوذ بعد العكس)
                s_if = normalize_score(-model_if.score_samples(Xs))

                if model_svm is not None:
                    # decision_function أكبر = أكثر "طبيعي" → نعكس
                    s_svm = normalize_score(-model_svm.decision_function(Xs))
                    s_ens = np.nanmean(np.c_[s_if, s_svm], axis=1)  # تجميعي IF+SVM
                    used_models = "IF + OCSVM"
                else:
                    s_svm = None
                    s_ens = s_if
                    used_models = "IF فقط"

                # حساب العتبة حسب النسبة المستهدفة
                thr = percentile_threshold(s_ens, target_rate)
                flags = (s_ens > thr).astype(int)

                # جدول تفصيلي
                detailed = df_infer.loc[X.index, [ID_COL] + FEATURE_COLS].copy()
                detailed["score_if"] = s_if
                if s_svm is not None:
                    detailed["score_svm"] = s_svm
                detailed["score_ensemble"] = s_ens
                detailed["is_anomaly"] = flags

                # ملخص لكل عداد
                grp = detailed.groupby(ID_COL, as_index=False).agg(
                    rows=("is_anomaly", "size"),
                    anomalies=("is_anomaly", "sum"),
                    max_score=("score_ensemble", "max"),
                    mean_score=("score_ensemble", "mean"),
                )
                grp["repeat_count"] = grp["anomalies"].astype(int)
                grp["confidence_from_score"] = grp["max_score"].clip(0, 1)

                conf_repeat = []
                for _, sub in detailed.groupby(ID_COL):
                    conf_repeat.append(compute_repeat_confidence(sub["is_anomaly"]))
                grp["confidence_from_repeats"] = np.array(conf_repeat).clip(0, 1)

                summary = grp[[ID_COL, "repeat_count", "confidence_from_score", "confidence_from_repeats"]].sort_values(
                    by=["confidence_from_repeats", "confidence_from_score", "repeat_count"],
                    ascending=False
                ).reset_index(drop=True)

                # عرض
                st.success(
                    f"تم تحليل {detailed.shape[0]} سجلًّا باستخدام {used_models}. "
                    f"الشذوذ: {int(flags.sum())} (≈ {100*flags.mean():.2f}%)."
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("ملخص العدادات")
                    st.dataframe(summary, use_container_width=True, height=450)
                    st.download_button(
                        "⬇️ تنزيل الملخص (Excel)",
                        data=create_download(summary),
                        file_name="summary_asdct.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                with c2:
                    st.subheader("النتائج التفصيلية")
                    st.dataframe(detailed, use_container_width=True, height=450)
                    st.download_button(
                        "⬇️ تنزيل النتائج التفصيلية (Excel)",
                        data=create_download(detailed),
                        file_name="detailed_asdct.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                st.markdown("---")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("عدد العدادات", detailed[ID_COL].nunique())
                k2.metric("إجمالي السجلات", int(detailed.shape[0]))
                k3.metric("نسبة الشذوذ", f"{100 * flags.mean():.2f}%")
                st.caption(f"العتبة (من التجميعي): {thr:.4f}")

            except Exception as e:
                st.exception(e)

with tab_help:
    st.markdown("""
### المتطلبات
- وجود الملفات التالية في **نفس مجلد** `app.py` (جذر المشروع):
  - `scaler.joblib`
  - `isolation_forest.joblib`
  - `ocsvm.joblib` *(اختياري — لو غير موجود سيعمل IF فقط)*
- ملف الإدخال يجب أن يحتوي الأعمدة:
  **Meter Number, V1, V2, V3, A1, A2, A3**

### آلية القرار
- يتم حساب درجة شذوذ من IF و OCSVM (إن وجد).
- الدرجة النهائية = متوسط الدرجتين.
- يتم تحديد العتبة تلقائيًا وفق **النسبة المستهدفة للشذوذ** من الشريط الجانبي.
- الملخص يجمع التكرار لكل عداد ويحسب ثقة من الدرجة وثقة من التكرار.
""")
