# -*- coding: utf-8 -*-
"""
ASDCT - Streamlit (IF + optional OCSVM) + Smart Fusion (Rules + Models)

تشغيل:
  pip install -r requirements.txt
  streamlit run app.py

ملفات مطلوبة بجذر المشروع:
  app.py
  scaler.joblib
  isolation_forest.joblib
  ocsvm.joblib  (اختياري)
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
    """
    يُحوِّل أي سكور إلى [0..1] بطريقة robust (5-95 percentile).
    """
    s = np.array(raw, dtype=float)
    if invert:
        s = -s
    lo, hi = np.nanpercentile(s, [5, 95])
    if hi - lo < 1e-9:
        return np.clip(np.full_like(s, 0.5), 0, 1)
    return np.clip((s - lo) / (hi - lo), 0, 1)

def percentile_threshold(scores: np.ndarray, target_rate: float) -> float:
    """
    عتبة بناءً على النسبة المستهدفة (تقريبًا).
    """
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

    st.download_button(
        "⬇️ تنزيل (Excel)",
        excel_bytes(df),
        file_name=download_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ===================== تحميل النماذج =====================
@lru_cache(maxsize=1)
def load_models():
    models = {}
    try:
        models["scaler"] = joblib.load(PATH_SCALER)
    except Exception as e:
        st.error(f"تعذر تحميل scaler.joblib: {e}")
        raise

    try:
        models["if"] = joblib.load(PATH_IF)
    except Exception as e:
        st.error(f"تعذر تحميل isolation_forest.joblib: {e}")
        raise

    if PATH_SVM.exists():
        try:
            models["svm"] = joblib.load(PATH_SVM)
        except Exception as e:
            st.warning(f"تعذر تحميل ocsvm.joblib، سيتم العمل بـ IF فقط. السبب: {e}")

    return models

# ===================== اشتقاقات المؤشرات (محسنة) =====================
def compute_signal_features(
    df: pd.DataFrame,
    v_eps: float = 1e-6,
    i_eps: float = 1e-6,
    r_eps: float = 1e-6,
    i_near_zero_thr: float = 0.05
) -> pd.DataFrame:
    """
    مؤشرات كهربائية لتحسين كشف:
    - V/I imbalance
    - weak phase
    - spread of I/V
    - near-zero current phase
    """
    V = df[["V1","V2","V3"]].astype(float)
    A = df[["A1","A2","A3"]].astype(float)
    out = df.copy()

    out["V_mean"] = V.mean(axis=1)
    out["I_mean"] = A.mean(axis=1)
    out["I_sum"]  = A.sum(axis=1)
    out["I_max"]  = A.abs().max(axis=1)

    # عدم الاتزان (robust)
    out["v_imb"] = (V.max(axis=1) - V.min(axis=1)) / out["V_mean"].abs().clip(lower=v_eps)
    out["i_imb"] = (A.max(axis=1) - A.min(axis=1)) / out["I_mean"].abs().clip(lower=i_eps)

    # طور ضعيف
    out["weak_ratio"] = (A.min(axis=1) / A.max(axis=1).abs().clip(lower=i_eps)).clip(0, 1)

    # I/V لكل طور (حماية من الصفر)
    R = pd.DataFrame({
        "r1": (A["A1"] / V["V1"].abs().clip(lower=v_eps)),
        "r2": (A["A2"] / V["V2"].abs().clip(lower=v_eps)),
        "r3": (A["A3"] / V["V3"].abs().clip(lower=v_eps)),
    }).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    r_min = R.min(axis=1).clip(lower=r_eps)
    out["r_spread"] = (R.max(axis=1) / r_min).clip(lower=1.0)

    # بدل eq(0): "قريب من صفر"
    out["near_zero_phase_present"] = (A.abs().le(i_near_zero_thr).sum(axis=1) >= 1).astype(int)

    return out

# ===================== قواعد + دمج ذكي (Fusion) =====================
def apply_rules_and_fusion(
    df: pd.DataFrame,
    # قواعد الجهد شبه صفر
    v_zero_pct: float = 0.10,
    # تيار يدل على وجود حمل على الطور
    i_min_for_loss: float = 1.0,
    # تيار "قريب من صفر" للطور (بدل ==0)
    i_near_zero_thr: float = 0.05,
    # فلترة الضجيج: أقل مجموع تيار
    i_sum_min: float = 0.5,

    # Confirmed: عدد فازات الجهد المنخفضة المطلوبة (2 أو 3)
    low_v_phases_for_confirm: int = 2,
    # Confirmed: عدم اتزان تيار شديد جدًا
    i_imb_confirm_thr: float = 1.50,

    # Suspected (bypass pattern)
    v_imb_max_for_bypass: float = 0.08,
    i_imb_min_for_bypass: float = 0.50,
    weak_phase_ratio: float = 0.40,
    r_spread_factor: float = 1.8,

    # ترقية من النموذج إلى Suspected
    bypass_promote_thr: float = 0.60,
):
    """
    يخرج:
    - confirmed_loss, suspected_loss, model_anomaly
    - bypass_score
    - final_label = Confirmed / Suspected / Model Anomaly / Normal
    - أعمدة أسباب القرار
    """
    out = df.copy()
    V = out[["V1","V2","V3"]].astype(float)
    A = out[["A1","A2","A3"]].astype(float)

    # حمل فعلي (نستخدم مجموع التيار أو max تيار)
    load_ok = (out["I_sum"] >= i_sum_min) | (out["I_max"] >= i_min_for_loss)

    # عتبة الجهد شبه صفر: نسبة من متوسط الجهد
    v_zero_thr = v_zero_pct * out["V_mean"].abs().replace(0, np.nan)

    # (C1) V≈0 مع I على نفس الفازة
    c1_v0_i = (
        ((out["V1"] < v_zero_thr) & (out["A1"] >= i_min_for_loss)) |
        ((out["V2"] < v_zero_thr) & (out["A2"] >= i_min_for_loss)) |
        ((out["V3"] < v_zero_thr) & (out["A3"] >= i_min_for_loss))
    )

    # (C2) 2 أو 3 فازات جهدها منخفض جدًا + حمل (قص/فصل جهد/VT)
    low_v_count = V.lt(v_zero_thr, axis=0).sum(axis=1)
    c2_many_v_low = (low_v_count >= int(low_v_phases_for_confirm))

    # (C3) طور/أطوار تيارها ~0 بينما يوجد حمل قوي في طور آخر (غير منطقي لعداد CT كبير)
    # مثال: I_max >= i_min_for_loss و يوجد طور I <= near_zero
    c3_i_near_zero_with_load = (out["I_max"] >= i_min_for_loss) & (A.abs().le(i_near_zero_thr).sum(axis=1) >= 1)

    # (C4) عدم اتزان تيار شديد جدًا
    c4_extreme_iimb = out["i_imb"].fillna(0) >= i_imb_confirm_thr

    # ========== Confirmed Loss (قواعد حاسمة) ==========
    confirmed = load_ok & (c1_v0_i | c2_many_v_low | c3_i_near_zero_with_load | c4_extreme_iimb)

    # أسباب Confirmed (للشرح)
    out["reason_confirm_v0_with_i"] = (load_ok & c1_v0_i).astype(int)
    out["reason_confirm_many_v_low"] = (load_ok & c2_many_v_low).astype(int)
    out["reason_confirm_i_near_zero"] = (load_ok & c3_i_near_zero_with_load).astype(int)
    out["reason_confirm_extreme_iimb"] = (load_ok & c4_extreme_iimb).astype(int)

    # ========== Suspected Loss (bypass pattern) ==========
    v_balanced   = out["v_imb"] <= v_imb_max_for_bypass
    i_unbalanced = out["i_imb"] >= i_imb_min_for_bypass
    weak_phase   = out["weak_ratio"] <= weak_phase_ratio
    r_spread_big = out["r_spread"]  >= r_spread_factor

    suspected_pattern = load_ok & v_balanced & i_unbalanced & weak_phase & r_spread_big

    # bypass_score (درجة)
    bypass_score = (
        0.55 * (out["i_imb"] / 2.0).clip(0, 1) +
        0.25 * (1.0 - out["weak_ratio"]).clip(0, 1) +
        0.20 * (np.log1p(out["r_spread"]) / np.log(6)).clip(0, 1)
    ).clip(0, 1).fillna(0.0)

    out["bypass_score"] = bypass_score

    # ========== ترقية من النموذج ==========
    # إذا النموذج شذوذ ومع حمل، وعنده دعم من مؤشرات أو bypass_score
    if "is_anomaly" in out.columns:
        model_anom = (out["is_anomaly"] == 1) & load_ok
    else:
        model_anom = pd.Series(False, index=out.index)

    model_promote_to_suspected = model_anom & (
        (out["bypass_score"] >= bypass_promote_thr) |
        (out["i_imb"].fillna(0) >= i_imb_min_for_bypass) |
        (out["r_spread"] >= r_spread_factor)
    )

    # ========== تجميع Suspected ==========
    suspected = (~confirmed) & (suspected_pattern | model_promote_to_suspected)

    # ========== Model Anomaly (شذوذ نموذج فقط) ==========
    # إذا شذوذ نموذج لكن بدون دعم قواعد/مؤشرات كافية (أو بدون حمل قوي)
    if "is_anomaly" in out.columns:
        model_only = (out["is_anomaly"] == 1) & (~confirmed) & (~suspected)
    else:
        model_only = pd.Series(False, index=out.index)

    out["reason_suspected_pattern"] = suspected_pattern.astype(int)
    out["reason_suspected_model_promote"] = model_promote_to_suspected.astype(int)
    out["reason_model_anomaly_only"] = model_only.astype(int)

    out["confirmed_loss"] = confirmed.astype(int)
    out["suspected_loss"] = suspected.astype(int)
    out["model_anomaly"]  = model_only.astype(int)

    out["final_label"] = np.select(
        [out["confirmed_loss"] == 1, out["suspected_loss"] == 1, out["model_anomaly"] == 1],
        ["Confirmed Loss", "Suspected Loss", "Model Anomaly"],
        default="Normal"
    )

    return out

# ===================== واجهة Streamlit =====================
st.set_page_config(page_title="ASDCT • IF + OCSVM + Smart Fusion", layout="wide")
st.title("نظام اكتشاف حالات الفاقد المحتملة — عدادات CT (دمج ذكي قواعد + نماذج)")

with st.sidebar:
    st.header("إعدادات النموذج")
    target_rate = st.slider("النسبة المستهدفة للشذوذ (%)", 1, 20, 5, 1) / 100.0
    st.caption("تحدد تقريبًا نسبة السجلات المصنفة كشذوذ عبر العتبة.")

    st.markdown("---")
    st.header("إعدادات قواعد الفاقد (Confirmed)")
    v_zero_pct = st.slider("نسبة الجهد لاعتباره ≈ صفر", 0.02, 0.30, 0.10, 0.01)
    i_min_for_loss = st.slider("حد التيار الدال على حمل (A)", 0.1, 50.0, 1.0, 0.1)
    i_near_zero_thr = st.slider("تيار يعتبر ≈ صفر (A)", 0.00, 2.0, 0.05, 0.01)
    i_sum_min = st.slider("حد مجموع التيار لإلغاء الضجيج (A)", 0.0, 50.0, 0.5, 0.1)

    low_v_phases_for_confirm = st.selectbox("Confirmed إذا عدد فازات الجهد المنخفض ≥", [1,2,3], index=1)
    i_imb_confirm_thr = st.slider("عدم اتزان تيار شديد (i_imb ≥) لتأكيد الفاقد", 0.50, 3.00, 1.50, 0.05)

    st.markdown("---")
    st.header("إعدادات فاقد محتمل (Suspected / Bypass)")
    v_imb_max_for_bypass = st.slider("اتزان الجهد (v_imb ≤)", 0.01, 0.25, 0.08, 0.01)
    i_imb_min_for_bypass = st.slider("عدم اتزان التيار (i_imb ≥)", 0.10, 2.50, 0.50, 0.05)
    weak_phase_ratio = st.slider("طور ضعيف: min/max ≤", 0.05, 0.90, 0.40, 0.05)
    r_spread_factor = st.slider("تشتيت نسب I/V (≥)", 1.1, 8.0, 1.8, 0.1)
    bypass_promote_thr = st.slider("ترقية شذوذ النموذج إلى Suspected إذا bypass_score ≥", 0.10, 0.95, 0.60, 0.05)

    st.markdown("---")
    st.subheader("ملفات النماذج (جذر المشروع)")
    st.code(str(PATH_IF))
    st.code(str(PATH_SVM))
    st.code(str(PATH_SCALER))

tab_infer, tab_help = st.tabs(["📈 تحليل ملف", "ℹ️ مساعدة"])

with tab_infer:
    st.subheader("رفع ملف Excel للتحليل")
    uploaded = st.file_uploader(
        "أعمدة مطلوبة: Meter Number, V1, V2, V3, A1, A2, A3",
        type=["xlsx"]
    )

    if not PATH_SCALER.exists() or not PATH_IF.exists():
        st.error("يجب توفر الملفات: scaler.joblib و isolation_forest.joblib في نفس مجلد app.py.")
    else:
        if uploaded is not None:
            try:
                df = pd.read_excel(uploaded)
                df.columns = [c.strip() for c in df.columns]
                validate_columns(df, [ID_COL] + FEATURE_COLS)

                # تنظيف وتحويل
                df_infer = df.dropna(subset=FEATURE_COLS).copy().reset_index(drop=True)
                X = df_infer[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

                # تحميل النماذج وتحويل الميزات
                models = load_models()
                scaler = models["scaler"]
                model_if = models["if"]
                model_svm = models.get("svm")

                Xs = scaler.transform(X.values)

                # درجات النماذج
                s_if = normalize_score(-model_if.score_samples(Xs))
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

                # تجميع النتائج التفصيلية (مزامنة الفهارس)
                detailed = df_infer.loc[X.index, [ID_COL] + FEATURE_COLS].copy()
                detailed["score_if"] = s_if
                if s_svm is not None:
                    detailed["score_svm"] = s_svm
                detailed["score_ensemble"] = s_ens
                detailed["is_anomaly"] = flags

                # مؤشرات + قواعد + دمج ذكي
                detailed = compute_signal_features(
                    detailed,
                    i_near_zero_thr=i_near_zero_thr
                )

                detailed = apply_rules_and_fusion(
                    detailed,
                    v_zero_pct=v_zero_pct,
                    i_min_for_loss=i_min_for_loss,
                    i_near_zero_thr=i_near_zero_thr,
                    i_sum_min=i_sum_min,
                    low_v_phases_for_confirm=int(low_v_phases_for_confirm),
                    i_imb_confirm_thr=i_imb_confirm_thr,
                    v_imb_max_for_bypass=v_imb_max_for_bypass,
                    i_imb_min_for_bypass=i_imb_min_for_bypass,
                    weak_phase_ratio=weak_phase_ratio,
                    r_spread_factor=r_spread_factor,
                    bypass_promote_thr=bypass_promote_thr
                )

                # ملخص لكل عداد: عدد السجلات وتصنيفات نهائية
                summary = detailed.groupby(ID_COL, as_index=False).agg(
                    rows=("final_label", "size"),
                    confirmed=("confirmed_loss", "sum"),
                    suspected=("suspected_loss", "sum"),
                    model_anomaly=("model_anomaly", "sum"),
                    anomalies=("is_anomaly", "sum"),
                    max_bypass_score=("bypass_score", "max"),
                    max_model_score=("score_ensemble", "max"),
                    max_i=("I_max", "max"),
                    max_vimb=("v_imb", "max"),
                    max_iimb=("i_imb", "max"),
                )

                # تصنيف نهائي للعداد (أولوية Confirmed ثم Suspected ثم Model Anomaly)
                def meter_label(row):
                    if row["confirmed"] > 0:
                        return "Confirmed Loss"
                    if row["suspected"] > 0:
                        return "Suspected Loss"
                    if row["model_anomaly"] > 0:
                        return "Model Anomaly"
                    return "Normal"

                summary["meter_final_label"] = summary.apply(meter_label, axis=1)

                # ترتيب
                summary = summary.sort_values(
                    by=["meter_final_label", "confirmed", "suspected", "max_bypass_score", "max_model_score"],
                    ascending=[True, False, False, False, False]
                ).reset_index(drop=True)

                # KPIs
                c_confirmed = int((detailed["final_label"] == "Confirmed Loss").sum())
                c_suspected = int((detailed["final_label"] == "Suspected Loss").sum())
                c_modelanom = int((detailed["final_label"] == "Model Anomaly").sum())
                c_normal    = int((detailed["final_label"] == "Normal").sum())

                st.success(
                    f"تم تحليل {detailed.shape[0]} سجلًّا باستخدام {used_models}. "
                    f"شذوذ النموذج: {int(flags.sum())} (≈ {100*flags.mean():.2f}%)."
                )

                k1,k2,k3,k4 = st.columns(4)
                k1.metric("Confirmed Loss (سجلات)", c_confirmed)
                k2.metric("Suspected Loss (سجلات)", c_suspected)
                k3.metric("Model Anomaly (سجلات)", c_modelanom)
                k4.metric("Normal (سجلات)", c_normal)

                st.caption(f"العتبة (score_ensemble): {thr:.4f}")

                # فلاتر عرض
                st.markdown("---")
                flt = st.multiselect(
                    "فلترة النتائج التفصيلية حسب التصنيف النهائي",
                    ["Confirmed Loss", "Suspected Loss", "Model Anomaly", "Normal"],
                    default=["Confirmed Loss", "Suspected Loss", "Model Anomaly"]
                )
                detailed_view = detailed[detailed["final_label"].isin(flt)].copy()

                # عرض وتنزيل
                render_table(summary, "ملخّص العدادات (Meter Summary)", "summary_asdct_fusion.xlsx")
                render_table(detailed_view, "النتائج التفصيلية (Filtered)", "detailed_asdct_fusion.xlsx")

                # عرض أهم الأسباب لمساعدة الفني
                st.markdown("---")
                st.subheader("أسباب القرار (مساعدة للفحص الميداني)")
                reason_cols = [
                    "reason_confirm_v0_with_i",
                    "reason_confirm_many_v_low",
                    "reason_confirm_i_near_zero",
                    "reason_confirm_extreme_iimb",
                    "reason_suspected_pattern",
                    "reason_suspected_model_promote",
                    "reason_model_anomaly_only"
                ]
                reasons_sum = detailed[reason_cols].sum().sort_values(ascending=False).reset_index()
                reasons_sum.columns = ["Reason", "Count"]
                st.dataframe(reasons_sum, use_container_width=True)

            except Exception as e:
                st.exception(e)

with tab_help:
    st.markdown("""
### كيف تُتخذ القرارات (المنهج الجديد)؟
نعتمد **دمج ذكي (Fusion)** بين:
- **نتائج النماذج** (IF + OCSVM إن وجد)
- **قواعد كهربائية** (قواعد حاسمة + قواعد مشتبه)

#### 1) Confirmed Loss (أولوية مطلقة)
تصنّف الحالة Confirmed إذا وُجد حمل ومع أحد الأدلة القاطعة:
- **V≈0 مع I** على نفس الفازة (قص جهد/تلاعب/VT)
- **2 أو 3 فازات جهد منخفض جدًا + حمل**
- **فازة تيارها ≈0 مع وجود تيار قوي في فازة أخرى** (غير منطقي لعداد CT)
- **عدم اتزان تيار شديد جدًا**

#### 2) Suspected Loss (جنابر/Bypass)
جهد متزن نسبيًا + تيارات غير متزنة + طور ضعيف + تشتت كبير I/V + حمل.
كما يتم **ترقية شذوذ النموذج** إلى Suspected عند وجود دعم من المؤشرات أو bypass_score.

#### 3) Model Anomaly
النموذج اكتشف شذوذًا لكن بدون دليل قوي من القواعد (للمراجعة — ليس حكم فاقد مباشر).

#### 4) Normal
لا يوجد ما يدعم فاقد أو شذوذ قوي.

### شكل ملف الإدخال
- Meter Number, V1, V2, V3, A1, A2, A3
""")

st.markdown("---")
st.markdown("👨‍💻 **تطوير :** مشهور العباس 2026 | 00966553339838 | (نسخة مطوّرة: Fusion Rules+Models)")
