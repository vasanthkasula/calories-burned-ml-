import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(
    page_title="Calories Burned Predictor",
    page_icon="🔥",
    layout="wide",
)

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    with open("df.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
df    = load_data()

st.markdown("# 🔥 Calories Burned Predictor")
st.markdown("Predict calories burned during exercise — powered by **XGBoost** (R² = 0.9995)")
st.divider()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Training Records", "15,000")
k2.metric("Model Accuracy (R²)", "99.95%")
k3.metric("Mean Abs. Error", "~1 cal")
k4.metric("Best Model", "XGBoost")

st.divider()

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("📋 Your Details")
    gender   = st.radio("Gender", ["Male", "Female"], horizontal=True)
    age      = st.slider("Age (years)",             10,  100,  30)
    height   = st.slider("Height (cm)",            140,  220, 170)
    weight   = st.slider("Weight (kg)",             30,  150,  70)
    duration = st.slider("Exercise Duration (min)",  1,   60,  20)
    hr       = st.slider("Heart Rate (bpm)",        60,  200, 100)
    temp     = st.slider("Body Temperature (°C)", 36.0, 42.0, 40.0, step=0.1)

with right:
    st.subheader("🎯 Prediction")

    gender_enc = 0 if gender == "Male" else 1
    row = pd.DataFrame(
        [[gender_enc, age, height, weight, duration, hr, temp]],
        columns=["Gender","Age","Height","Weight","Duration","Heart_Rate","Body_Temp"]
    )
    cal = float(model.predict(row)[0])

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#FF6B35,#FF9F1C);
                    border-radius:16px;padding:2rem;text-align:center;margin-bottom:1rem;">
            <p style="color:rgba(255,255,255,.8);margin:0;font-size:1rem;">Estimated Calories Burned</p>
            <h1 style="color:white;font-size:4rem;margin:0.2rem 0;">{cal:.1f}</h1>
            <p style="color:rgba(255,255,255,.8);margin:0;">kcal</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pct = float((df["Calories"] < cal).mean() * 100)
    st.progress(int(pct), text=f"You burn more than **{pct:.0f}%** of users in the dataset")

    if pct >= 75:
        st.success("🚀 High Burn — great effort!")
    elif pct >= 40:
        st.warning("💪 Moderate Burn — solid workout!")
    else:
        st.info("🧘 Light Activity — perfect for recovery!")

    st.markdown("**🍕 Equivalent to burning off:**")
    fa, fb = st.columns(2)
    fa.metric("🍎 Apples (52 cal)",        max(0, int(cal / 52)))
    fb.metric("🍕 Pizza slices (285 cal)",  max(0, int(cal / 285)))
    fc, fd = st.columns(2)
    fc.metric("🍫 Chocolates (55 cal)",     max(0, int(cal / 55)))
    fd.metric("🥤 Sodas (105 cal)",         max(0, int(cal / 105)))

st.divider()
st.subheader("📊 Dataset Insights")

tab1, tab2, tab3 = st.tabs(["Calories Distribution", "Feature Correlations", "Calories vs Duration"])

with tab1:
    st.bar_chart(df["Calories"].value_counts().sort_index(), use_container_width=True)

with tab2:
    numeric_cols = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp","Calories","Gender"]
    corr = df[numeric_cols].corr().round(2)
    # colour cells manually without matplotlib
    def color_corr(val):
        v = float(val)
        if v > 0.7:   bg = "#1a5c1a"; color = "white"
        elif v > 0.4: bg = "#2d7a2d"; color = "white"
        elif v > 0:   bg = "#c8e6c9"; color = "black"
        elif v > -0.4:bg = "#ffcdd2"; color = "black"
        else:          bg = "#b71c1c"; color = "white"
        return f"background-color:{bg}; color:{color}"
    st.dataframe(corr.style.applymap(color_corr), use_container_width=True)

with tab3:
    sample = df.sample(min(2000, len(df)), random_state=42)[["Duration","Calories"]].rename(
        columns={"Duration":"Duration (min)","Calories":"Calories burned"}
    )
    st.scatter_chart(sample, x="Duration (min)", y="Calories burned", use_container_width=True)

st.divider()
st.subheader("🏆 Model Comparison")
results = pd.DataFrame({
    "Model":               ["XGBoost","SVR","Gradient Boosting","Random Forest","Ridge","Linear Regression"],
    "R² Score":            [0.9995,   0.9996, 0.9993,           0.9983,         0.9845,  0.9845],
    "MAE (cal)":           [1.01,     0.55,   1.21,             1.67,           5.81,    5.83],
    "RMSE":                [1.49,     1.24,   1.71,             2.62,           7.91,    7.90],
    "Accuracy (±10 cal)":  ["99.9%","99.8%","99.9%",           "98.9%",        "85.6%","85.4%"],
}).set_index("Model")

st.dataframe(results, use_container_width=True)

st.caption("Built by Vasanth · XGBoost · Scikit-learn · Streamlit · 15,000 exercise records")
