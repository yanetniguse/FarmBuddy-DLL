import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
# =============================
# App Configuration
# =============================
st.set_page_config(
    page_title="Farm Buddy 🌱",
    layout="wide"
)

# =============================
# Load Model
# =============================
@st.cache_resource
def load_model():
    return joblib.load("yield_rf_pipeline.joblib")

model = load_model()

# =============================
# Sidebar – Inputs
# =============================
# -----------------------------
# Initialize Session State
# -----------------------------
defaults = {
    "crop": "Maize",
    "season": "Kharif",
    "state": "Tamil Nadu",
    "year": 2020,
    "area": 1.0,
    "fertilizer": 50.0,
    "pesticide": 5.0,
    "current_prediction": 0.0,
    "decision_history": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.sidebar.header("🌾 Farm Inputs")

st.session_state.crop = st.sidebar.selectbox(
    "Crop",
    ["Rice", "Maize", "Cotton(lint)", "Coconut", "Groundnut"],
    index=["Rice", "Maize", "Cotton(lint)", "Coconut", "Groundnut"].index(st.session_state.crop),
    key="crop_input"
)

st.session_state.season = st.sidebar.selectbox(
    "Season",
    ["Kharif", "Rabi", "Whole Year"],
    key="season_input"
)

st.session_state.state = st.sidebar.selectbox(
    "State",
    ["Tamil Nadu", "Andhra Pradesh", "Karnataka", "Kerala"],
    key="state_input"
)

st.session_state.year = st.sidebar.slider(
    "Year",
    2000, 2025,
    st.session_state.year,
    key="year_input"
)

st.session_state.area = st.sidebar.slider(
    "Area (hectares)",
    0.5, 20.0,
    st.session_state.area,
    step=0.5,
    key="area_input"
)

st.session_state.fertilizer = st.sidebar.slider(
    "Fertilizer (kg)",
    0.0, 500.0,
    st.session_state.fertilizer,
    key="fertilizer_input"
)

st.session_state.pesticide = st.sidebar.slider(
    "Pesticide (kg)",
    0.0, 200.0,
    st.session_state.pesticide,
    key="pesticide_input"
)


# =============================
# Prepare Input for Model
# =============================
input_df = pd.DataFrame([{
    "crop": st.session_state.crop,
    "year": st.session_state.year,
    "season": st.session_state.season,
    "state": st.session_state.state,
    "area": st.session_state.area,
    "fertilizer": st.session_state.fertilizer,
    "pesticide": st.session_state.pesticide,
    "production": 0
}])


st.session_state.current_prediction = model.predict(input_df)[0]
prediction = st.session_state.current_prediction


# =============================
# Tabs Layout
# =============================
# In your existing app, add new tab:
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "📊 Prediction",
    "📈 Scenario Comparison",
    "📘 Decision Log",
    "🧠 Learning Summary"
])

# =============================
# TAB 1 — HOME
# =============================
with tab1:
    st.title("Farm Buddy")
    st.subheader("Learning Agriculture Through Simple AI")

    st.markdown("""
    Farming success depends on **many factors working together**.
    This tool helps farmers and students **see, learn, and understand**
    how everyday decisions affect crop yield.

    You don’t need technical knowledge,just adjust the inputs and learn by doing.
    """)

    st.divider()

    st.subheader("🌾 What affects crop yield?")
    factors = {
        "Factor": [
            "Land Size",
            "Fertilizer Use",
            "Crop Type",
            "Season",
            "Weather & Region",
            "Pesticide Use"
        ],
        "Impact Level": [5, 4, 4, 3, 3, 2]
    }

    df_factors = pd.DataFrame(factors).set_index("Factor")
    st.bar_chart(df_factors)

    st.markdown("""
    **How to understand this chart:**
    - Taller bars mean **bigger influence**
    - More land and good fertilizer help most
    - Too much pesticide may not help
    """)
with tab2:
    st.title("Farm Buddy – Crop Yield Predictor")
    st.caption("AI-powered crop yield estimation for learning & planning")

    # -----------------------------
    # Load Model
    # -----------------------------
    @st.cache_resource
    def load_model():
        return joblib.load("yield_rf_pipeline.joblib")

    model = load_model()
    st.success("Model loaded successfully")

    # -----------------------------
    # Sidebar – Farm Inputs
    # -----------------------------
    
    # Predict
    prediction = model.predict(input_df)[0]

    # -----------------------------
    # Main Dashboard
    # -----------------------------
    st.subheader("📈 Predicted Yield")
    st.metric(
    label=f"Estimated Yield for {st.session_state.crop} "
          f"({st.session_state.season}, {st.session_state.state})",
    value=f"{prediction:.2f} units"
    )


    st.markdown(f"""
    **What this means:**  

    - If you plant **1 hectare** of {st.session_state.crop} under these conditions, you can expect to harvest about **{prediction:.2f} tonnes/tons**.  
    - Planting more land with the same care can increase total harvest proportionally.  
    - Adjusting fertilizer, pesticide, or choosing the right season may increase yield further.  

    This is an **AI-powered estimate**, helping you plan and make better decisions on your farm.
    """)

    # -----------------------------
    # Educational Explanation
    # -----------------------------
    st.divider()
    st.subheader("How Your Inputs Affect Yield")
    st.markdown(f"""
    - **Crop type:** Different crops have different potential yields. For example, {st.session_state.crop} tends to have {'higher' if st.session_state.crop in ['Maize','Rice'] else 'lower'} yield in your region.
    - **Season:** {st.session_state.season} season affects rainfall and growth conditions.
    - **State/Region:** Soil quality, climate, and local practices in {st.session_state.state} influence crop productivity.
    - **Area:** Yield generally scales with area, but efficient farming practices matter more than size alone.
    - **Fertilizer:** Using {st.session_state.fertilizer} kg can increase growth, but too much can harm plants or the soil.
    - **Pesticide:** {st.session_state.pesticide} kg helps protect crops, but excessive use may reduce effectiveness.
    - **Year:** Models learn from historical trends; recent years may reflect improved practices or climate changes.
    """)

    st.info("Try changing the inputs to see how different choices affect crop yield and understand the cause-effect relationship.")

    # -----------------------------
    # Optional: Show input summary
    # -----------------------------
    st.divider()
    st.subheader("🌾 Input Summary")
    st.table(input_df)
with tab3:
    st.subheader("Personalized Visualization")

    # Example bar chart for predicted yield
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar([f"{st.session_state.crop} Yield"], [prediction], color='lightgreen')
    ax.set_ylabel("Yield (tons/ha)")
    ax.set_title(f"Predicted Yield for {st.session_state.crop} ({st.session_state.season}, {st.session_state.state})")
    st.pyplot(fig)

    # Explanation in plain language
    st.markdown(f"""
    **💡 Insight for your scenario:**  
    - Using **{st.session_state.fertilizer} kg** of fertilizer on **{st.session_state.area} ha** of {st.session_state.crop} gives an estimated yield of **{prediction:.2f} tons/ha**  
    - Crop type and season strongly influence yield  
    - Proper pesticide management prevents losses, overuse may reduce effectiveness  
    - Experiment with sliders to see how yield changes dynamically
    """)
    st.subheader("📊 What Really Affects Yield?")

    st.markdown("""
    This section shows **how sensitive the predicted yield is**  
    when we change one farming decision at a time.
    """)

    import pandas as pd
    import matplotlib.pyplot as plt

    base_row = {
        "crop": st.session_state.crop,
        "year": st.session_state.year,
        "season": st.session_state.season,
        "state": st.session_state.state,
        "area": st.session_state.area,
        "fertilizer": st.session_state.fertilizer,
        "pesticide": st.session_state.pesticide,
        "production": 0
    }

    base_yield = model.predict(pd.DataFrame([base_row]))[0]

    scenarios = {
        "Increase Area (+30%)": {**base_row, "area": st.session_state.area * 1.3},
        "Increase Fertilizer (+30%)": {**base_row, "fertilizer": st.session_state.fertilizer * 1.3},
        "Increase Pesticide (+30%)": {**base_row, "pesticide": st.session_state.pesticide * 1.3},
    }

    impacts = {}
    for name, row in scenarios.items():
        pred = model.predict(pd.DataFrame([row]))[0]
        impacts[name] = pred - base_yield

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(impacts.keys(), impacts.values())
    ax.axhline(0)
    ax.set_ylabel("Change in Predicted Yield")
    ax.set_title("Impact of Changing One Input")

    st.pyplot(fig)

    st.markdown("""
    **How to read this chart (very important):**
    - Bars near **zero** → changing this input does **not help much**
    - Taller bars → this decision has **more impact**
    - If fertilizer is flat → adding more won’t increase yield
    """)

    st.info("""
    📌 **Key Learning**
    If an input shows little change, it means the **data does not support**
    that it strongly affects yield in this scenario.
    This helps avoid unnecessary costs.
    """)


with tab4:
        st.header("Decision & Learning Log")

        st.subheader("Current Situation")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Crop", st.session_state.crop)
            st.metric("Season", st.session_state.season)
            st.metric("Fertilizer", f"{st.session_state.fertilizer} kg")

        with col2:
            st.metric(
                "Current Predicted Yield",
                f"{st.session_state.current_prediction:.2f} tons/ha"
            )

        st.divider()

        st.subheader("Log Your Decision")

        decision_type = st.radio(
            "What are you changing?",
            ["Increase fertilizer", "Reduce fertilizer", "Add pesticide",
            "Increase area", "Change crop", "Other"]
        )

        reasoning = st.text_area(
            "Why are you making this decision?",
            height=150,
            help="Explain your thinking clearly."
        )

        if st.button("💾 Save Decision", type="primary"):
            if len(reasoning.split()) < 30:
                st.error("Please explain your reasoning (at least 30 words).")
            else:
                st.session_state.decision_history.append({
                    "time": datetime.now(),
                    "decision": decision_type,
                    "reasoning": reasoning,
                    "yield_before": st.session_state.current_prediction
                })
                st.success("Decision saved. Now change inputs and observe results.")

        if st.session_state.decision_history:
            st.subheader("📊 Decision History")

            for i, d in enumerate(reversed(st.session_state.decision_history[-3:])):
                with st.expander(f"Decision {len(st.session_state.decision_history)-i}: {d['decision']}"):
                    st.write(f"🕒 {d['time'].strftime('%Y-%m-%d %H:%M')}")
                    st.info(d["reasoning"])

                    change = st.session_state.current_prediction - d["yield_before"]
                    st.metric(
                        "Yield Change",
                        f"{st.session_state.current_prediction:.2f} tons/ha",
                        delta=f"{change:+.2f}"
                    )

with tab5:
    st.header("Learning Summary")
    st.write("This section automatically summarizes what you have learned from your decisions.")

    if len(st.session_state.decision_history) < 1:
        st.info("Make at least one decision to generate a learning summary.")
    else:
        history = st.session_state.decision_history
        first = history[0]
        last = history[-1]

        initial_prediction = first.get(
            "prediction_before",
            first.get("yield_before", st.session_state.current_prediction)
        )

        current_prediction = st.session_state.current_prediction
        yield_change = current_prediction - initial_prediction

        changes = set(d["decision"] for d in history)

        st.subheader("Key Observations")
        st.markdown(f"""
- **Initial predicted yield:** {initial_prediction:.2f} tons/ha  
- **Current predicted yield:** {current_prediction:.2f} tons/ha  
- **Net yield change:** {yield_change:+.2f} tons/ha  
""")

        st.subheader("What Changed")
        for c in changes:
            st.write(f"• {c}")

        conclusion = (
            "led to an improvement in predicted yield"
            if yield_change > 0 else
            "resulted in a decrease in predicted yield"
            if yield_change < 0 else
            "did not significantly affect the predicted yield"
        )

        st.subheader("Auto-Generated Learning Summary")
        st.markdown(f"""
**What the system learned from your decisions:**

Your sequence of decisions **{conclusion}**.
This suggests that your current configuration is
**{'more optimal' if yield_change > 0 else 'near optimal'}**
according to the trained model.
""")

        st.success("✨ Learning summary generated successfully.")
