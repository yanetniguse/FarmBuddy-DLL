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
    st.subheader("Crop Yield Predictor")
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

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import copy

# --------------------------
# Tab 3: Before vs After Comparison
# --------------------------
with tab3:
    st.subheader("📈 Before vs After Comparison")

    if not st.session_state.decision_history:
        st.info("Log at least one decision to see comparisons.")
    else:
        # Labels for selection
        decision_labels = [
            f"{i+1}. {d['decision_type']} ({d['timestamp'].strftime('%Y-%m-%d %H:%M')})"
            for i, d in enumerate(st.session_state.decision_history)
        ]
        selected_index = st.selectbox("Select a decision to compare", range(len(decision_labels)),
                                      format_func=lambda i: decision_labels[i])

        selected_decision = st.session_state.decision_history[selected_index]

        # Use frozen snapshots
        before = selected_decision["prediction_snapshot"]
        after = selected_decision.get("prediction_after", st.session_state.current_prediction)

        # Display input comparison table
        before_inputs = selected_decision["inputs"]
        after_inputs = {
            "crop": st.session_state.crop,
            "season": st.session_state.season,
            "state": st.session_state.state,
            "year": st.session_state.year,
            "area": st.session_state.area,
            "fertilizer": st.session_state.fertilizer,
            "pesticide": st.session_state.pesticide,
        }

        comparison_df = pd.DataFrame({
            "Parameter": list(before_inputs.keys()),
            "Before": list(before_inputs.values()),
            "After": list(after_inputs.values())
        })

        st.subheader("🌱 Inputs Before and After")
        st.table(comparison_df)

        # Bar chart for predicted yield
        st.subheader("📊 Predicted Yield Before vs After")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(["Before Decision", "After Change"], [before, after], color=['skyblue','lightgreen'])
        ax.set_ylabel("Predicted Yield (tons/ha)")
        ax.set_title("Impact of Your Decision")

        for i, v in enumerate([before, after]):
            ax.text(i, v, f"{v:.2f}", ha='center', va='bottom')

        st.pyplot(fig)

        delta = after - before
        if abs(delta) < 0.05:
            st.warning("⚠️ This change had little effect. Input likely not limiting.")
        elif delta > 0:
            st.success(f"✅ Your changes increased yield by {delta:.2f} tons/ha.")
        else:
            st.error(f"❌ Your changes reduced yield by {abs(delta):.2f} tons/ha.")

        # Plain language insight
        st.markdown(f"""
**💡 Insight:**  
- Predicted yield before your decision: **{before:.2f} tons/ha**  
- Predicted yield after your decision: **{after:.2f} tons/ha**  
- Net change: **{delta:+.2f} tons/ha**  

Compare the input table above to see which changes had the most impact.
""")

# --------------------------
# Tab 4: Decision & Learning Log
# --------------------------
with tab4:
    st.subheader("🌱 Current Situation")

    current_data = {
        "Parameter": ["Crop", "Season", "Fertilizer (kg)", "Current Predicted Yield (tons/ha)"],
        "Value": [
            st.session_state.crop,
            st.session_state.season,
            f"{st.session_state.fertilizer:.2f}",
            f"{st.session_state.current_prediction:.2f}"
        ]
    }
    st.table(pd.DataFrame(current_data))

    st.divider()
    st.subheader("📝 Log Your Decision")

    decision_type = st.radio(
        "What are you changing?",
        ["Increase fertilizer", "Reduce fertilizer", "Add pesticide", "Increase area", "Change crop", "Other"]
    )

    reasoning = st.text_area(
        "Why are you making this decision?",
        height=150,
        help="Explain your thinking clearly (minimum 30 words)."
    )

    if st.button("💾 Save Decision", type="primary"):
        if len(reasoning.split()) < 30:
            st.error("Please explain your reasoning (at least 30 words).")
        else:
            # Freeze current inputs and prediction
            decision_snapshot = {
                "timestamp": datetime.now(),
                "decision_type": decision_type,
                "reasoning": reasoning,
                "inputs": copy.deepcopy({
                    "crop": st.session_state.crop,
                    "season": st.session_state.season,
                    "state": st.session_state.state,
                    "year": st.session_state.year,
                    "area": st.session_state.area,
                    "fertilizer": st.session_state.fertilizer,
                    "pesticide": st.session_state.pesticide,
                }),
                "prediction_snapshot": float(st.session_state.current_prediction),
                "prediction_after": float(st.session_state.current_prediction)  # will update after inputs change
            }
            st.session_state.decision_history.append(decision_snapshot)
            st.success("Decision saved. Now change inputs to see the impact.")

    st.subheader("📊 Decision History (Last 3)")
    if st.session_state.decision_history:
        for i, d in enumerate(reversed(st.session_state.decision_history[-3:])):
            with st.expander(f"Decision {len(st.session_state.decision_history)-i}: {d['decision_type']}"):
                st.write(f"🕒 {d['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                st.info(d["reasoning"])

                before = d["prediction_snapshot"]
                after = d.get("prediction_after", st.session_state.current_prediction)
                delta = after - before

                st.metric("Yield at decision time", f"{before:.2f} tons/ha")
                st.metric("Current yield", f"{after:.2f} tons/ha", delta=f"{delta:+.2f} tons/ha")

        # Chart for latest decision
        st.subheader("📈 Yield Change From Latest Decision")
        last_decision = st.session_state.decision_history[-1]
        before = last_decision["prediction_snapshot"]
        after = last_decision.get("prediction_after", st.session_state.current_prediction)

        fig, ax = plt.subplots(figsize=(5,4))
        ax.bar(["Before Decision", "After Change"], [before, after], color=['skyblue','lightgreen'])
        ax.set_ylabel("Predicted Yield (tons/ha)")
        ax.set_title("Yield Change")
        for i, v in enumerate([before, after]):
            ax.text(i, v, f"{v:.2f}", ha='center', va='bottom')
        st.pyplot(fig)
        st.caption(f"Decision logged on {last_decision['timestamp'].strftime('%Y-%m-%d %H:%M')}")

# --------------------------
# Tab 5: Learning Summary
# --------------------------
with tab5:
    st.subheader("🧠 Learning Summary")

    history = st.session_state.decision_history
    if not history:
        st.info("Log at least one decision to generate insights.")
    else:
        initial_yield = history[0]["prediction_snapshot"]
        current_yield = st.session_state.current_prediction
        delta_total = current_yield - initial_yield

        st.subheader("Key Results")
        st.markdown(f"""
- **Initial yield:** {initial_yield:.2f} tons/ha  
- **Current yield:** {current_yield:.2f} tons/ha  
- **Net change:** {delta_total:+.2f} tons/ha  
""")

        st.subheader("🔍 What Changed?")
        for i, d in enumerate(history):
            before = d["prediction_snapshot"]
            after = d.get("prediction_after", st.session_state.current_prediction)
            delta = after - before
            changed_inputs = [k for k,v in d["inputs"].items() if v != st.session_state.__dict__.get(k, v)]
            st.markdown(f"**Decision {i+1}: {d['decision_type']}**")
            st.write(f"- Yield change: {delta:+.2f} tons/ha")
            if changed_inputs:
                st.write(f"- Inputs changed: {', '.join(changed_inputs)}")
            st.write(f"- Timestamp: {d['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            st.write("---")

        # Auto insight
        if delta_total > 0:
            outcome = "improved"
        elif delta_total < 0:
            outcome = "reduced"
        else:
            outcome = "did not significantly change"

        st.subheader("Auto-Generated Insight")
        st.markdown(f"""
Your decisions **{outcome}** the predicted yield.

This suggests:
- Not all inputs have equal impact
- Some factors show diminishing returns
- Focus on **high-impact changes** for best yield
""")
