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
    "📘 Decision Log",
    "📈 Scenario Comparison",
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
    st.subheader("📘 Decision & Learning Log")

    # -------------------------------
    # Current Inputs Table
    # -------------------------------
    current_data = {
        "Parameter": ["Crop", "Season", "State", "Year", "Area (ha)", "Fertilizer (kg)", "Pesticide (kg)", "Current Predicted Yield (tons/ha)"],
        "Value": [
            st.session_state.crop,
            st.session_state.season,
            st.session_state.state,
            st.session_state.year,
            f"{st.session_state.area:.2f}",
            f"{st.session_state.fertilizer:.2f}",
            f"{st.session_state.pesticide:.2f}",
            f"{st.session_state.current_prediction:.2f}"
        ]
    }
    st.table(pd.DataFrame(current_data))

    st.divider()

    # -------------------------------
    # Log a Decision
    # -------------------------------
    st.subheader("📝 Log Your Decision")

    decision_type = st.radio(
        "What are you changing?",
        [
            "Increase fertilizer",
            "Reduce fertilizer",
            "Add pesticide",
            "Increase area",
            "Change crop",
            "Other",
        ]
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
                "inputs": {
                    "crop": st.session_state.crop,
                    "season": st.session_state.season,
                    "state": st.session_state.state,
                    "year": st.session_state.year,
                    "area": st.session_state.area,
                    "fertilizer": st.session_state.fertilizer,
                    "pesticide": st.session_state.pesticide,
                },
                "prediction_snapshot": float(st.session_state.current_prediction),
            }

            st.session_state.decision_history.append(decision_snapshot)
            st.success("Decision saved. Now adjust inputs to see the impact.")

    # -------------------------------
    # If no decisions yet
    # -------------------------------
    if not st.session_state.decision_history:
        st.info("Save at least one decision to see Before/After comparison.")
        st.stop()

    # -------------------------------
    # Decision History (last 3)
    # -------------------------------
    st.subheader("📊 Recent Decisions")

    for i, d in enumerate(reversed(st.session_state.decision_history[-3:])):
        with st.expander(
            f"Decision {len(st.session_state.decision_history)-i}: {d['decision_type']}"
        ):
            st.write(f"🕒 {d['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            st.info(d["reasoning"])

            before = d["prediction_snapshot"]

            # Prepare current inputs for dynamic "after" prediction
            model_columns = ["crop", "season", "state", "year", "area", "fertilizer", "pesticide", "production"]
            current_inputs = {
                "crop": st.session_state.crop,
                "season": st.session_state.season,
                "state": st.session_state.state,
                "year": st.session_state.year,
                "area": st.session_state.area,
                "fertilizer": st.session_state.fertilizer,
                "pesticide": st.session_state.pesticide,
                "production": 0,  # placeholder
            }

            input_df = pd.DataFrame([current_inputs], columns=model_columns)
            after = float(model.predict(input_df)[0])

            delta = after - before

            st.metric("Yield at Decision Time", f"{before:.2f} tons/ha")
            st.metric("Current Yield", f"{after:.2f} tons/ha", delta=f"{delta:+.2f} tons/ha")

    # -------------------------------
    # Before vs After Chart (Latest Decision)
    # -------------------------------
    st.subheader("📈 Yield Change From Latest Decision")

    last_decision = st.session_state.decision_history[-1]

    before = last_decision["prediction_snapshot"]

    # Current "after" prediction using current inputs
    current_inputs = {
        "crop": st.session_state.crop,
        "season": st.session_state.season,
        "state": st.session_state.state,
        "year": st.session_state.year,
        "area": st.session_state.area,
        "fertilizer": st.session_state.fertilizer,
        "pesticide": st.session_state.pesticide,
        "production": 0,
    }

    input_df = pd.DataFrame([current_inputs], columns=model_columns)
    after = float(model.predict(input_df)[0])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(["Before Decision", "After Change"], [before, after], color=['skyblue', 'lightgreen'])
    ax.set_ylabel("Predicted Yield (tons/ha)")
    ax.set_title("Before vs After Yield Comparison")
    for i, v in enumerate([before, after]):
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom')

    st.pyplot(fig)
    st.caption(f"Decision logged on {last_decision['timestamp'].strftime('%Y-%m-%d %H:%M')}")


# --------------------------
# Tab 4: Decision & Learning Log
# --------------------------
with tab4:
    st.subheader("📈 Before vs After Comparison")

    if not st.session_state.decision_history:
        st.info("Log at least one decision to see comparisons.")
    else:
        # Labels for selection
        decision_labels = [
            f"{i+1}. {d['decision_type']} ({d['timestamp'].strftime('%Y-%m-%d %H:%M')})"
            for i, d in enumerate(st.session_state.decision_history)
        ]
        selected_index = st.selectbox(
        "Select a decision to analyze",
        range(len(decision_labels)),
        format_func=lambda i: decision_labels[i],
        key="tab5_learning_decision_select"
)


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
# Tab 5: Learning Summary
# --------------------------
with tab5:
    st.subheader("Learning Summary & Insights")

    history = st.session_state.decision_history

    if len(history) < 1:
        st.info("Log at least one decision to generate insights.")
        st.stop()

    # -------------------------------
    # Overall Yield Change
    # -------------------------------
    initial_yield = history[0]["prediction_snapshot"]
    current_yield = st.session_state.current_prediction
    total_delta = current_yield - initial_yield

    st.metric(
        label="Initial Predicted Yield",
        value=f"{initial_yield:.2f} tons/ha"
    )
    st.metric(
        label="Current Predicted Yield",
        value=f"{current_yield:.2f} tons/ha",
        delta=f"{total_delta:+.2f} tons/ha"
    )

    st.markdown("---")

    # -------------------------------
    # Decision Analysis Table
    # -------------------------------
    analysis_rows = []
    for idx, d in enumerate(history, 1):
        before = d["prediction_snapshot"]
        after = d.get("prediction_after", current_yield)
        delta = after - before

        # Determine which inputs changed
        changed_inputs = [
            k for k, v in d["inputs"].items()
            if v != st.session_state.__dict__.get(k, v)
        ]
        changed_inputs_str = ", ".join(changed_inputs) if changed_inputs else "N/A"

        analysis_rows.append({
            "Decision #": idx,
            "Type": d["decision_type"],
            "Inputs Changed": changed_inputs_str,
            "Yield Before (t/ha)": f"{before:.2f}",
            "Yield After (t/ha)": f"{after:.2f}",
            "Δ Yield": f"{delta:+.2f}"
        })

    st.subheader("📋 Decision Impact Table")
    st.table(pd.DataFrame(analysis_rows))

    st.markdown("---")

    # -------------------------------
    # Key Learnings
    # -------------------------------
    st.subheader("💡 Key Learnings & Recommendations")

    # Identify inputs with most impact across all decisions
    impact_counter = {}
    for d in history:
        before = d["prediction_snapshot"]
        after = d.get("prediction_after", current_yield)
        delta = after - before
        if delta != 0:
            for k, v in d["inputs"].items():
                if k not in impact_counter:
                    impact_counter[k] = 0
                impact_counter[k] += abs(delta)

    # Sort inputs by total impact
    sorted_impacts = sorted(impact_counter.items(), key=lambda x: x[1], reverse=True)

    if sorted_impacts:
        st.write("The inputs that most influenced predicted yield were:")
        for k, v in sorted_impacts:
            st.markdown(f"• **{k.capitalize()}** — impact score: {v:.2f}")
    else:
        st.write("No input had a significant impact yet.")

    # -------------------------------
    # Personalized Insights per Decision
    # -------------------------------
    st.subheader("🔍 Insights per Decision")

    for i, d in enumerate(history, 1):
        before = d["prediction_snapshot"]
        after = d.get("prediction_after", current_yield)
        delta = after - before

        if delta > 0:
            outcome = "✅ This decision increased predicted yield"
        elif delta < 0:
            outcome = "❌ This decision decreased predicted yield"
        else:
            outcome = "⚠️ This decision had minimal effect"

        st.markdown(f"**Decision {i}: {d['decision_type']}**")
        st.info(d["reasoning"])
        st.write(f"- Yield Before: {before:.2f} t/ha")
        st.write(f"- Yield After: {after:.2f} t/ha")
        st.write(f"- Δ Yield: {delta:+.2f} t/ha")
        st.success(outcome)
        st.divider()

    # -------------------------------
    # Overall Recommendations
    # -------------------------------
    st.subheader("🎯 Overall Recommendations")

    if total_delta > 0:
        st.markdown(f"""
- Your decisions so far **improved the predicted yield** by {total_delta:.2f} t/ha.
- Focus on decisions that showed the **highest impact** (see key inputs above).
- Inputs that had minimal impact can be deprioritized to save cost/resources.
- Keep experimenting with combinations of high-impact factors for optimal results.
""")
    elif total_delta < 0:
        st.markdown(f"""
- Your decisions so far **reduced the predicted yield** by {abs(total_delta):.2f} t/ha.
- Review which inputs caused negative impacts and adjust accordingly.
- Avoid repeating low-impact or counterproductive changes.
- Prioritize high-impact inputs based on the analysis table.
""")
    else:
        st.markdown("""
- Overall, your decisions did not significantly change the predicted yield.
- This suggests current inputs are close to optimal.
- Focus on exploring **other parameters** or **combinations** to see meaningful improvements.
""")

    st.caption("This summary helps you reflect, understand the impact of your decisions, and plan future actions for better yield outcomes.")
