from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bus Occupancy Monitor", page_icon="🚌", layout="wide")
st.title("🚌 Real-Time Bus Occupancy Monitoring")
st.caption("TECH FUSIONISTS | Decision Tree + Random Forest | Simulated demonstration")
st.warning("Demo only: sensor-like values and dataset are synthetic. No physical thermal sensor or live bus feed is connected.")

DATA = Path("data/bus_occupancy_synthetic.csv")
MODEL_PATH = Path("models/random_forest.joblib")
if not DATA.exists():
    st.error("Dataset not found. Run `python generate_dataset.py` first.")
    st.stop()
if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()

df = pd.read_csv(DATA)
model = joblib.load(MODEL_PATH)
left, right = st.columns([1, 1])
with left:
    st.subheader("Simulate a bus")
    bus_id = st.text_input("Bus ID", "BUS-01")
    route_id = st.text_input("Route ID", "ROUTE-01")
    capacity = st.slider("Bus capacity", 20, 80, 50)
    passengers = st.slider("Current passengers", 0, capacity + 10, min(28, capacity + 10))
    hour = st.slider("Hour of day", 5, 23, 8)
    day_type = st.selectbox("Day type", ["Weekday", "Weekend"])
    previous = st.slider("Previous-stop passenger count", 0, capacity + 10, min(24, capacity + 10))
    thermal = st.number_input("Simulated thermal signal", min_value=0.0, value=float(passengers * 0.85), step=0.5)
    if st.button("Predict occupancy", type="primary"):
        row = pd.DataFrame([{
            "passenger_count": passengers, "bus_capacity": capacity, "hour": hour,
            "day_type": day_type, "previous_passenger_count": previous,
            "thermal_signal": thermal
        }])
        status = model.predict(row)[0]
        ratio = passengers / capacity
        st.session_state["result"] = (status, ratio, bus_id, route_id, passengers, capacity)
with right:
    st.subheader("Occupancy result")
    if "result" in st.session_state:
        status, ratio, saved_bus, saved_route, p_count, cap = st.session_state["result"]
        st.metric("Predicted status", status)
        st.metric("Occupancy ratio", f"{ratio:.0%}")
        st.progress(min(ratio, 1.0))
        st.write(f"**Bus:** {saved_bus}  |  **Route:** {saved_route}")
        st.write(f"**Passengers:** {p_count}/{cap}")
        if status == "Full":
            st.error("High occupancy: consider another bus if available.")
        elif status == "Moderate":
            st.info("Moderate occupancy.")
        else:
            st.success("Lower occupancy.")
    else:
        st.info("Choose values and click **Predict occupancy**.")
st.divider()
st.subheader("Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Synthetic records", f"{len(df):,}")
c2.metric("Simulated buses", df["bus_id"].nunique())
c3.metric("Routes", df["route_id"].nunique())
st.bar_chart(df["occupancy_status"].value_counts())
st.dataframe(df.head(10), use_container_width=True)
st.download_button("Download synthetic dataset CSV", df.to_csv(index=False), "bus_occupancy_synthetic.csv", "text/csv")
