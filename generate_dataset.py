import numpy as np
import pandas as pd
from pathlib import Path

def generate_dataset(n=2500, seed=42, output_path="data/bus_occupancy_synthetic.csv"):
    rng = np.random.default_rng(seed)
    capacity = rng.choice([40, 45, 50, 55, 60], size=n)
    hour = rng.integers(5, 24, size=n)
    day_type = rng.choice(["Weekday", "Weekend"], size=n, p=[0.72, 0.28])
    peak = (((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 20))).astype(int)
    ratio = np.clip(rng.beta(2.2, 2.0, size=n) + peak*0.15
                    + np.where(day_type == "Weekend", -0.08, 0)
                    + rng.normal(0, 0.08, n), 0.02, 1.25)
    passengers = np.clip(np.round(ratio * capacity), 0, capacity + 10).astype(int)
    thermal_signal = np.clip(passengers * 0.85 + rng.normal(0, 3.0, n), 0, None).round(2)
    prev_ratio = np.clip(ratio + rng.normal(0, 0.12, n), 0, 1.25)
    previous = np.clip(np.round(prev_ratio * capacity), 0, capacity + 10).astype(int)
    next_count = np.clip(np.round(passengers + rng.normal(0, 5, n) + np.where(peak == 1, 2, -1)),
                         0, capacity + 10).astype(int)
    occupancy_ratio = passengers / capacity
    status = np.where(occupancy_ratio < 0.35, "Empty",
             np.where(occupancy_ratio < 0.75, "Moderate", "Full"))
    data = pd.DataFrame({
        "bus_id": [f"BUS-{i%25+1:02d}" for i in range(n)],
        "route_id": [f"ROUTE-{i%12+1:02d}" for i in range(n)],
        "passenger_count": passengers, "bus_capacity": capacity, "hour": hour,
        "day_type": day_type, "previous_passenger_count": previous,
        "thermal_signal": thermal_signal, "occupancy_ratio": occupancy_ratio.round(3),
        "next_stop_passenger_count": next_count, "occupancy_status": status
    })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data

if __name__ == "__main__":
    df = generate_dataset()
    print(f"Saved {len(df)} rows to data/bus_occupancy_synthetic.csv")
    print(df.head())
