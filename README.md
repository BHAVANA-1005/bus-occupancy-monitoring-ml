# Real-Time Bus Occupancy Monitoring and Prediction

**Team:** TECH FUSIONISTS  
**Team lead:** Samyuktha P  
**Team member:** Bhavana S  
**Institution:** Panimalar Engineering College, Chennai  
**PPT date:** 17 October 2024

## Project overview
This project demonstrates the TECH FUSIONISTS presentation concept: monitor bus occupancy, classify buses as **Empty**, **Moderate**, or **Full**, compare a Decision Tree with a Random Forest, and display simulated live predictions.

### Features
- Reproducible synthetic dataset for immediate testing
- Decision Tree and Random Forest classifiers
- Accuracy, classification report, and confusion matrix
- Simulated live bus occupancy dashboard
- Streamlit UI for changing passenger count, capacity, and time
- CSV export of the generated dataset

## Important limitation
The presentation does not include a real dataset or physical sensor readings. The included CSV is **synthetic demonstration data**, not real thermal-sensor data. `thermal_signal` is simulated. Reported metrics measure performance on this synthetic dataset and must not be presented as validated real-world accuracy.

## Run in Google Colab
1. Open `Bus_Occupancy_ML_Colab.ipynb` in Google Colab.
2. Run **Runtime → Run all**.
3. The notebook creates/loads the dataset, trains both models, prints metrics, plots results, and demonstrates a simulated live prediction.
4. Download the notebook or CSV from the Colab Files panel if needed.

## Run locally
```bash
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
streamlit run app.py
```
Open the local URL printed by Streamlit.

## Dataset columns
| Column | Meaning |
|---|---|
| `bus_id`, `route_id` | Simulated identifiers |
| `passenger_count` | Current passenger count |
| `bus_capacity` | Nominal bus capacity |
| `hour` | Hour of day (5–23) |
| `day_type` | Weekday or Weekend |
| `previous_passenger_count` | Simulated previous-stop count |
| `thermal_signal` | Simulated sensor-like signal; not measured temperature |
| `occupancy_ratio` | Passenger count divided by capacity |
| `next_stop_passenger_count` | Simulated next-stop passenger count |
| `occupancy_status` | Label derived from occupancy ratio |

## Classification rules used to generate demonstration labels
- Empty: occupancy ratio below 0.35
- Moderate: occupancy ratio from 0.35 to below 0.75
- Full: occupancy ratio 0.75 or above

## Suggested future improvements
Connect actual thermal sensors and validate passenger counting, gather labeled data across routes and seasons, add time-series forecasting, deploy a real mobile app/notification service, and add an operator scheduling dashboard.

## GitHub
Suggested repository name: `bus-occupancy-monitoring-ml`
