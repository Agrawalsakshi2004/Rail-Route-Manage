import pandas as pd
import joblib
from datetime import datetime, timedelta

# LOAD MODEL
model = joblib.load("train_delay_model.pkl")

# LOAD DATA (ek baar)
df = pd.read_csv("final_train_data.csv")

# CLEAN
df["station_name"] = df["station_name"].astype(str).str.strip().str.lower()
df["scheduled_arrival"] = pd.to_datetime(df["scheduled_arrival"]).dt.time
df["actual_arrival"] = pd.to_datetime(df["actual_arrival"]).dt.time


# ENCODING
def encode_dataframe(df):
    df["day_of_week"] = df["day_of_week"].astype("category").cat.codes
    df["month"] = df["month"].astype("category").cat.codes
    df["time_block"] = df["time_block"].astype("category").cat.codes
    df["train_type"] = df["train_type"].astype("category").cat.codes
    df["season"] = df["season"].astype("category").cat.codes
    return df

df = encode_dataframe(df)


# GET TRAINS
def get_all_trains(source, destination):
    
    trains = []
    
    journeys = df[df["station_name"] == source]["journey_id"].unique()
    
    for jid in journeys:
        temp = df[df["journey_id"] == jid]
        
        if destination not in temp["station_name"].values:
            continue
        
        src_seq = temp[temp["station_name"] == source]["station_sequence"].values[0]
        dest_seq = temp[temp["station_name"] == destination]["station_sequence"].values[0]
        
        if src_seq >= dest_seq:
            continue
        
        row = temp[temp["station_name"] == source].iloc[0]
        
        trains.append({
            "journey_id": jid,
            "time": row["scheduled_arrival"]
        })
    
    return sorted(trains, key=lambda x: x["time"])


# 🔥 CORE ML (RECURSIVE)
def predict_future_delays(train_df, source, destination):

    train_df = train_df.sort_values("station_sequence").reset_index(drop=True)

    src_row = train_df[train_df["station_name"] == source]
    dest_row = train_df[train_df["station_name"] == destination]

    src_seq = int(src_row.iloc[0]["station_sequence"])
    dest_seq = int(dest_row.iloc[0]["station_sequence"])

    route_df = train_df[
        (train_df["station_sequence"] >= src_seq) &
        (train_df["station_sequence"] <= dest_seq)
    ].reset_index(drop=True)

    # 🔥 LIVE CURRENT TIME
    current_time_now = datetime.now().time()

    temp_sorted = route_df.sort_values("station_sequence")

    passed = temp_sorted[
        temp_sorted["scheduled_arrival"] <= current_time_now
    ]

    if passed.empty:
        current_row = temp_sorted.iloc[0]
    else:
        current_row = passed.iloc[-1]

    # 🔥 STATUS LOGIC
    dest_time = route_df.iloc[-1]["scheduled_arrival"]

    if current_time_now >= dest_time:
        status = "Reached Destination"
    elif current_time_now > current_row["scheduled_arrival"]:
        status = f"Left {current_row['station_name']}"
    else:
        status = str(current_row["station_name"])

    # 🔥 DELAY INIT
    current_delay = float(current_row["arrival_delay_min"])
    current_cum_delay = float(current_row["cumulative_delay"])

    current_time = datetime.combine(datetime.today(), current_row["scheduled_arrival"])

    predictions = []

    # 🔥 FUTURE STATIONS ML
    for i in range(1, len(route_df)):

        row = route_df.iloc[i]
        seq = int(row["station_sequence"])

        input_data = pd.DataFrame([{
            "station_sequence": seq,
            "hour": int(current_row["hour"]),
            "day_of_week": int(current_row["day_of_week"]),
            "is_weekend": int(current_row["is_weekend"]),
            "month": int(current_row["month"]),
            "train_type": int(current_row["train_type"]),
            "previous_delay": current_delay,
            "cumulative_delay": current_cum_delay,
            "distance_from_source": seq * 5
        }])

        pred_delay = float(model.predict(input_data)[0])

        if pd.isna(pred_delay):
            pred_delay = 0

        pred_delay = round(pred_delay, 2)

        scheduled_time = datetime.combine(datetime.today(), row["scheduled_arrival"])
        expected_time = scheduled_time + timedelta(minutes=pred_delay)

        predictions.append({
            "station": str(row["station_name"]),
            "scheduled_time": str(scheduled_time.time()),
            "predicted_delay": pred_delay,
            "expected_time": str(expected_time.time())
        })

        current_cum_delay += pred_delay
        current_delay = pred_delay

    # 🔥 RETURN FINAL
    return {
        "current_station": status,
        "current_delay": float(current_row["arrival_delay_min"]),
        "current_scheduled": str(current_row["scheduled_arrival"]),
        "current_expected": str(current_row["actual_arrival"]),
        "next_station": str(route_df.iloc[1]["station_name"]) if len(route_df) > 1 else None,
        "future_predictions": predictions
    }

from datetime import datetime

def track_trains(source, destination, selected_journey):

    current_time = datetime.now().time()
    result = []

    # 🟢 SELECTED TRAIN
    selected_df = df[df["journey_id"] == selected_journey]
    temp_sorted = selected_df.sort_values("station_sequence")

    passed = temp_sorted[
        temp_sorted["scheduled_arrival"] <= current_time
    ]

    if passed.empty:
        current_row = temp_sorted.iloc[0]
    else:
        current_row = passed.iloc[-1]

    dest_time = selected_df[selected_df["station_name"] == destination]["scheduled_arrival"].values[0]

    if current_time >= dest_time:
        status = "Reached Destination"
    elif current_time > current_row["scheduled_arrival"]:
        status = f"Left {current_row['station_name']}"
    else:
        status = str(current_row["station_name"])

    selected_distance = float(current_row["distance_from_source"])

    # ❗ selected ko current tabhi banayenge jab wo actually current ho
    selected_is_current = True

    result.append({
        "type": "selected",
        "train": selected_journey,
        "current_station": status,
        "scheduled_time": str(current_row["scheduled_arrival"]),
        "delay": float(current_row["arrival_delay_min"]),
        "expected_time": str(current_row["actual_arrival"]),
        "is_current": selected_is_current
    })

    # 🔴 OTHER TRAINS
    journeys = df[df["station_name"] == source]["journey_id"].unique()

    prev_distance = selected_distance

    # 🔥 CONTROL → sirf 1 train current hogi
    current_assigned = False

    for jid in journeys:

        if jid == selected_journey:
            continue

        temp = df[df["journey_id"] == jid]

        if destination not in temp["station_name"].values:
            continue

        src_seq = temp[temp["station_name"] == source]["station_sequence"].values[0]
        dest_seq = temp[temp["station_name"] == destination]["station_sequence"].values[0]

        if src_seq >= dest_seq:
            continue

        temp_sorted = temp.sort_values("station_sequence")

        passed = temp_sorted[
            temp_sorted["scheduled_arrival"] <= current_time
        ]

        if passed.empty:
            current_row = temp_sorted.iloc[0]
        else:
            current_row = passed.iloc[-1]

        dest_time = temp[temp["station_name"] == destination]["scheduled_arrival"].values[0]

        if current_time >= dest_time:
            status = "Reached Destination"
        elif current_time > current_row["scheduled_arrival"]:
            status = f"Left {current_row['station_name']}"
        else:
            status = str(current_row["station_name"])

        distance = float(current_row["distance_from_source"])

        # 🔥 ONLY ONE CURRENT TRAIN
        if not current_assigned:
            is_current_flag = True
            current_assigned = True
        else:
            is_current_flag = False

        result.append({
            "type": "other",
            "train": jid,
            "current_station": status,
            "scheduled_time": str(current_row["scheduled_arrival"]),
            "delay": float(current_row["arrival_delay_min"]),
            "actual_time": str(current_row["actual_arrival"]),
            "distance_from_prev": round(abs(distance - prev_distance), 2),
            "is_current": is_current_flag
        })

        prev_distance = distance

    return result