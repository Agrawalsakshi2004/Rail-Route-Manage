from django.shortcuts import render
from .ml_model import get_all_trains, predict_future_delays, track_trains, df


# STATIONS (autocomplete)
def get_stations():
    stations = sorted(
        df["station_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )
    return stations


def home(request):

    stations = get_stations()
    trains = []
    selected_train = None
    prediction = None
    tracking = []

    if request.method == "POST":

        source = request.POST.get("source").strip().lower()
        destination = request.POST.get("destination").strip().lower()

        # STEP 1 → trains (ML file se)
        trains = get_all_trains(source, destination)

        # STEP 2 → selected train
        selected_train = request.POST.get("selected_train")

        # STEP 3 → recursive ML
        if selected_train:
            train_df = df[df["journey_id"] == selected_train]

            prediction = predict_future_delays(train_df, source, destination)

            tracking = track_trains(source, destination, selected_train)

    return render(request, "index.html", {
        "stations": stations,
        "trains": trains,
        "selected_train": selected_train,
        "prediction": prediction,
        "tracking": tracking
    })
