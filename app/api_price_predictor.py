from flask import Flask, request, render_template_string
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ================= LOAD MODEL & ARTIFACTS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "..", "model", "flight_prediction")

xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_regressor.pkl"))
from_encoder = joblib.load(os.path.join(MODEL_DIR, "from_encoder.pkl"))
to_encoder = joblib.load(os.path.join(MODEL_DIR, "to_encoder.pkl"))
agency_encoder = joblib.load(os.path.join(MODEL_DIR, "agency_encoder.pkl"))
flight_type_map = joblib.load(os.path.join(MODEL_DIR, "flight_type_map.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

# ================= DROPDOWN OPTIONS =================
from_options = from_encoder.categories_[0].tolist()
to_options = to_encoder.categories_[0].tolist()
agency_options = agency_encoder.categories_[0].tolist()
flight_type_options = list(flight_type_map.keys())

# ================= HTML UI =================
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Flight Price Predictor</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(120deg,#1e3c72,#2a5298);
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 700px;
            background: white;
            margin: 60px auto;
            padding: 40px;
            border-radius: 14px;
            box-shadow: 0 25px 45px rgba(0,0,0,0.2);
        }
        h1 {
            text-align: center;
            color: #1e3c72;
        }
        label {
            font-weight: 600;
            margin-top: 15px;
            display: block;
        }
        input, select {
            width: 100%;
            padding: 12px;
            margin-top: 6px;
            border-radius: 6px;
            border: 1px solid #ccc;
        }
        button {
            width: 100%;
            margin-top: 30px;
            padding: 16px;
            background: #1e3c72;
            color: white;
            font-size: 18px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        button:hover {
            background: #16315c;
        }
        .result {
            margin-top: 30px;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            color: #1e3c72;
        }
        .error {
            color: red;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>✈ Flight Price Predictor</h1>

    <form method="POST">
        <label>From</label>
        <select name="from_city">
            {% for c in from_options %}
            <option value="{{ c }}">{{ c }}</option>
            {% endfor %}
        </select>

        <label>To</label>
        <select name="to_city">
            {% for c in to_options %}
            <option value="{{ c }}">{{ c }}</option>
            {% endfor %}
        </select>

        <label>Flight Type</label>
        <select name="flight_type">
            {% for f in flight_type_options %}
            <option value="{{ f }}">{{ f }}</option>
            {% endfor %}
        </select>

        <label>Agency</label>
        <select name="agency">
            {% for a in agency_options %}
            <option value="{{ a }}">{{ a }}</option>
            {% endfor %}
        </select>

        <label>Flight Time (hours)</label>
        <input type="number" step="0.01" name="time" required>

        <label>Distance (km)</label>
        <input type="number" step="0.01" name="distance" required>

        <label>Date</label>
        <input type="date" name="date" required>

        <button type="submit">Predict Price</button>
    </form>

    {% if prediction %}
    <div class="result">
        {{ prediction }}
    </div>
    {% endif %}
</div>
</body>
</html>
"""

# ================= ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            from_city = request.form["from_city"]
            to_city = request.form["to_city"]
            flight_type = request.form["flight_type"]
            agency = request.form["agency"]
            time = float(request.form["time"])
            distance = float(request.form["distance"])

            date = pd.to_datetime(request.form["date"])
            day, month, year = date.day, date.month, date.year

            # ---- Encoding (SAME AS TRAINING) ----
            from_val = from_encoder.transform(pd.DataFrame([[from_city]], columns=["from"]))[0][0]
            to_val = to_encoder.transform(pd.DataFrame([[to_city]], columns=["to"]))[0][0]
            agency_val = agency_encoder.transform(pd.DataFrame([[agency]], columns=["agency"]))[0][0]
            flight_val = flight_type_map[flight_type]

            # ---- Build input in SAME FEATURE ORDER ----
            X = pd.DataFrame([{
                "from": from_val,
                "to": to_val,
                "flightType": flight_val,
                "time": time,
                "distance": distance,
                "agency": agency_val,
                "day": day,
                "month": month,
                "year": year
            }])

            X = X[feature_columns]

            price = int(xgb_model.predict(X)[0])
            prediction = f"Predicted Price: ₹ {price}"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template_string(
        HTML,
        prediction=prediction,
        from_options=from_options,
        to_options=to_options,
        agency_options=agency_options,
        flight_type_options=flight_type_options
    )

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

