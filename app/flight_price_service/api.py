from flask import Flask, request, jsonify
from src.predict import FlightPricePredictor
from shared.logger import logger

app = Flask(__name__)

# Load model once when the application starts
try:
    predictor = FlightPricePredictor()
    logger.info("Flight Price Predictor loaded successfully.")
except Exception as e:
    predictor = None
    logger.error(f"Failed to load model: {e}")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Flight Price Prediction API",
        "status": "Running",
        "version": "1.0.0"
    })


@app.route("/health", methods=["GET"])
def health():
    if predictor is None:
        return jsonify({
            "status": "Unhealthy",
            "message": "Model not loaded"
        }), 500

    return jsonify({
        "status": "Healthy"
    })


@app.route("/predict", methods=["POST"])
def predict():

    if predictor is None:
        return jsonify({
            "error": "Prediction model is unavailable."
        }), 500

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Request body must contain JSON data."
            }), 400

        required_fields = [
            "from",
            "to",
            "flightType",
            "agency",
            "time",
            "distance",
            "date"
        ]

        missing = [field for field in required_fields if field not in data]

        if missing:
            return jsonify({
                "error": f"Missing fields: {missing}"
            }), 400

        prediction = predictor.predict(data)

        return jsonify({
            "success": True,
            "predicted_price": round(prediction, 2)
        })

    except Exception as e:
        logger.exception("Prediction failed")

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )