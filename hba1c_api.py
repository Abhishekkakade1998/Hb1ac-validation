from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ==========================================================
# Utility: Expected HbA1c from Glucose
# ==========================================================
def expected_hba1c(glucose):
    return (glucose + 46.7) / 28.7


# ==========================================================
# Core Clinical Logic Engine
# ==========================================================
def assess_patient(data):

    hba1c = float(data["hba1c"])
    glucose = float(data["fasting_glucose"])
    hb = float(data["haemoglobin"])
    mcv = float(data["mcv"])
    ferritin = float(data["ferritin"])
    bilirubin = float(data["bilirubin"])
    lifespan = float(data["rbc_lifespan_days"])

    exp_hba1c = expected_hba1c(glucose)
    difference = hba1c - exp_hba1c

    disorder = "none"
    confidence = 0.7
    corrected = hba1c
    reliable = True
    recommendation = "No further action required"

    # ------------------------------------------------------
    # Iron Deficiency
    # ------------------------------------------------------
    if hb < 11 and ferritin < 30 and difference > 0.8:
        disorder = "iron_deficiency"
        corrected = exp_hba1c
        reliable = False
        confidence = 0.85

    # ------------------------------------------------------
    # Thalassemia
    # ------------------------------------------------------
    elif hb < 12 and mcv < 70 and ferritin >= 50 and difference < -0.8:
        disorder = "thalassemia"
        corrected = exp_hba1c
        reliable = False
        confidence = 0.80

    # ------------------------------------------------------
    # Sickle Cell
    # ------------------------------------------------------
    elif hb < 10 and lifespan < 80 and bilirubin > 1.2:
        disorder = "sickle_cell"
        corrected = exp_hba1c
        reliable = False
        confidence = 0.78

    # ------------------------------------------------------
    # G6PD
    # ------------------------------------------------------
    elif hb < 11 and bilirubin > 1.8 and lifespan < 90:
        disorder = "g6pd"
        corrected = exp_hba1c
        reliable = False
        confidence = 0.82

    # ------------------------------------------------------
    # Extreme Mismatch (Anomaly)
    # ------------------------------------------------------
    elif abs(difference) > 2:
        disorder = "anomaly"
        corrected = exp_hba1c
        reliable = False
        recommendation = "Repeat test / Investigate anomaly"
        confidence = 0.75

    # ------------------------------------------------------
    # Final Output
    # ------------------------------------------------------
    if disorder != "none" and disorder != "anomaly":
        recommendation = "Confirm blood disorder diagnosis"

    return {
        "patient_id": data["patient_id"],
        "test_validity": {
            "is_reliable": reliable
        },
        "hba1c_values": {
            "measured_hba1c": round(hba1c, 2),
            "expected_hba1c": round(exp_hba1c, 2),
            "corrected_hba1c": round(corrected, 2)
        },
        "disorder_assessment": {
            "predicted_disorder": disorder,
            "confidence": confidence
        },
        "clinical_recommendation": recommendation,
        "timestamp": datetime.now().isoformat()
    }


# ==========================================================
# API Endpoints
# ==========================================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Minimal HbA1c Clinical API"
    })


@app.route("/api/validate-hba1c", methods=["POST"])
def validate():

    data = request.get_json()

    required = [
        "patient_id",
        "hba1c",
        "fasting_glucose",
        "haemoglobin",
        "mcv",
        "ferritin",
        "bilirubin",
        "rbc_lifespan_days"
    ]

    missing = [f for f in required if f not in data]

    if missing:
        return jsonify({
            "success": False,
            "error": f"Missing fields: {', '.join(missing)}"
        }), 400

    result = assess_patient(data)

    return jsonify({
        "success": True,
        "result": result
    })


# ==========================================================
# Run
# ==========================================================
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
