"""
HbA1c Validation API
====================
Flask API for validating HbA1c test results, detecting blood disorders,
and predicting corrected HbA1c values.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import threading
import numpy as np

from hba1c_validation_model import ClinicalDecisionSupport, generate_synthetic_training_data

# -----------------------------
# JSON provider to handle numpy types
# -----------------------------
from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
app.json = NumpyJSONProvider(app)
CORS(app)

# -----------------------------
# ML model state
# -----------------------------
cds = None
models_ready = False
models_error = None

def load_models():
    """Train ML models in background to avoid blocking server startup."""
    global cds, models_ready, models_error
    try:
        print("Initializing ML models in background...")
        training_data = generate_synthetic_training_data(1000)
        instance = ClinicalDecisionSupport()
        instance.initialize_models(training_data)
        cds = instance
        models_ready = True
        print("✓ Models ready!")
    except Exception as e:
        models_error = str(e)
        print(f"Model initialization failed: {e}")

# Start background thread
threading.Thread(target=load_models, daemon=True).start()

def require_models():
    """Return an error response if models aren't ready yet."""
    if models_error:
        return jsonify({'success': False, 'error': f'Model initialization failed: {models_error}'}), 500
    if not models_ready:
        return jsonify({'success': False, 'error': 'Models are still initializing. Please retry in a few seconds.'}), 503
    return None

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "HbA1c Validation API running",
        "health_check": "/api/health",
        "validate_endpoint": "/api/validate-hba1c"
    })

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        'status': 'healthy' if models_ready else 'initializing',
        'models_loaded': models_ready,
        'timestamp': datetime.now().isoformat(),
        'service': 'HbA1c Validation API'
    })

@app.route("/api/validate-hba1c", methods=["POST"])
def validate_hba1c():
    err = require_models()
    if err:
        return err
    try:
        patient_data = request.get_json()
        if not patient_data:
            return jsonify({'success': False, 'error': 'No patient data provided'}), 400

        required_fields = ['patient_id', 'hba1c', 'fasting_glucose', 'haemoglobin']
        missing = [f for f in required_fields if f not in patient_data]
        if missing:
            return jsonify({'success': False, 'error': f'Missing required fields: {", ".join(missing)}'}), 400

        result = cds.assess_test_result(patient_data)
        return jsonify({'success': True, 'timestamp': datetime.now().isoformat(), 'assessment': result})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/api/model-info", methods=["GET"])
def model_info():
    if not models_ready:
        return jsonify({'success': False, 'error': 'Models still initializing'}), 503
    return jsonify({
        'success': True,
        'models': {
            'anomaly_detector': {'trained': cds.anomaly_detector.is_trained, 'type': 'Isolation Forest'},
            'disorder_classifier': {
                'trained': cds.disorder_classifier.is_trained,
                'type': 'Random Forest Classifier',
                'categories': ['none', 'iron_deficiency', 'thalassemia', 'sickle_cell', 'g6pd']
            },
            'hba1c_corrector': {'trained': cds.hba1c_corrector.is_trained, 'type': 'Gradient Boosting Regressor'}
        },
        'training_data_size': 1000
    })

@app.route("/api/example-request", methods=["GET"])
def example_request():
    example = {
        "patient_id": "P12345",
        "hba1c": 7.2,
        "fasting_glucose": 120,
        "random_glucose": 140,
        "ogtt_2hr": 160,
        "avg_glucose_cgm": 125,
        "haemoglobin": 9.5,
        "rbc_count": 4.2,
        "mcv": 75,
        "mch": 25,
        "mchc": 32,
        "reticulocyte_count": 0.8,
        "wbc_count": 6.5,
        "platelet_count": 280,
        "serum_iron": 30,
        "ferritin": 12,
        "transferrin_saturation": 15,
        "tibc": 450,
        "bilirubin": 0.6,
        "ldh": 140,
        "haptoglobin": 100,
        "age": 35,
        "gender": "F",
        "disorder": "iron_deficiency",
        "rbc_lifespan_days": 90
    }
    required = ['patient_id', 'hba1c', 'fasting_glucose', 'haemoglobin']
    return jsonify({
        'example_request': example,
        'required_fields': required,
        'optional_fields': list(set(example.keys()) - set(required))
    })

# -----------------------------
# Run app
# -----------------------------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
