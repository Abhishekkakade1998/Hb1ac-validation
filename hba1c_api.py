"""
HbA1c Test Validation API
==========================
REST API for validating HbA1c test results and detecting blood disorders
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from hba1c_validation_model import (
    ClinicalDecisionSupport,
    generate_synthetic_training_data
)
from datetime import datetime
import threading

app = Flask(__name__)
CORS(app)

# --- Model state ---
cds = None
models_ready = False
models_error = None

def load_models():
    """Train models in background so the port binds immediately."""
    global cds, models_ready, models_error
    try:
        print("Initializing ML models in background...")
        instance = ClinicalDecisionSupport()
        training_data = generate_synthetic_training_data(1000)
        instance.initialize_models(training_data)
        cds = instance
        models_ready = True
        print("Models ready!")
    except Exception as e:
        models_error = str(e)
        print(f"Model initialization failed: {e}")

# Start background training thread immediately
threading.Thread(target=load_models, daemon=True).start()


def require_models():
    """Return an error response if models aren't ready yet, else None."""
    if models_error:
        return jsonify({'success': False, 'error': f'Model initialization failed: {models_error}'}), 500
    if not models_ready:
        return jsonify({'success': False, 'error': 'Models are still initializing, please retry in 30 seconds.'}), 503
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if models_ready else 'initializing',
        'models_loaded': models_ready,
        'timestamp': datetime.now().isoformat(),
        'service': 'HbA1c Validation API'
    })


@app.route('/api/validate-hba1c', methods=['POST'])
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


@app.route('/api/detect-anomaly', methods=['POST'])
def detect_anomaly():
    err = require_models()
    if err:
        return err
    try:
        patient_data = request.get_json()
        anomaly_result = cds.anomaly_detector.detect_anomaly(patient_data)
        return jsonify({'success': True, 'patient_id': patient_data.get('patient_id', 'unknown'), 'anomaly_detection': anomaly_result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict-disorder', methods=['POST'])
def predict_disorder():
    err = require_models()
    if err:
        return err
    try:
        patient_data = request.get_json()
        disorder_result = cds.disorder_classifier.predict_disorder(patient_data)
        return jsonify({'success': True, 'patient_id': patient_data.get('patient_id', 'unknown'), 'disorder_prediction': disorder_result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/correct-hba1c', methods=['POST'])
def correct_hba1c():
    err = require_models()
    if err:
        return err
    try:
        patient_data = request.get_json()
        correction_result = cds.hba1c_corrector.predict_corrected_hba1c(patient_data)
        return jsonify({'success': True, 'patient_id': patient_data.get('patient_id', 'unknown'), 'correction': correction_result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/batch-validate', methods=['POST'])
def batch_validate():
    err = require_models()
    if err:
        return err
    try:
        data = request.get_json()
        patients = data.get('patients', [])
        if not patients:
            return jsonify({'success': False, 'error': 'No patient data provided'}), 400

        results = []
        for patient_data in patients:
            try:
                result = cds.assess_test_result(patient_data)
                results.append({'success': True, 'assessment': result})
            except Exception as e:
                results.append({'success': False, 'patient_id': patient_data.get('patient_id', 'unknown'), 'error': str(e)})

        unreliable_count = sum(
            1 for r in results
            if r.get('success') and not r['assessment']['test_validity']['is_reliable']
        )
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'total_patients': len(patients),
            'processed': len(results),
            'unreliable_tests': unreliable_count,
            'results': results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
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


@app.route('/api/example-request', methods=['GET'])
def example_request():
    example = {
        "patient_id": "P12345", "hba1c": 7.2, "fasting_glucose": 120,
        "random_glucose": 140, "ogtt_2hr": 160, "avg_glucose_cgm": 125,
        "haemoglobin": 9.5, "rbc_count": 4.2, "mcv": 75, "mch": 25,
        "mchc": 32, "reticulocyte_count": 0.8, "wbc_count": 6.5,
        "platelet_count": 280, "serum_iron": 30, "ferritin": 12,
        "transferrin_saturation": 15, "tibc": 450, "bilirubin": 0.6,
        "ldh": 140, "haptoglobin": 100, "age": 35, "gender": "F",
        "disorder": "iron_deficiency", "rbc_lifespan_days": 90
    }
    required = ['patient_id', 'hba1c', 'fasting_glucose', 'haemoglobin']
    return jsonify({
        'example_request': example,
        'required_fields': required,
        'optional_fields': list(set(example.keys()) - set(required))
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
