"""
HbA1c Test Validation API
==========================
REST API for validating HbA1c test results and detecting blood disorders
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from hba1c_validation_model import (
    ClinicalDecisionSupport,
    generate_synthetic_training_data
)
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)

# =========================
# Utility: Convert NumPy Types
# =========================
def convert_numpy_types(obj):
    """
    Recursively convert numpy data types to native Python types
    so they can be JSON serialized.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# =========================
# Initialize ML Models
# =========================
cds = ClinicalDecisionSupport()

print("Initializing ML models...")
training_data = generate_synthetic_training_data(1000)
cds.initialize_models(training_data)
print("Models ready!")


# =========================
# Health Check
# =========================
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'HbA1c Validation API',
        'models_loaded': True
    })


# =========================
# Validate HbA1c Endpoint
# =========================
@app.route('/api/validate-hba1c', methods=['POST'])
def validate_hba1c():
    try:
        patient_data = request.get_json()

        if not patient_data:
            return jsonify({'success': False, 'error': 'No patient data provided'}), 400

        required_fields = ['patient_id', 'hba1c', 'fasting_glucose', 'haemoglobin']
        missing_fields = [field for field in required_fields if field not in patient_data]

        if missing_fields:
            return jsonify({'success': False, 'error': f'Missing fields: {", ".join(missing_fields)}'}), 400

        result = cds.assess_test_result(patient_data)
        safe_result = convert_numpy_types(result)

        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'assessment': safe_result
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =========================
# Detect Anomaly
# =========================
@app.route('/api/detect-anomaly', methods=['POST'])
def detect_anomaly():
    try:
        patient_data = request.get_json()
        result = cds.anomaly_detector.detect_anomaly(patient_data)
        safe_result = convert_numpy_types(result)

        return jsonify({
            'success': True,
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'anomaly_detection': safe_result
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =========================
# Predict Disorder
# =========================
@app.route('/api/predict-disorder', methods=['POST'])
def predict_disorder():
    try:
        patient_data = request.get_json()
        result = cds.disorder_classifier.predict_disorder(patient_data)
        safe_result = convert_numpy_types(result)

        return jsonify({
            'success': True,
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'disorder_prediction': safe_result
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =========================
# Correct HbA1c
# =========================
@app.route('/api/correct-hba1c', methods=['POST'])
def correct_hba1c():
    try:
        patient_data = request.get_json()
        result = cds.hba1c_corrector.predict_corrected_hba1c(patient_data)
        safe_result = convert_numpy_types(result)

        return jsonify({
            'success': True,
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'correction': safe_result
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =========================
# Batch Validate
# =========================
@app.route('/api/batch-validate', methods=['POST'])
def batch_validate():
    try:
        data = request.get_json()
        patients = data.get('patients', [])

        if not patients:
            return jsonify({'success': False, 'error': 'No patient data provided'}), 400

        results = []

        for patient_data in patients:
            try:
                result = cds.assess_test_result(patient_data)
                safe_result = convert_numpy_types(result)
                results.append({'success': True, 'assessment': safe_result})
            except Exception as e:
                results.append({
                    'success': False,
                    'patient_id': patient_data.get('patient_id', 'unknown'),
                    'error': str(e)
                })

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


# =========================
# Run Server
# =========================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("HbA1c Test Validation API Server")
    print("=" * 70)
    print("\nServer starting on http://localhost:5000")
    print("=" * 70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
