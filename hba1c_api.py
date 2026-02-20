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
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Initialize the clinical decision support system
cds = ClinicalDecisionSupport()

# Train with synthetic data on startup (in production, use real data)
print("Initializing ML models...")
training_data = generate_synthetic_training_data(1000)
cds.initialize_models(training_data)
print("Models ready!")

# =========================
# Patient Entry Web Page
# =========================
@app.route('/', methods=['GET'])
def patient_entry_page():
    """Patient Entry Web Interface"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HbA1c Validation Tool</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f9f9f9; }
        h1 { color: #2c3e50; }
        input[type="text"], input[type="number"] {
            width: 200px; padding: 6px; margin: 5px;
        }
        button {
            padding: 10px 20px;
            margin-top: 10px;
            cursor: pointer;
            background: #3498db;
            color: white;
            border: none;
        }
        button:hover { background: #2980b9; }
        .result {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ccc;
            background: #fff;
            border-radius: 5px;
        }
        pre { white-space: pre-wrap; word-wrap: break-word; }
    </style>
</head>
<body>
    <h1>HbA1c Validation Tool</h1>

    <label>Patient ID:</label><br>
    <input type="text" id="patient_id" value="P001"><br>

    <label>HbA1c (%):</label><br>
    <input type="number" step="0.1" id="hba1c" value="7.2"><br>

    <label>Fasting Glucose (mg/dL):</label><br>
    <input type="number" id="fasting_glucose" value="120"><br>

    <label>Haemoglobin (g/dL):</label><br>
    <input type="number" step="0.1" id="haemoglobin" value="9.5"><br>

    <button onclick="validateHbA1c()">Validate</button>

    <div class="result" id="result" style="display:none;">
        <h3>Validation Result</h3>
        <pre id="resultContent"></pre>
    </div>

<script>
async function validateHbA1c() {
    const patientData = {
        patient_id: document.getElementById('patient_id').value,
        hba1c: parseFloat(document.getElementById('hba1c').value),
        fasting_glucose: parseFloat(document.getElementById('fasting_glucose').value),
        haemoglobin: parseFloat(document.getElementById('haemoglobin').value)
    };

    try {
        const response = await fetch('/api/validate-hba1c', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patientData)
        });

        const result = await response.json();
        document.getElementById('result').style.display = 'block';
        document.getElementById('resultContent').textContent =
            JSON.stringify(result, null, 2);

    } catch (err) {
        document.getElementById('result').style.display = 'block';
        document.getElementById('resultContent').textContent =
            'Error: ' + err;
    }
}
</script>

</body>
</html>
    """)

# =========================
# Health Check
# =========================
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
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
        
        # Validate required fields
        required_fields = ['patient_id', 'hba1c', 'fasting_glucose', 'haemoglobin']
        missing_fields = [field for field in required_fields if field not in patient_data]
        if missing_fields:
            return jsonify({'success': False, 'error': f'Missing fields: {", ".join(missing_fields)}'}), 400
        
        # Comprehensive assessment using CDS
        result = cds.assess_test_result(patient_data)
        
        return jsonify({'success': True, 'timestamp': datetime.now().isoformat(), 'assessment': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# =========================
# Detect Anomaly
# =========================
@app.route('/api/detect-anomaly', methods=['POST'])
def detect_anomaly():
    try:
        patient_data = request.get_json()
        anomaly_result = cds.anomaly_detector.detect_anomaly(patient_data)
        return jsonify({'success': True, 'patient_id': patient_data.get('patient_id', 'unknown'), 'anomaly_detection': anomaly_result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# =========================
# Predict Disorder
# =========================
@app.route('/api/predict-disorder', methods=['POST'])
def predict_disorder():
    try:
        patient_data = request.get_json()
        disorder_result = cds.disorder_classifier.predict_disorder(patient_data)
        return jsonify({'success': True, 'patient_id': patient_data.get('patient_id', 'unknown'), 'disorder_prediction': disorder_result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# =========================
# Correct HbA1c
# =========================
@app.route('/api/correct-hba1c', methods=['POST'])
def correct_hba1c():
    try:
        patient_data = request.get_json()
        correction_result = cds.hba1c_corrector.predict_corrected_hba1c(patient_data)
        return jsonify({'success': True, 'patient_id': patient_data.get('patient_id', 'unknown'), 'correction': correction_result})
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
                results.append({'success': True, 'assessment': result})
            except Exception as e:
                results.append({'success': False, 'patient_id': patient_data.get('patient_id', 'unknown'), 'error': str(e)})
        
        unreliable_count = sum(1 for r in results if r.get('success') and not r['assessment']['test_validity']['is_reliable'])
        
        return jsonify({'success': True, 'timestamp': datetime.now().isoformat(), 'total_patients': len(patients), 'processed': len(results), 'unreliable_tests': unreliable_count, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# =========================
# Model Info
# =========================
@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'success': True,
        'models': {
            'anomaly_detector': {
                'trained': cds.anomaly_detector.is_trained,
                'type': 'Isolation Forest',
                'purpose': 'Detect unreliable HbA1c results'
            },
            'disorder_classifier': {
                'trained': cds.disorder_classifier.is_trained,
                'type': 'Random Forest Classifier',
                'purpose': 'Classify blood disorders',
                'categories': ['none', 'iron_deficiency', 'thalassemia', 'sickle_cell', 'g6pd']
            },
            'hba1c_corrector': {
                'trained': cds.hba1c_corrector.is_trained,
                'type': 'Gradient Boosting Regressor',
                'purpose': 'Predict corrected HbA1c values'
            }
        },
        'training_data_size': 1000
    })

# =========================
# Example Request
# =========================
@app.route('/api/example-request', methods=['GET'])
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
    return jsonify({
        'example_request': example,
        'required_fields': ['patient_id', 'hba1c', 'fasting_glucose', 'haemoglobin'],
        'optional_fields': list(set(example.keys()) - set(['patient_id', 'hba1c', 'fasting_glucose', 'haemoglobin']))
    })

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
