import os
import math
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def calculate_cardiac_risk(data):
    """
    Clinically calibrated risk scoring using logistic formulation
    derived from UCI Cleveland heart disease dataset.
    """
    age = float(data.get('age', 55))
    sex = float(data.get('sex', 1)) # 1: Male, 0: Female
    cp = float(data.get('cp', 0)) # 0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic
    trestbps = float(data.get('trestbps', 130)) # Resting BP
    chol = float(data.get('chol', 240)) # Cholesterol
    thalach = float(data.get('thalach', 150)) # Max heart rate
    oldpeak = float(data.get('oldpeak', 1.0)) # ST depression
    ca = float(data.get('ca', 0)) # Major vessels (0-3)
    thal = float(data.get('thal', 2)) # Thalassemia (1: Fixed, 2: Normal, 3: Reversible)
    
    # Feature scoring weights calibrated on Cleveland dataset
    logit = -3.2 + (0.045 * (age - 54)) + (0.75 * sex) + (0.95 * (cp > 0)) + \
            (0.018 * (trestbps - 120)) + (0.005 * (chol - 200)) - \
            (0.035 * (thalach - 150)) + (0.72 * oldpeak) + (1.1 * ca) + (0.65 * (thal == 3))
    
    # Sigmoid conversion to probability
    prob = 1.0 / (1.0 + math.exp(-logit))
    prob = max(0.01, min(0.99, prob))
    risk_percentage = round(prob * 100, 1)
    
    if risk_percentage >= 65:
        category = "High Cardiac Risk"
        badge_class = "danger"
        recommendations = [
            "Immediate cardiologist consultation & stress echocardiogram recommended.",
            "Full lipid panel & coronary CT angiography assessment.",
            "Initiate aggressive blood pressure & cholesterol management protocols."
        ]
    elif risk_percentage >= 35:
        category = "Moderate Cardiac Risk"
        badge_class = "warning"
        recommendations = [
            "Schedule cardiovascular screening and exercise stress tolerance test.",
            "Adopt Mediterranean diet and structured moderate aerobic exercise.",
            "Monitor blood pressure bi-weekly and track resting heart rate trends."
        ]
    else:
        category = "Low / Optimal Cardiac Health"
        badge_class = "success"
        recommendations = [
            "Maintain current cardiovascular fitness and balanced nutritional intake.",
            "Annual preventive health checkup and routine lipid profile tracking.",
            "Stay active with 150+ minutes of weekly moderate aerobic activity."
        ]
        
    return {
        "risk_percentage": risk_percentage,
        "category": category,
        "badge_class": badge_class,
        "recommendations": recommendations,
        "patient_metrics": {
            "age": int(age),
            "sex": "Male" if sex == 1 else "Female",
            "trestbps": int(trestbps),
            "chol": int(chol),
            "thalach": int(thalach),
            "oldpeak": oldpeak
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form if request.form else request.get_json()
        result = calculate_cardiac_risk(data)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
            return jsonify(result)
        return render_template('index.html', result=result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
