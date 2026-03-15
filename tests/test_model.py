import pandas as pd
# Assuming they wrote a prediction script in src/evaluate_model.py
from src.EvaluateModel import load_model_and_predict

def test_model_loading_and_prediction():
    # 1. Create a dummy patient row that matches the heart failure dataset columns
    dummy_patient = pd.DataFrame({
        'age': [65], 
        'anaemia': [0], 
        'creatinine_phosphokinase': [146], 
        'diabetes': [0], 
        'ejection_fraction': [20], 
        'high_blood_pressure': [1], 
        'platelets': [162000], 
        'serum_creatinine': [1.3], 
        'serum_sodium': [129], 
        'sex': [1], 
        'smoking': [1], 
        'time': [7]
    })
    
    # 2. Run the prediction
    prediction = load_model_and_predict(dummy_patient)
    
    # 3. Verify model loading and prediction  outputs a valid class (0 or 1)
    assert prediction[0] in [0, 1], "Model did not output a valid binary prediction!"
