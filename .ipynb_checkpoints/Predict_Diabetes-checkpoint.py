import json
import requests

def predict_diabetes(bmi, age, glucose):
    url = 'http://127.0.0.1:5000/diabetes/v1/predict'
    data = {"BMI": float(bmi), "Age": float(age), "Glucose": float(glucose)}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    return response.json()

if __name__ == "__main__":
    try:
        bmi = input('BMI? ')
        age = input('Age? ')
        glucose = input('Glucose? ')
        
        result = predict_diabetes(bmi, age, glucose)
        label = "Diabetic (Mac tieu duong)" if result["prediction"] == 1 else "Not Diabetic (Khong mac tieu duong)"
        
        print(f"\nResult: {label}")
        print(f"Confidence: {result['confidence']}%")
    except requests.exceptions.ConnectionError:
        print("\n[Loi] Ban chua bat REST_API.py! Hay chay file REST_API.py truoc.")