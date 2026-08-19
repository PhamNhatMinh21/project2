import pickle
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="House Price AI App")

# 1. Mount thư mục static chứa file HTML
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Load model .sav
with open("house_price_model.sav", "rb") as f:
    model = pickle.load(f)


# 3. Định nghĩa schema nhận dữ liệu
class HouseInput(BaseModel):
    Area: float
    Frontage: float
    Access_Road: float
    Floors: float
    Bedrooms: float
    Bathrooms: float
    Legal_status: str = "Have certificate"
    Furniture_state: str = "Full"
    Province: str = "Hà Nội"


# 4. Khi vào link web -> Mở thẳng file index.html giao diện
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


# 5. Endpoint xử lý tính toán giá tiền
@app.post("/predict")
def predict_price(data: HouseInput):
    df_input = pd.DataFrame(
        {
            "Area": [data.Area],
            "Frontage": [data.Frontage],
            "Access Road": [data.Access_Road],
            "Floors": [data.Floors],
            "Bedrooms": [data.Bedrooms],
            "Bathrooms": [data.Bathrooms],
            "Legal status": [data.Legal_status],
            "Furniture state": [data.Furniture_state],
            "Province": [data.Province],
        }
    )

    pred = model.predict(df_input)[0]
    return {"status": "success", "predicted_price": round(float(pred), 2)}