import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Khởi tạo server và load mô hình .sav
app = FastAPI(title="Dự đoán giá nhà API")

with open("house_price_model.sav", "rb") as f:
    model = pickle.load(f)


# 2. Định nghĩa cấu trúc dữ liệu Mobile gửi lên
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


@app.get("/")
def home():
    return {"message": "Server AI Dự đoán giá nhà đang hoạt động!"}


# 3. API nhận data từ Mobile -> Trả về giá tiền
@app.post("/predict")
def predict_price(data: HouseInput):
    # Khớp đúng tên cột như lúc huấn luyện
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

    # Dự đoán
    prediction = model.predict(df_input)[0]

    return {
        "status": "success",
        "predicted_price": round(float(prediction), 2),  # Tỷ VNĐ
    }