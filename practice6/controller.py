from fastapi import APIRouter ,Request  

router = APIRouter(prefix="/api")

from service import car_service

@router.post("/receive")
async def receive( car_list: list[dict]):
    return car_service.car_data(car_list)

@router.post("/predict")
async def predict(car: dict):
    return car_service.predict(car)