from fastapi import FastAPI
from sqlalchemy.orm import Session

from database import SessionLocal, engine
from models import Base, Car
from schemas import CarResponse

Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.get("/")
def root():
    return {
        "message": "AutoMooden API работает"
    }

@app.get(
    "/api/car/{plate}",
    response_model=CarResponse
)
def get_car(plate: str):

    db: Session = SessionLocal()

    try:

        car = db.query(Car).filter(
            Car.plate_number == plate.upper()
        ).first()

        if not car:

            return {
                "found": False,
                "plate_number": plate,
                "owner_name": "",
                "rating": 0,
                "votes": 0,
                "warning": "Номер не найден"
            }

        return {
            "found": True,
            "plate_number": car.plate_number,
            "owner_name": car.owner_name,
            "rating": car.rating,
            "votes": car.votes,
            "warning": car.warning
        }

    finally:
        db.close()