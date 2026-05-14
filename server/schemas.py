from pydantic import BaseModel

class CarResponse(BaseModel):

    found: bool

    plate_number: str

    owner_name: str

    rating: float

    votes: int

    warning: str | None = None

    class Config:
        from_attributes = True