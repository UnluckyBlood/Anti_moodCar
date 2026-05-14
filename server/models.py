from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy import Text

from database import Base

class Car(Base):

    __tablename__ = "cars"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    plate_number = Column(
        String,
        unique=True,
        nullable=False
    )

    owner_name = Column(String)

    rating = Column(
        Float,
        default=0
    )

    votes = Column(
        Integer,
        default=0
    )

    warning = Column(Text)