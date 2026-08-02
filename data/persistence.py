from __future__ import annotations

from sqlalchemy import JSON, DateTime, Float, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class OrderRecord(Base):
    __tablename__ = "orders"
    order_id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    limit_price: Mapped[float] = mapped_column(Float, nullable=False)
    legs: Mapped[list[dict[str, object]]] = mapped_column(JSON, nullable=False)


class FillRecord(Base):
    __tablename__ = "fills"
    order_id: Mapped[str] = mapped_column(String, primary_key=True)
    fill_price: Mapped[float] = mapped_column(Float, nullable=False)
    filled_at: Mapped[str] = mapped_column(DateTime, nullable=False)


def create_session_factory(db_url: str = "sqlite:///./spyder.db") -> sessionmaker[Session]:
    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(engine)
