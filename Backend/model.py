# models.py
from typing import Optional
from sqlmodel import SQLModel, Field
from datetime import datetime

class DataRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    # Example features - adapt to your dataset
    feature1: Optional[float] = None
    feature2: Optional[float] = None
    label: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class MLMetadata(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    version: str
    trained_at: Optional[datetime] = None
    metrics: Optional[str] = None  # JSON string or small text
