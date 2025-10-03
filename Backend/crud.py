# crud.py
from db import get_session
from models import DataRecord, MLMetadata
from typing import List
import pandas as pd
from sqlmodel import select

def insert_dataframe(df: pd.DataFrame):
    """
    Insert dataframe rows into DataRecord, mapping columns to fields.
    Columns must match DataRecord fields names (feature1, feature2, label).
    """
    session = get_session()
    records = []
    for _, row in df.iterrows():
        rec = DataRecord(
            feature1 = float(row.get("feature1")) if "feature1" in row else None,
            feature2 = float(row.get("feature2")) if "feature2" in row else None,
            label = int(row.get("label")) if "label" in row and not pd.isna(row.get("label")) else None
        )
        session.add(rec)
        records.append(rec)
    session.commit()
    session.refresh(records[-1])
    session.close()
    return len(records)

def fetch_all_data_as_df() -> pd.DataFrame:
    session = get_session()
    q = select(DataRecord)
    results = session.exec(q).all()
    session.close()
    if not results:
        return pd.DataFrame()
    # Convert list of DataRecord to DataFrame
    df = pd.DataFrame([r.dict(exclude_none=True) for r in results])
    # drop DB fields not needed
    df = df.drop(columns=["id", "created_at"], errors="ignore")
    return df

def save_ml_metadata(name: str, version: str, trained_at, metrics: str):
    session = get_session()
    meta = MLMetadata(name=name, version=version, trained_at=trained_at, metrics=metrics)
    session.add(meta)
    session.commit()
    session.refresh(meta)
    session.close()
    return meta
