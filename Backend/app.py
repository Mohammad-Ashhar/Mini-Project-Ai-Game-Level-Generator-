from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import uuid
import json

app = FastAPI(title="Level Generator API")

DB_PATH = "levels.db"

@app.get("/")
def read_root():
    return {"message": "Level Generator API is running"}
# Pydantic model for request
class GenerateRequest(BaseModel):
    difficulty: str
    size: int

# Pydantic model for response
class LevelResponse(BaseModel):
    id: str
    level: dict
    difficulty: str
    size: int


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS levels (
        id TEXT PRIMARY KEY,
        difficulty TEXT,
        size INTEGER,
        level_json TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()


@app.post("/generate", response_model=LevelResponse)
def generate_level(request: GenerateRequest):
    """
    Stub: Pretend ML generates a level.
    """
    level_id = str(uuid.uuid4())
    # Fake level (ML stub)
    level_data = {
        "grid": [[0 for _ in range(request.size)] for _ in range(request.size)]
    }

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO levels (id, difficulty, size, level_json) VALUES (?, ?, ?, ?)",
                (level_id, request.difficulty, request.size, json.dumps(level_data)))
    conn.commit()
    conn.close()

    return LevelResponse(
        id=level_id,
        level=level_data,
        difficulty=request.difficulty,
        size=request.size
    )


@app.get("/levels/{level_id}", response_model=LevelResponse)
def get_level(level_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, difficulty, size, level_json FROM levels WHERE id = ?", (level_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Level not found")

    return LevelResponse(
        id=row[0],
        difficulty=row[1],
        size=row[2],
        level=json.loads(row[3])
    )
