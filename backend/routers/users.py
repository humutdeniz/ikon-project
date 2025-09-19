from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..db import get_connection


router = APIRouter()


class UserCreate(BaseModel):
    name: str
    status: str
    password: str


class UserOut(BaseModel):
    id: int
    name: str
    status: Optional[str]


@router.get("/users")
def list_users(name: Optional[str] = Query(None), status: Optional[str] = Query(None)):
    sql = "SELECT id, name, status, password FROM users"
    params: List[object] = []
    conditions: List[str] = []
    name = name.upper() if name else name

    if name:
        conditions.append("name = ?")
        params.append(name)
    if status:
        conditions.append("status = ?")
        params.append(status)

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    conn = get_connection()
    try:
        cur = conn.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        return {"message": "success", "data": rows}
    finally:
        conn.close()


@router.post("/users", status_code=201)
def create_user(payload: UserCreate):
    if not payload.name or not payload.status or not payload.password:
        raise HTTPException(
            status_code=400, detail="Name, status, and password are required."
        )
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO users (name, status, password) VALUES (?, ?, ?)",
            (payload.name.upper(), payload.status, payload.password),
        )
        conn.commit()
        return {
            "message": "success",
            "data": {
                "id": cur.lastrowid,
                "name": payload.name,
                "status": payload.status,
            },
        }
    finally:
        conn.close()
