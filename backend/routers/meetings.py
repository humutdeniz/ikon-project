from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..db import get_connection


router = APIRouter()


class MeetingCreate(BaseModel):
    host: str
    guest: str
    date: str


@router.get("/meetings")
def list_meetings(
    host: Optional[str] = Query(None),
    guest: Optional[str] = Query(None),
    date: Optional[str] = Query(None),
):
    sql = "SELECT * FROM meetings"
    params: List[object] = []
    conditions: List[str] = []

    if host:
        conditions.append("host LIKE ?")
        params.append(f"%{host}%")
    if guest:
        conditions.append("guest = ?")
        params.append(guest)
    if date:
        conditions.append("date = ?")
        params.append(date)

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    conn = get_connection()
    try:
        cur = conn.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        return {"message": "success", "data": rows}
    finally:
        conn.close()


@router.post("/meetings", status_code=201)
def create_meeting(payload: MeetingCreate):
    if not payload.host or not payload.guest or not payload.date:
        raise HTTPException(status_code=400, detail="Host, guest and date are required.")

    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO meetings (host, guest, date) VALUES (?, ?, ?)",
            (payload.host, payload.guest, payload.date),
        )
        conn.commit()
        return {
            "message": "success",
            "data": {"id": cur.lastrowid, "host": payload.host, "guest": payload.guest, "date": payload.date},
        }
    finally:
        conn.close()

