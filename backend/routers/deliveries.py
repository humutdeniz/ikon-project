from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..db import get_connection


router = APIRouter()


class DeliveryCreate(BaseModel):
    recipient: str
    company: str
    status: Optional[str] = "pending"


class DeliveryUpdate(BaseModel):
    id: int
    recipient: Optional[str] = None
    company: Optional[str] = None
    status: Optional[str] = None


@router.get("/deliveries")
def list_deliveries(
    recipient: Optional[str] = Query(None),
    company: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
):
    sql = "SELECT id, recipient, company, status FROM deliveries"
    params: List[object] = []
    conditions: List[str] = []

    if recipient:
        conditions.append("recipient LIKE ?")
        params.append(f"%{recipient}%")
    if company:
        # Mirror existing Node: company LIKE ? (without wildcards)
        conditions.append("company LIKE ?")
        params.append(company)
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


@router.post("/deliveries", status_code=201)
def create_delivery(payload: DeliveryCreate):
    if not payload.recipient or not payload.company:
        raise HTTPException(status_code=400, detail="Recipient, and company are required.")

    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO deliveries (recipient, company, status) VALUES (?, ?, ?)",
            (payload.recipient, payload.company, payload.status),
        )
        conn.commit()
        return {
            "message": "success",
            "data": {
                "id": cur.lastrowid,
                "recipient": payload.recipient,
                "company": payload.company,
                "status": payload.status,
            },
        }
    finally:
        conn.close()


@router.post("/editDeliveries")
def edit_delivery(payload: DeliveryUpdate):
    if not payload.id:
        raise HTTPException(status_code=400, detail="id is required.")
    if payload.recipient is None and payload.company is None and payload.status is None:
        raise HTTPException(status_code=400, detail="Nothing to update.")

    set_parts: List[str] = []
    params: List[object] = []
    if payload.recipient is not None:
        set_parts.append("recipient = ?")
        params.append(payload.recipient)
    if payload.company is not None:
        set_parts.append("company = ?")
        params.append(payload.company)
    if payload.status is not None:
        set_parts.append("status = ?")
        params.append(payload.status)

    sql = f"UPDATE deliveries SET {', '.join(set_parts)} WHERE id = ?"
    params.append(payload.id)

    conn = get_connection()
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Delivery not found or no changes.")

        cur2 = conn.execute("SELECT * FROM deliveries WHERE id = ?", (payload.id,))
        row = cur2.fetchone()
        return {"message": "success", "data": dict(row)}
    finally:
        conn.close()

