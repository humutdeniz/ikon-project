import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime

from pprint import pp

UNKNOWN = "unknown"

defaultContext = {
    "intent": UNKNOWN,
    "employeeName": UNKNOWN,
    "password": UNKNOWN,
    "company": UNKNOWN,
    "recipient": UNKNOWN,
    "host": UNKNOWN,
    "guest": UNKNOWN,
    "time": UNKNOWN,
}

userContext = {
    "intent": UNKNOWN,
    "employeeName": UNKNOWN,
    "password": UNKNOWN,
    "company": UNKNOWN,
    "recipient": UNKNOWN,
    "host": UNKNOWN,
    "guest": UNKNOWN,
    "time": UNKNOWN,
}

messageCount = 0


def getContext() -> Dict[str, str]:
    return userContext


def setContext(newContext: Dict[str, str]) -> None:
    userContext.update(newContext)
    print("------CONTEXT CHANGE ------")
    pp(userContext)
    print("------CONTEXT CHANGE ------")

    messageCount += 1
    if messageCount > 4 and userContext["intent"] == UNKNOWN:
        print(
            "Ne demek istediğinizi anlayamadım. Size yardımcı olacak görevliyi çağırıyorum."
        )
        resetContext()
        messageCount = 0
        callSecurityFn()


def resetContext() -> None:
    print("-----------------------------CONTEXT RESET--------------------------------")
    userContext.update(defaultContext)


def getTodayDate() -> str:
    return datetime.datetime.now().isoformat(timespec="minutes")


def _db_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "ikon-project" / "backend" / "ai-concierge.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 3000")
    except Exception:
        pass
    return conn


def _rows_to_dicts(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


def _success(data: Any) -> Dict[str, Any]:
    return {"message": "success", "data": data}


def callSecurityFn():
    print("callSecurityFn")
    resetContext()
    return {"ok": True}


def verifyUserFn():
    try:
        with _connect() as conn:
            cur = conn.execute(
                "SELECT id, name, status FROM users WHERE name = ? AND password = ?",
                (userContext["employeeName"].upper(), userContext["password"]),
            )
            rows = _rows_to_dicts(cur.fetchall())
            print(
                "verifyUserFn",
                userContext["employeeName"].upper(),
                userContext["password"],
            ),
            if len(rows) > 0:
                resetContext()
                return "Kapıyı açıyorum. Hoş geldiniz " + rows[0]["name"] + "."
            else:
                return "Kullanıcı doğrulanamadı. Lütfen adınızı tam, şifrenizi doğru giriniz."
    except Exception as e:
        callSecurityFn()
        return "Teknik bir sorun oluştu. Size yardımcı olacak görevliyi çağırıyorum."


def findDeliveriesFn():
    try:
        sql = "SELECT id, recipient, company, status FROM deliveries"
        params: List[Any] = []
        where: List[str] = []

        if userContext["recipient"] != UNKNOWN:
            where.append("recipient LIKE ?")
            params.append(f"%{userContext['recipient'].upper()}%")
        if userContext["company"] != UNKNOWN:
            where.append("company LIKE ?")
            params.append(userContext["company"])

        where.append("status = ?")
        params.append("pending")

        if where:
            sql += " WHERE " + " AND ".join(where)

        with _connect() as conn:
            cur = conn.execute(sql, params)
            rows = _rows_to_dicts(cur.fetchall())
            print("findDeliveriesFn")
            pp(userContext)
            print("lenght:", len(rows))
            if len(rows) > 0:
                editDeliveriesFn(rows[0]["id"], None, None, "delivered")
                # alertUser()
                resetContext()
                return "Hoşgeldiniz, lütfen teslimatını resepsiyondaki kargo bölümüne bırakınız. Hemen alıcıya haber veriyorum."
            else:
                resetContext()
                return "Teslimat bulunamadı. Lütfen alıcı ismi ve şirket ismini doğru giriniz."
    except Exception as e:
        callSecurityFn()
        return "Teknik bir sorun oluştu. Size yardımcı olacak görevliyi çağırıyorum."


def editDeliveriesFn(
    id: int, company: Optional[str], recipient: Optional[str], status: Optional[str]
):
    try:
        sets: List[str] = []
        params: List[Any] = []
        if recipient is not None:
            sets.append("recipient = ?")
            params.append(recipient)
        if company is not None:
            sets.append("company = ?")
            params.append(company)
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if not sets:
            return {"ok": False, "status_code": 400}
        params.append(id)
        with _connect() as conn:
            cur = conn.execute(
                f"UPDATE deliveries SET {', '.join(sets)} WHERE id = ?", params
            )
            conn.commit()
            return {
                "ok": cur.rowcount > 0,
                "status_code": 200 if cur.rowcount > 0 else 404,
            }
    except Exception as e:
        callSecurityFn()
        return "Teknik bir sorun oluştu. Size yardımcı olacak görevliyi çağırıyorum."


def findMeetingFn():
    try:
        sql = "SELECT id, host, guest, date FROM meetings"
        params: List[Any] = []
        where: List[str] = []

        if userContext["host"] != UNKNOWN:
            where.append("host LIKE ?")
            params.append(f"%{userContext['host'].upper()}%")

        if userContext["guest"] != UNKNOWN:
            where.append("guest = ?")
            params.append(userContext["guest"].upper())
        if userContext["time"] != UNKNOWN:
            where.append("date = ?")
            today = datetime.date.today().strftime("%Y-%m-%d")
            formatted = f"{today}T{userContext['time']}"
            params.append(formatted)

        if where:
            sql += " WHERE " + " AND ".join(where)

        with _connect() as conn:
            cur = conn.execute(sql, params)
            rows = _rows_to_dicts(cur.fetchall())
            print("findMeetingFn")
            pp(userContext)
            if len(rows) > 0:
                # alertUser()
                resetContext()
                return (
                    "Hoşgeldiniz, "
                    + rows[0]["guest"].upper()
                    + ". Lütfen resepsiyondaki toplantı odasına geçiniz. Hemen ev sahibine haber veriyorum."
                )
            else:
                return "Toplantı bulunamadı. Lütfen ev sahibi ismi, misafir ismi ve zamanı doğru giriniz."
    except Exception as e:
        callSecurityFn()
        return "Teknik bir sorun oluştu. Size yardımcı olacak görevliyi çağırıyorum."


def signalDoorFn(action: str, person: str | None):
    # No backend call; simulate side-effect locally
    print("signalDoor", action, person)
    return {"ok": True, "action": action, "person": person}


def alertSecurityFn(reason: str, details: dict | None):
    # No backend call; simulate side-effect locally
    print("alertSecurityFn", reason, details)
    return {"ok": True, "reason": reason}
