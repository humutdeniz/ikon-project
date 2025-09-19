import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
import smtplib
from email.message import EmailMessage
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
# When True, the next model call should ignore provided chat history
_history_reset_flag = False


def getContext() -> Dict[str, str]:
    print("------CONTEXT GET ------")
    return userContext


def setContext(newContext: Dict[str, str]) -> None:
    global messageCount
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
    global messageCount, _history_reset_flag
    print("-----------------------------CONTEXT RESET--------------------------------")
    userContext.update(defaultContext)
    messageCount = 0
    _history_reset_flag = True


def setHistoryResetFlag() -> None:
    global _history_reset_flag
    _history_reset_flag = True

def consumeHistoryResetFlag() -> bool:
    global _history_reset_flag
    flagged = _history_reset_flag
    _history_reset_flag = False
    return flagged


def getTodayDate() -> str:
    return datetime.datetime.now().isoformat(timespec="minutes")


def _db_path() -> Path:
    # 1) Allow explicit override via env var
    env_path = os.getenv("DB_PATH")
    if env_path:
        return Path(env_path)

    # 2) Default: use the repo's backend DB file next to this package
    # This file lives in backend/newModel/, so go up to backend/
    backend_dir = Path(__file__).resolve().parent.parent
    return backend_dir / "ai-concierge.db"


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
            )
            if len(rows) > 0:
                resetContext()
                return "Kapıyı açıyorum. Hoş geldiniz " + rows[0]["name"] + "."
            else:
                return "Kullanıcı doğrulanamadı. Lütfen adınızı tam, şifrenizi doğru giriniz."
    except Exception as e:
        print("verifyUserFn error:", e)
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
            params.append(f"%{userContext['company'].upper()}%")

        where.append("status = ?")
        params.append("pending")

        if where:
            sql += " WHERE " + " AND ".join(where)

        with _connect() as conn:
            cur = conn.execute(sql, params)
            rows = _rows_to_dicts(cur.fetchall())
            print("findDeliveriesFn")
            pp(userContext)
            print("length:", len(rows))
            if len(rows) > 0:
                editDeliveriesFn(rows[0]["id"], None, None, "delivered")
                try:
                    subject = "Teslimat bildirimi"
                    msg = (
                        f"Sayın {rows[0]['recipient']},\n\n"
                        f"{rows[0]['company']} firmasından bir teslimatınız resepsiyona bırakıldı.\n"
                        f"Durum: delivered\n"
                        f"Zaman: {getTodayDate()}\n\n"
                        "İyi çalışmalar."
                    )
                    alertUserFn(rows[0]["recipient"], subject, msg)
                except Exception as _e:
                    print("alertUserFn (delivery) error:", _e)
                resetContext()
                return "Hoşgeldiniz, lütfen teslimatını resepsiyondaki kargo bölümüne bırakınız. Hemen alıcıya haber veriyorum."
            else:
                # Context'i koru ki LLM eksikleri tamamlamak için sorular sorabilsin
                return "Teslimat bulunamadı. Lütfen alıcı ismini ve şirket ismini kontrol ediniz."
    except Exception as e:
        print("findDeliveriesFn error:", e)
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
        print("editDeliveriesFn error:", e)
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
            where.append("guest LIKE ?")
            params.append(f"%{userContext['guest'].upper()}%")
        if userContext["time"] != UNKNOWN:
            where.append("date = ?")
            params.append(userContext['time'])

        if where:
            sql += " WHERE " + " AND ".join(where)

        with _connect() as conn:
            cur = conn.execute(sql, params)
            rows = _rows_to_dicts(cur.fetchall())
            print("findMeetingFn")
            pp(userContext)
            if len(rows) > 0:
                try:
                    subject = "Misafir geldi bildirimi"
                    when = rows[0].get("date") or getTodayDate()
                    msg = (
                        f"Sayın {rows[0]['host']},\n\n"
                        f"Misafiriniz {rows[0]['guest']} geldi ve resepsiyonda bekliyor.\n"
                        f"Zaman: {when}\n\n"
                        "İyi çalışmalar."
                    )
                    alertUserFn(rows[0]["host"], subject, msg)
                except Exception as _e:
                    print("alertUserFn (meeting) error:", _e)
                resetContext()
                return (
                    "Hoşgeldiniz, "
                    + rows[0]["guest"].upper()
                    + ". Lütfen resepsiyondaki toplantı odasına geçiniz. Hemen ev sahibine haber veriyorum."
                )
            else:
                return "Toplantı bulunamadı. Lütfen ev sahibi ismi, misafir ismi ve zamanı doğru giriniz."
    except Exception as e:
        print("findMeetingFn error:", e)
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


def _send_email_smtp(to_email: str, subject: str, body: str) -> Dict[str, Any]:
    """Best-effort SMTP sender using env configuration.

    Required env vars when using SMTP:
      - SMTP_HOST
    Optional env vars:
      - SMTP_PORT (default: 587), SMTP_USER, SMTP_PASS, SMTP_FROM (default: no-reply@local)
      - SMTP_USE_SSL ("1"/"true" to force SSL; TLS otherwise)
    """
    host = os.getenv("SMTP_HOST")
    if not host:
        return {"ok": False, "reason": "SMTP not configured"}

    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    from_addr = os.getenv("SMTP_FROM", "no-reply@local")
    use_ssl = os.getenv("SMTP_USE_SSL", "0").lower() in ("1", "true", "yes")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_email
    msg.set_content(body)

    try:
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=10) as smtp:
                if user and password:
                    smtp.login(user, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=10) as smtp:
                smtp.ehlo()
                try:
                    smtp.starttls()
                    smtp.ehlo()
                except Exception:
                    # If server doesn't support TLS, continue without it.
                    pass
                if user and password:
                    smtp.login(user, password)
                smtp.send_message(msg)
        return {"ok": True, "method": "smtp"}
    except Exception as e:
        print("SMTP send failed:", e)
        return {"ok": False, "reason": str(e)}


def alertUserFn(person_name: str, subject: str, message: str) -> Dict[str, Any]:
    """Notify a user by email about an event (delivery/meeting).

    - Looks up the user's email in the users table by name.
    - Tries to send via SMTP if configured; otherwise prints a simulated email.
    """
    try:
        with _connect() as conn:
            # Attempt to fetch email if the column exists.
            # We first detect available columns to avoid crashes if email column is missing.
            cursor = conn.execute("PRAGMA table_info(users)")
            cols = {row[1] for row in cursor.fetchall()}  # name is at index 1
            email_addr: Optional[str] = None

            if "email" in cols:
                cur = conn.execute(
                    "SELECT email FROM users WHERE name = ?",
                    (person_name.upper(),),
                )
                rec = cur.fetchone()
                if rec:
                    email_addr = rec[0]

        full_message = message
        if not email_addr:
            # No email column or no address — simulate notification for visibility
            print("[Simulated email] To:", person_name)
            print("Subject:", subject)
            print(full_message)
            return {"ok": True, "method": "stdout", "note": "email missing or not configured"}

        # Try SMTP send if we have an address
        res = _send_email_smtp(email_addr, subject, full_message)
        if not res.get("ok"):
            # Fallback to simulated email logging
            print("[Simulated email after SMTP failure] To:", email_addr)
            print("Subject:", subject)
            print(full_message)
            res = {"ok": True, "method": "stdout"}
        return res
    except Exception as e:
        print("alertUserFn error:", e)
        return {"ok": False, "error": str(e)}


def addDeliveryFn():
    try:
        if userContext["employeeName"] == UNKNOWN or userContext["password"] == UNKNOWN:
            return "Lütfen adınızı ve şifrenizi giriniz."

        with _connect() as conn:
            cur = conn.execute(
                "SELECT id FROM users WHERE name = ? AND password = ?",
                (userContext["employeeName"].upper(), userContext["password"]),
            )
            user = cur.fetchone()
            if not user:
                return "Kullanıcı doğrulanamadı. Lütfen adınızı tam, şifrenizi doğru giriniz."

            if userContext["company"] == UNKNOWN:
                return "Lütfen şirket ismini giriniz."

            conn.execute(
                "INSERT INTO deliveries (recipient, company, status) VALUES (?, ?, ?)",
                (
                    userContext["employeeName"].upper(),
                    userContext["company"].upper(),
                    "pending",
                ),
            )
            conn.commit()

            resetContext()
            return "Teslimat başarıyla kaydedildi."
    except Exception as e:
        print("addDeliveryFn error:", e)
        callSecurityFn()
        return "Teknik bir sorun oluştu. Size yardımcı olacak görevliyi çağırıyorum."


def addMeetingFn():
    try:
        if userContext["employeeName"] == UNKNOWN or userContext["password"] == UNKNOWN:
            return "Lütfen adınızı ve şifrenizi giriniz."

        with _connect() as conn:
            cur = conn.execute(
                "SELECT id FROM users WHERE name = ? AND password = ?",
                (userContext["employeeName"].upper(), userContext["password"]),
            )
            user = cur.fetchone()
            if not user:
                return "Kullanıcı doğrulanamadı. Lütfen adınızı tam, şifrenizi doğru giriniz."

            if userContext["guest"] == UNKNOWN or userContext["time"] == UNKNOWN:
                return "Lütfen misafir ve zamanı giriniz."

            conn.execute(
                "INSERT INTO meetings (host, guest, date) VALUES (?, ?, ?)",
                (
                    userContext["employeeName"].upper(),
                    userContext["guest"].upper(),
                    userContext["time"],
                ),
            )
            conn.commit()

            resetContext()
            return "Toplantı başarıyla oluşturuldu."
    except Exception:
        callSecurityFn()
        return "Teknik bir sorun oluştu. Size yardımcı olacak görevliyi çağırıyorum."
