import json
import os
import re
from typing import Any, Dict, List, Optional
import openai
from openai import OpenAI
from datetime import datetime

try:
    # When imported as part of the package: backend.newModel.model
    from .utility import (
        callSecurityFn,
        getContext,
        setContext,
        resetContext,
        verifyUserFn,
        consumeHistoryResetFlag,
        findDeliveriesFn,
        findMeetingFn,
        addDeliveryFn,
        addMeetingFn,
        signalDoorFn,
        alertSecurityFn,
    )
except ImportError:
    # When running this file directly: python backend/newModel/model.py
    from utility import (
        callSecurityFn,
        getContext,
        setContext,
        resetContext,
        verifyUserFn,
        consumeHistoryResetFlag,
        findDeliveriesFn,
        findMeetingFn,
        addDeliveryFn,
        addMeetingFn,
        signalDoorFn,
        alertSecurityFn,
    )



# Sistem promptu (temel): aracı konteks ile yöneten, araç kullanımı zorunlu, Türkçe konuşan konsiyerj
agentSystemPromptBase = """
Rolün: Türkçe konuşan bir sekreter/güvenlik konsiyerjisin.

Görevlerin: - Çalışan girişleri, teslimatlar ve toplantı misafirlerini doğrula.
- SADECE ARAÇLARDAN, KULLANICI BİLGİSİNDEN VE CONTEXT'TEN YARARLAN. KESİNLİKLE UYDURMA. 
- Şüpheli durumlarda güvenliği uyar; kapı kontrolünü araçlarla yönet. 
- Kısa, nazik, Türkçe; araç mesajı varsa onu tercih et; araç nesne döndürürse doğal cümleye çevir 
- İç araç adlarını/uygulama detaylarını kullanıcıya söyleme 

Bağlam (context) yönetimi: 
- Güncel durum userContext ile tutulur. 
- userContext'i değiştirmek için updateContext aracını kullan
- Erişmek için getContext aracını, uygun alanları doldurmak için updateContext aracını çağır.
- Alanlar: intent, employeeName, password, company, recipient, host, guest, time.
- HER ZAMAN ÖNCE INTENT BELİRLE.


NİYETLER ve ZORUNLU ALANLAR
- employee: employeeName + password → verifyUser → doğrulanırsa signalDoor("open"), değilse "deny".
- delivery: recipient veya company → findDeliveries → eşleşirse yönlendir/“open”, aksi halde resepsiyona yönlendir.
- meeting: host/guest/time alanlarından en az ikisi → findMeeting → uygun ise “open”.
- suspicious: alertSecurity → signalDoor("lock").
- unknown: netleştirici tek bir soru sor.
- Zorla girme, peşinden girme, kimlik vermeme, tehdit/hakaret/ısrar, mesai dışı ısrarlı giriş.
- Aksiyon sırası: 
    (1) Kapıyı kilitle. 
    (2) Tek uyarı cümlesi yaz: “Güvenlik çağrıldı; lütfen resepsiyonda bekleyiniz.” 
    
    
BOŞ ALAN DAVRANIŞI
intent'e göre eksik alanları tek tek sor:
- delivery iken ad/kurye firması söylenirse öncelik: recipient/company.
- employee iken önce employeeName, sonra password sor.
- meeting iken eksik olanı tek soru ile tamamla.

Araç kullanımı kuralları:
- Araç sonucu birincil gerçek kaynağıdır. Araç çağrıldıysa, cevabı yalnızca araç ve kullanıcı verisine dayandır.” 
- KULLANICIYA SORU SORMADAN ÖNCE HER ZAMAN CONTEXT'E BAK, EĞER O BİLGİ VARSA SORMA.
- EĞER ARAÇTAN BİR CÜMLE GELİRSE, ONU KULLANARAK CEVAPLA. 
- Kapı işlemleri için signalDoor kullan.
- Uydurma yapma; sadece araç sonuçlarına ve kullanıcının verdiği bilgilere dayan.

Yemek tarifi, hava durumu, spor, genel kültür, teknoloji, tarih, felsefe, kişisel gelişim, film önerisi gibi konularda yardımcı olma.


"""


def buildSystemPrompt() -> str:

    today_str = datetime.now().strftime("%H:%M")
    return agentSystemPromptBase + f"\n\nGüncel tarih: {today_str}\n"


# Araç tanımları (LLM'e gösterilen fonksiyon şemaları)
tools = [
    {
        "type": "function",
        "function": {
            "name": "getContext",
            "description": "userContext'i döndürür. Kullanıcıdan bilgi istemeden önce çağır. Eğer userContext'te varsa isteme. ",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "updateContext",
            "description": "Konuşmadan çıkarılan alanlarla userContext'i günceller KULLANICI BİLGİ VERDİĞİNDE KESİNLİKLE ÇAĞIR. Sadece verilen alanlar yazılır (None/boş gönderme)."
            + " intent=delivery için recipient/company, intent=employee için employeeName/password, intent=meeting için host/guest/time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": [
                            "unknown",
                            "employee",
                            "delivery",
                            "meeting",
                            "suspicious",
                            "addMeeting",
                            "addDelivery",
                        ],
                    },
                    "employeeName": {"type": "string"},
                    "password": {"type": "string"},
                    "company": {"type": "string"},
                    "recipient": {"type": "string"},
                    "host": {"type": "string"},
                    "guest": {"type": "string"},
                    "time": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resetContext",
            "description": "userContext'i varsayılana sıfırlar; sadece işlem tamamlandıktan sonra veya yeni oturuma geçerken kullan.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verifyUser",
            "description": "YALNIZ intent=employee ve çalışan bulunduysa çağır. Öncesinde updateContext ile employeeName ve password alanlarını yaz.  Araçtan dönen mesajı ilet",
            "parameters": {
                "type": "object",
                "properties": {
                    "employeeName": {"type": "string"},
                    "password": {"type": "string"},
                },
                "required": ["employeeName", "password"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "findDeliveries",
            "description": "YALNIZCA intent=delivery VE company, recipient bilgileri verilmişse çağır. Araçtan dönen mesajı ilet",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "company": {"type": "string"},
                },
                "anyOf": [{"required": ["recipient"]}, {"required": ["company"]}],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "findMeeting",
            "description": "YALNIZ intent=meeting VE time,host,guest dolu iken çağır. Toplantı akışı: host/guest/time alanlarından en az ikisiyle arama yap; mümkünse guest ve time veya host ve time ver.",
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string"},
                    "guest": {"type": "string"},
                    "time": {"type": "string"},
                },
                "anyOf": [
                    {"required": ["guest", "time"]},
                    {"required": ["host", "time"]},
                    {"required": ["host", "guest"]},
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "addDelivery",
            "description": "YALNIZ intent=addDelivery is çağır. Veritabanına yeni teslimat ekler.  Eğer kullanıcı bir teslimat veya kargo geleceğini söylüyorsa çağır. Çalışan adı, şifre, şirket ZORUNLU; alıcı bilgisi çalışan adı kabul edilir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "employeeName": {"type": "string"},
                    "password": {"type": "string"},
                    "company": {"type": "string"},
                },
                "required": ["employeeName", "password", "company"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "addMeeting",
            "description": "YALNIZ intent=addMeeting is çağır. Veritabanına yeni toplantı ekler. Eğer kullanıcı birinin onu ziyarete ya da toplantıya geleceğini söylüyorsa çağır. Çalışan adı ve şifre ZORUNLU; guest ve time ZORUNLU. Host bilgisi çalışan adı kabul edilir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "employeeName": {"type": "string"},
                    "password": {"type": "string"},
                    "guest": {"type": "string"},
                    "time": {"type": "string"},
                },
                "required": ["employeeName", "password", "guest", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "signalDoor",
            "description": "Kapı kontrolü (open/deny/lock/unlock). Yalnız doğrulama kararı sonrası çağır: employee doğrulandıysa open, teslimat/toplantı eşleştiyse giriş yönlendir; şüpheli/uygunsuzsa deny veya lock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["open", "deny", "lock", "unlock"],
                    },
                    "person": {"type": "string", "description": "İlgili kişi"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alertSecurity",
            "description": "YALNIZ intent=suspicious veya tehdit içerirken çağır. Güvenliği uyar; nedeni net ve kısa yaz, varsa detayları ekle. Bu araç çağrıldıysa kullanıcıya beklemesini söyle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "details": {"type": "object"},
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "callSecurity",
            "description": "Kullanıcıya yardımcı olamadığında veya LLM bağlantı hatası varsa çağır. Güvenliği acil olarak uyar; kapı kilitli kalmalı. Bu araç çağrıldıysa kullanıcıya beklemesini söyle.",
        },
        "parameters": {"type": "object", "properties": {}}
    },
]


client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
)
modelId = os.getenv("MODEL_ID", "gpt-oss:20b")
#modelId = os.getenv("MODEL_ID", "gpt-oss:20b")


def _filtered(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in (d or {}).items() if v is not None}


def _normalize_time(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return text

    normalized = text.lower().replace("’", "'")

    def _format(hour_str: str, minute_str: Optional[str]) -> Optional[str]:
        try:
            hour = int(hour_str)
            minute = int(minute_str) if minute_str is not None else 0
        except ValueError:
            return None

        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None

        if hour < 7:
            hour+=12

        return f"{hour:02d}:{minute:02d}"

    direct_match = re.fullmatch(r"\s*(\d{1,2})\s*[:.]\s*(\d{2})\s*", text)
    if direct_match:
        formatted = _format(direct_match.group(1), direct_match.group(2))
        if formatted:
            return formatted

    patterns = [
        r"saat\s*(\d{1,2})\s*[:.]\s*(\d{2})",
        r"(\d{1,2})\s*[:.]\s*(\d{2})",
        r"saat\s*(\d{1,2})\b",
        r"(\d{1,2})\s*(?:'de|'te|'da|'ta|de|te|da|ta)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue

        hour_str = match.group(1)
        minute_str = match.group(2) if match.lastindex and match.lastindex >= 2 else None
        formatted = _format(hour_str, minute_str)
        if formatted:
            return formatted

    return text


def _as_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except Exception:
        return str(result)


# LLM araç çağrılarını backend fonksiyonlarına bağlayan adaptörler
def _get_ctx(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return getContext()
    except Exception:
        return {}


def _update_ctx(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = dict(args or {})
        if "time" in payload:
            payload["time"] = _normalize_time(payload.get("time"))
        setContext(_filtered(payload))
    except Exception:
        pass
    try:
        return getContext()
    except Exception:
        return {}


def _tool_verify_user(args: Dict[str, Any]) -> Any:
    _update_ctx(
        {"employeeName": args.get("employeeName"), "password": args.get("password")}
    )
    return verifyUserFn()


def _tool_find_deliveries(args: Dict[str, Any]) -> Any:
    _update_ctx(
        {
            "recipient": args.get("recipient"),
            "company": args.get("company"),
        }
    )
    return findDeliveriesFn()


def _tool_find_meeting(args: Dict[str, Any]) -> Any:
    _update_ctx(
        {
            "host": args.get("host"),
            "guest": args.get("guest"),
            "time": args.get("time"),
        }
    )
    return findMeetingFn()


def _tool_add_delivery(args: Dict[str, Any]) -> Any:
    _update_ctx(
        {
            "employeeName": args.get("employeeName"),
            "password": args.get("password"),
            "company": args.get("company"),
        }
    )
    return addDeliveryFn()


def _tool_add_meeting(args: Dict[str, Any]) -> Any:
    _update_ctx(
        {
            "employeeName": args.get("employeeName"),
            "password": args.get("password"),
            "guest": args.get("guest"),
            "time": args.get("time"),
        }
    )
    return addMeetingFn()


def _tool_signal_door(args: Dict[str, Any]) -> Any:
    return signalDoorFn(args.get("action"), args.get("person"))


def _tool_alert_security(args: Dict[str, Any]) -> Any:
    return alertSecurityFn(args.get("reason"), args.get("details"))

def _tool_call_security(args: Dict[str, Any]) -> Any:
    return callSecurityFn()

toolMap = {
    "getContext": _get_ctx,
    "updateContext": _update_ctx,
    "resetContext": lambda args: (resetContext() or getContext()),
    "verifyUser": _tool_verify_user,
    "findDeliveries": _tool_find_deliveries,
    "findMeeting": _tool_find_meeting,
    "addDelivery": _tool_add_delivery,
    "addMeeting": _tool_add_meeting,
    "signalDoor": _tool_signal_door,
    "alertSecurity": _tool_alert_security,
    "callSecurity": _tool_call_security,
}


def runAgent(
    userInput: Dict[str, Any],
    history: Optional[List[Dict]] = None,
) -> Dict[str, Any]:

    user_text = ""
    if isinstance(userInput, dict):
        user_text = str(userInput.get("text") or "").strip()
    if not user_text:
        return {
            "reply": "Nasıl yardımcı olabilirim? Çalışan girişi, teslimat veya toplantı için mi geldiniz?",
            "history_was_reset": False,
            "history_reset_before_call": False,
            "history_reset_during_call": False,
        }

    def _normalize_history(h: Optional[List[Dict]], max_pairs: int = 8) -> List[Dict[str, str]]:
        if not h:
            return []
        buf: List[Dict[str, str]] = []
        for turn in h:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                buf.append({"role": role, "content": content.strip()})
        return buf[-max_pairs * 2 :]

    reset_history = False
    try:
        reset_history = consumeHistoryResetFlag()
    except Exception:
        reset_history = False

    history_reset_occurred = reset_history
    history_reset_during_call = False
    history_reset_before_call = reset_history

    def _finalize(text: str) -> Dict[str, Any]:
        nonlocal history_reset_occurred, history_reset_during_call
        try:
            if consumeHistoryResetFlag():
                history_reset_occurred = True
                history_reset_during_call = True
        except Exception:
            pass
        return {
            "reply": (text or "").strip(),
            "history_was_reset": history_reset_occurred,
            "history_reset_before_call": history_reset_before_call,
            "history_reset_during_call": history_reset_during_call,
        }

    recent = [] if reset_history else _normalize_history(history, max_pairs=8)

    try:
        state_snapshot = toolMap["getContext"]({}) or {}
    except Exception:
        state_snapshot = {}

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": buildSystemPrompt()}
    ]
    messages.append(
        {
            "role": "system",
            "content": "Durum özeti (sadece bağlam için, nihai gerçeklik araç sonuçlarıdır): "
                       + json.dumps(state_snapshot, ensure_ascii=False),
        }
    )
    messages.extend(recent)
    messages.append({"role": "user", "content": user_text})

    last_text = ""
    MAX_STEPS = 10
    step = 0

    try:
        while True:
            step += 1
            if step > MAX_STEPS:
                return _finalize(
                    last_text or "Üzgünüm, bir karar veremedim. Lütfen tekrar deneyiniz."
                )

            resp = client.chat.completions.create(
                model=modelId,             
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
            )

            choice = resp.choices[0]
            msg = choice.message

            assistant_payload: Dict[str, Any] = {"role": "assistant"}
            if getattr(msg, "content", None):
                assistant_payload["content"] = msg.content
                last_text = msg.content or last_text
            if getattr(msg, "tool_calls", None):
                assistant_payload["tool_calls"] = msg.tool_calls
            messages.append(assistant_payload)

            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    try:
                        handler = toolMap.get(name)
                        if handler is None:
                            result = {"error": f"tool_not_allowed:{name}"}
                        else:
                            result = handler(args)
                        tool_content = _as_content(result)
                    except Exception as e:
                        tool_content = _as_content({"error": "tool_error", "message": str(e)})

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": tool_content,
                        }
                    )
                continue

            if getattr(msg, "content", None):
                return _finalize(msg.content or "")

            if choice.finish_reason in ("stop", "length", "content_filter"):
                return _finalize(last_text or "")

    except (openai.APIConnectionError, Exception) as e:
        print("LLM connection error:", repr(e))
        try:
            callSecurityFn()
        except Exception:
            pass
        return _finalize(
            "Üzgünüm, şu anda yardımcı olamıyorum. Lütfen kapıda bekleyiniz; yetkiliye haber veriyorum."
        )



if __name__ == "__main__":
    tests = [
        {"text": "Merhaba, Aras Kargo'dan geldim. Umut Deniz'e teslimat var."},
        {"text": "Merhaba, Trendyol'dan geldim. Umut Deniz'e teslimat var."},
        {"text": "Ben Mustafa Alkan, personelim. Şifrem 4567."},
        {"text": "Saat 16:00'da Umut Deniz ile toplantım var. Adım Arda."},
        {"text": "Kartım yok, kapıyı açmıyorum derseniz zorla girerim."},
    ]
    for t in tests:
        print(json.dumps(runAgent(t), ensure_ascii=False))
        print("----------------------------------------------------")
