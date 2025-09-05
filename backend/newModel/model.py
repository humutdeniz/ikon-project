import json
import os
from typing import Any, Dict, List, Optional
from openai import OpenAI

from utility import (
    getContext,
    setContext,
    resetContext,
    verifyUserFn,
    findDeliveriesFn,
    findMeetingFn,
    signalDoorFn,
    alertSecurityFn,
)


try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


# Sistem promptu: aracı konteks ile yöneten, araç kullanımı zorunlu, Türkçe konuşan konsiyerj
agentSystemPrompt = """
Rolün: Türkçe konuşan bir sekreter/güvenlik konsiyerjisin. Görevlerin:
- Çalışan girişleri, teslimatlar ve toplantı misafirlerini doğrula.
- SADECE ARAÇLARDAN, KULLANICI BİLGİSİNDEN VE CONTEXT'TEN YARARLAN. KESİNLİKLE UYDURMA.
- Şüpheli durumlarda güvenliği uyar; kapı kontrolünü araçlarla yönet.
- Kısa, nazik, Türkçe; araç mesajı varsa onu tercih et; araç nesne döndürürse doğal cümleye çevir
- İç araç adlarını/uygulama detaylarını kullanıcıya söyleme

Bağlam (context) yönetimi:
- Güncel durum userContext ile tutulur. Erişmek için getContext aracını, uygun alanları doldurmak için updateContext aracını çağır.
- Alanlar: intent, employeeName, password, company, recipient, host, guest, time.
- Eksik kritik bilgi varsa önce NETLEŞTİRİCİ soru sor, sonra aracı çağır.
- EĞER INTENT=DELIVERY İSE öncelikli olarak recipient VEYA company doldur.
- EĞER INTENT=EMPLOYEE İSE öncelikli olarak employeeName VE password doldur.
- EĞER INTENT=MEETING İSE öncelikli olarak HOST, GUEST VE TIME doldur.
- INTENT=UNKNOWN İSE önce netleştirici soru sor.

ŞÜPHELİ/TEHDİT
- Zorla girme, peşinden girme, kimlik vermeme, tehdit/hakaret/ısrar, mesai dışı ısrarlı giriş.
- Aksiyon sırası: (1) Kapıyı kilitle. (2) Tek uyarı cümlesi yaz: “Güvenlik çağrıldı; lütfen resepsiyonda bekleyiniz.”

Araç kullanımı kuralları:
- Araç sonucu birincil gerçek kaynağıdır. Araç çağrıldıysa, cevabı yalnızca araç ve kullanıcı verisine dayandır.”
- KULLANICIYA SORU SORMADAN ÖNCE HER ZAMAN CONTEXT'E BAK, EĞER O BİLGİ VARSA SORMA.
- Her turda en az bir araç çağır; sonuçlara göre yanıt ver.
- EĞER ARAÇTAN BİR CÜMLE GELİRSE, ONU KULLANARAK CEVAPLA.
- Kapı işlemleri için signalDoor, güvenlik için alertSecurity kullan.
- Uydurma yapma; sadece araç sonuçlarına ve kullanıcının verdiği bilgilere dayan.

"""


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
            "description": "Konuşmadan çıkarılan alanlarla userContext'i kısmi günceller;"
            + " HER AKIŞTA ÖNCE çağır. Sadece verilen alanlar yazılır (None/boş gönderme)."
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
]


client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
)
modelId = os.getenv("MODEL_ID", "gpt-oss:latest")


def _filtered(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in (d or {}).items() if v is not None}


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
        setContext(_filtered(args))
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


def _tool_signal_door(args: Dict[str, Any]) -> Any:
    return signalDoorFn(args.get("action"), args.get("person"))


def _tool_alert_security(args: Dict[str, Any]) -> Any:
    return alertSecurityFn(args.get("reason"), args.get("details"))


toolMap = {
    "getContext": _get_ctx,
    "updateContext": lambda args: _update_ctx(args),
    "resetContext": lambda args: (resetContext() or getContext()),
    "verifyUser": _tool_verify_user,
    "findDeliveries": _tool_find_deliveries,
    "findMeeting": _tool_find_meeting,
    "signalDoor": _tool_signal_door,
    "alertSecurity": _tool_alert_security,
}


def runAgent(
    userInput: Dict[str, Any],
    decisionOnly: bool = True,
    history: Optional[List[Dict]] = None,
) -> str:
    user_text = None
    try:
        if isinstance(userInput, dict):
            user_text = userInput.get("text")
    except Exception:
        user_text = None

    messages: List[Dict[str, Any]] = [{"role": "system", "content": agentSystemPrompt}]

    if history:
        for turn in history:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                messages.append({"role": role, "content": content})

    messages.append(
        {
            "role": "user",
            "content": user_text or json.dumps(userInput, ensure_ascii=False),
        }
    )

    lastText = ""
    used_tool_this_turn = False
    safeguard_attempts = 0

    # Araç çağrılı çok-adımlı döngü
    while True:
        resp = client.chat.completions.create(
            model=modelId,
            tools=tools,
            tool_choice="auto",
            messages=messages,
            temperature=0.0,
        )

        choice = resp.choices[0]
        msg = choice.message

        assistantPayload: Dict[str, Any] = {"role": "assistant"}
        if getattr(msg, "content", None):
            assistantPayload["content"] = msg.content
            lastText = msg.content or lastText
        if getattr(msg, "tool_calls", None):
            assistantPayload["tool_calls"] = msg.tool_calls
        messages.append(assistantPayload)

        # Araç çağrıları geldiyse çalıştır ve cevapları ekle
        if getattr(msg, "tool_calls", None):
            used_tool_this_turn = True
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                try:
                    handler = toolMap.get(name)
                    if handler is None:
                        result = {"error": "tool_not_allowed"}
                    else:
                        result = handler(args)
                    content = _as_content(result)
                except Exception as e:
                    content = _as_content({"error": "tool_error", "message": str(e)})

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": content,
                    }
                )
            continue

        if getattr(msg, "content", None):
            if not used_tool_this_turn and safeguard_attempts < 2:
                safeguard_attempts += 1
                messages.append(
                    {
                        "role": "system",
                        "content": "Araç kullanımı zorunludur. Önce gerekli alanları updateContext ile ayarla, sonra uygun aracı çağır.",
                    }
                )
                continue
            return msg.content.strip()

        if choice.finish_reason in ("stop", "length", "content_filter"):
            return (lastText or "").strip()


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
