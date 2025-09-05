from __future__ import annotations

import json
import os
from typing import Any, Dict

from openai import OpenAI

from .schema import Context, StepResult, Turn
from ..model.utility import (
    findUserByNameFn,
    verifyUserFn,
    findDeliveriesFn,
    findMeetingFn,
    signalDoorFn,
    alertSecurityFn,
)
from ..model.model import agentSystemPrompt as baseSystemPrompt


client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
)
modelId = os.getenv("MODEL_ID", "qwen2.5:7b")


def _sanitize_context(ctx: Context) -> Dict[str, Any]:
    # Expose persisted slots; omit UI turns for the model to avoid relying on history
    return ctx.model_dump(exclude={"turns"})


def _edit_context_tool(ctx: Context, args: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "role",
        "employeeName",
        "password",
        "company",
        "recipient",
        "host",
        "guest",
        "time",
        "threat",
        "decision",
        "doorAction",
        "note",
    }
    updated: Dict[str, Any] = {}
    for k, v in (args or {}).items():
        if k in allowed:
            setattr(ctx, k, v)
            updated[k] = v
    return {"ok": True, "updated": updated, "context": _sanitize_context(ctx)}


tools = [
    {
        "type": "function",
        "function": {
            "name": "editContext",
            "description": (
                "Kalıcı durum (context) güncelleme aracı. Kullanıcının mesajından çıkardığın rol/isim/saat/şirket gibi bilgileri"
                " bu araçla context'e yaz. Sadece değiştireceğin alanları gönder."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["employee", "courier", "meeting", "unknown"],
                    },
                    "employeeName": {"type": "string"},
                    "password": {"type": "string"},
                    "company": {"type": "string"},
                    "recipient": {"type": "string"},
                    "host": {"type": "string"},
                    "guest": {"type": "string"},
                    "time": {"type": "string"},
                    "threat": {"type": "boolean"},
                    "decision": {
                        "type": "string",
                        "enum": ["allow", "deny", "lock", "ask"],
                    },
                    "doorAction": {"type": "string", "enum": ["unlock", "lock"]},
                    "note": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "findUserByName",
            "description": "Personel kaydı arama (tam ad ile).",
            "parameters": {
                "type": "object",
                "properties": {"employeeName": {"type": "string"}},
                "required": ["employeeName"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verifyUser",
            "description": "Çalışanı ad + şifre ile doğrula (ikisi de zorunlu).",
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
            "description": "Teslimat arama (şirket, alıcı, durum).",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "recipient": {"type": "string"},
                    "status": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "findMeeting",
            "description": "Toplantı arama (ev sahibi, misafir, saat).",
            "parameters": {
                "type": "object",
                "properties": {
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
            "name": "signalDoor",
            "description": "Kapı kontrol (unlock/lock).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["unlock", "lock"]},
                    "person": {"type": "string"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alertSecurity",
            "description": "Güvenliği çağır (tehdit/acil durumda).",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "details": {"type": "object"},
                },
            },
        },
    },
]


def _unwrap_and_count(res) -> Dict[str, Any]:
    try:
        data = res.get("data") if isinstance(res, dict) else None
        if isinstance(data, (list, tuple)):
            count = len(data)
        elif data:
            count = 1
        else:
            count = 0
        return {"found": count > 0, "count": count}
    except Exception:
        return {"found": False, "count": 0}


def step(message: str, context: Dict[str, Any] | Context) -> StepResult:
    ctx = context if isinstance(context, Context) else Context(**(context or {}))

    # Compose a system prompt focused on context-driven flow
    system_prompt = (
        baseSystemPrompt
        + "\n\nKONTEXT ODAKLI AKIŞ (KRİTİK)\n"
        + "- Sohbet geçmişine güvenme; kalıcı alanlar sadece 'context' içindedir.\n"
        + "- Her turda önce 'editContext' aracını kullanarak mesajdan çıkardığın alanları (rol, ad, şifre, şirket, alıcı, ev sahibi, misafir, saat, tehdit) güncelle.\n"
        + "- Eksik bilgi varsa kısa bir soruyla NETLEŞTİR ve yanıtı bekle.\n"
        + "- Onay/veri doğrulaması için uygun aracı (verifyUser, findDeliveries, findMeeting) kullanmadan karar verme.\n"
        + "- Erişim verilecekse kapıyı 'signalDoor' ile aç; tehditte önce 'lock' sonra tek uyarı cümlesi.\n"
        + "- Araç adlarını ya da parametrelerini ASLA yazma; sadece sonuç mesajını 1 cümlede ver.\n"
        + "- Context alanları: role, employeeName, password, company, recipient, host, guest, time, threat, decision, doorAction, note.\n"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "system",
            "content": "Geçerli context (JSON):\n" + json.dumps(_sanitize_context(ctx), ensure_ascii=False),
        },
        {"role": "user", "content": message},
    ]

    # Tool dispatcher uses the live context instance
    toolMap = {
        "editContext": lambda args: _edit_context_tool(ctx, args),
        "findUserByName": lambda args: _unwrap_and_count(
            findUserByNameFn(args.get("employeeName"))
        ),
        "verifyUser": lambda args: _unwrap_and_count(
            verifyUserFn(args.get("employeeName"), args.get("password"))
        ),
        "findDeliveries": lambda args: _unwrap_and_count(
            findDeliveriesFn(
                args.get("company"), args.get("recipient"), args.get("status")
            )
        ),
        "findMeeting": lambda args: _unwrap_and_count(
            findMeetingFn(args.get("host"), args.get("guest"), args.get("time"))
        ),
        "signalDoor": lambda args: (
            signalDoorFn(args.get("action"), args.get("person"))
            or {"ok": True, "action": args.get("action")}
        ),
        "alertSecurity": lambda args: (
            alertSecurityFn(args.get("reason"), args.get("details"))
            or {"ok": True, "reason": args.get("reason")}
        ),
    }

    last_text = ""
    used_tool = False
    safeguard_attempts = 0

    while True:
        resp = client.chat.completions.create(
            model=modelId,
            tools=tools,
            tool_choice="required",  # force at least an editContext or a lookup
            messages=messages,
            temperature=0,
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

        if getattr(msg, "tool_calls", None):
            used_tool = True
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                try:
                    result = toolMap.get(name, lambda a: {"error": "tool_not_allowed"})(
                        args
                    )
                    content = (
                        result
                        if isinstance(result, str)
                        else json.dumps(result, ensure_ascii=False)
                    )
                except Exception as e:
                    content = json.dumps(
                        {"error": "tool_error", "message": str(e)}, ensure_ascii=False
                    )

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
            # Encourage at least one tool call (editContext/lookup) before final answer
            if not used_tool and safeguard_attempts < 2:
                safeguard_attempts += 1
                messages.append(
                    {
                        "role": "system",
                        "content": "Araç kullanımı zorunludur. Önce context'i güncelleyin ve gerekiyorsa doğrulayın.",
                    }
                )
                continue

            # Finalize
            reply = msg.content.strip()
            # Update UI turns; the agent itself does not rely on these
            ctx.turns.append(Turn(role="user", content=message))
            ctx.turns.append(Turn(role="assistant", content=reply))
            return StepResult(reply=reply, context=ctx)

        # Safety fallbacks
        if choice.finish_reason in ("stop", "length", "content_filter"):
            text = (last_text or "").strip()
            ctx.turns.append(Turn(role="user", content=message))
            if text:
                ctx.turns.append(Turn(role="assistant", content=text))
            return StepResult(reply=text, context=ctx)
