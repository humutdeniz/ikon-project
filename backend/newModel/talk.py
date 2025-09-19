from typing import Any, Dict, List

try:
    from .model import runAgent  # package import
    from .utility import setHistoryResetFlag
except ImportError:
    from model import runAgent  # direct script execution
    from utility import setHistoryResetFlag


_agent_history: List[Dict[str, str]] = []
_MAX_HISTORY_PAIRS = 8


def _append_history(role: str, content: str) -> None:
    text = (content or "").strip()
    if not text:
        return
    _agent_history.append({"role": role, "content": text})
    max_messages = _MAX_HISTORY_PAIRS * 2
    if len(_agent_history) > max_messages:
        del _agent_history[: len(_agent_history) - max_messages]


def _prepare_history(history: list | None) -> List[Dict[str, str]]:
    if history is not None:
        return [dict(item) for item in history if isinstance(item, dict)]
    return list(_agent_history)


def talkToAgent(userText: str, isReset: bool, history: list | None = None) -> Dict[str, Any]:
    agent_resp = runAgent({"text": userText}, history=_prepare_history(history))
    if isReset:
        setHistoryResetFlag()
    if not isinstance(agent_resp, dict):
        agent_resp = {"reply": str(agent_resp or ""), "history_was_reset": False}

    if history is None:
        if agent_resp.get("history_reset_before_call") or agent_resp.get(
            "history_reset_during_call"
        ):
            _agent_history.clear()

        if not agent_resp.get("history_reset_during_call"):
            _append_history("user", userText)
            _append_history("assistant", agent_resp.get("reply", ""))

    return agent_resp

def chatLoop():
    print("type your message (or /quit):")
    while True:
        userText = input("> ").strip()
        if not userText:
            continue
        cmd = userText.lower()
        if cmd in ("/quit", "/exit"):
            break
        try:
            ans = talkToAgent(userText)
            print(ans.get("reply", ans))
        except Exception as e:
            print("error:", e)


if __name__ == "__main__":
    chatLoop()
