from .model import runAgent


def talkToAgent(userText: str, history: list) -> str:
    response = runAgent({"text": userText}, decisionOnly=True, history=history)
    history.append({"role": "user", "content": userText})
    history.append({"role": "assistant", "content": response})
    return response


def chatLoop():
    print("type your message (or /quit, /reset):")
    history: list[dict] = []
    while True:
        userText = input("> ").strip()
        if not userText:
            continue
        cmd = userText.lower()
        if cmd in ("/quit", "/exit"):
            break
        if cmd == "/reset":
            history.clear()
            print("history cleared")
            continue
        try:
            ans = talkToAgent(userText, history)
            print(ans)
        except Exception as e:
            print("error:", e)


if __name__ == "__main__":
    chatLoop()
