from .model import runAgent


def talkToAgent(userText: str, history: list | None = None) -> str:
    return runAgent({"text": userText}, decisionOnly=True, history=history)


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
            print(ans)
        except Exception as e:
            print("error:", e)


if __name__ == "__main__":
    chatLoop()
