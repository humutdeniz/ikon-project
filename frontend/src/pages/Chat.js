import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  AppBar,
  Box,
  Button,
  Container,
  Divider,
  IconButton,
  InputAdornment,
  LinearProgress,
  List,
  ListItem,
  Paper,
  Stack,
  TextField,
  Toolbar,
  Typography,
  Card,
  CardContent,
  Chip,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import StopIcon from "@mui/icons-material/Stop";
import ReplayIcon from "@mui/icons-material/Replay";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Configure your backend base URL (FastAPI)
// The backend exposes routes under the /api prefix
const apiBase = "http://localhost:5000/api";

/**
 * Chat message shape
 * id: string
 * role: "user" | "assistant" | "tool"
 * content: string
 * meta?: { toolName?: string; toolCallId?: string; }
 */

export function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [lastError, setLastError] = useState("");
  const [transport, setTransport] = useState("auto"); // "auto" | "sse" | "fetchStream" | "json"
  const controllerRef = useRef(null);
  const listBottomRef = useRef(null);

  const canSend = input.trim().length > 0 && !isStreaming;

  useEffect(() => {
    listBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  const addMessage = (msg) => {
    setMessages((prev) => [...prev, { ...msg, id: crypto.randomUUID() }]);
  };

  const updateLastAssistant = (chunk) => {
    setMessages((prev) => {
      const next = [...prev];
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].role === "assistant" && !next[i].finalized) {
          next[i] = { ...next[i], content: (next[i].content || "") + chunk };
          break;
        }
      }
      return next;
    });
  };

  const finalizeAssistant = () => {
    setMessages((prev) => {
      const next = [...prev];
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].role === "assistant" && !next[i].finalized) {
          next[i] = { ...next[i], finalized: true };
          break;
        }
      }
      return next;
    });
  };

  const addToolCall = ({ toolName, toolCallId, args }) => {
    setMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: "tool",
        content: JSON.stringify(args, null, 2),
        meta: { toolName, toolCallId },
      },
    ]);
  };

  const addToolResult = ({ toolName, toolCallId, result }) => {
    setMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: "tool",
        content: typeof result === "string" ? result : JSON.stringify(result, null, 2),
        meta: { toolName: toolName || "toolResult", toolCallId },
      },
    ]);
  };

  const handleSubmit = async (e) => {
    e?.preventDefault();
    if (!canSend) return;

    const userText = input.trim();
    setInput("");
    setLastError("");

    addMessage({ role: "user", content: userText });
    // Prepare assistant message shell to stream into
    addMessage({ role: "assistant", content: "", finalized: false });

    try {
      await streamChat(userText);
    } catch (err) {
      setLastError(err?.message || String(err));
    }
  };

  const stopStreaming = () => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setIsStreaming(false);
    finalizeAssistant();
  };

  const streamChat = async (userText) => {
    setIsStreaming(true);

    // POST /api/chat — backend expects { message, history }
    controllerRef.current = new AbortController();

    const resp = await fetch(`${apiBase}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // Add Authorization here if your API needs it
        // Authorization: `Bearer ${yourToken}`,
      },
      body: JSON.stringify({
        // Backend contract: message (current input) + prior history
        message: userText,
        history: messages
          .filter((m) => m.role === "user" || m.role === "assistant")
          .map((m) => ({ role: m.role, content: m.content })),
      }),
      signal: controllerRef.current.signal,
    });

    const ct = resp.headers.get("content-type") || "";

    if (ct.includes("text/event-stream")) {
      setTransport("fetchStream");
      await consumeSse(resp);
    } else if (ct.includes("application/json")) {
      setTransport("json");
      const data = await resp.json();
      // Prefer FastAPI reply field; fallback to content
      updateLastAssistant(data?.reply ?? data?.content ?? "");
      finalizeAssistant();
    } else {
      // Fallback: plain text
      setTransport("json");
      const txt = await resp.text();
      updateLastAssistant(txt);
      finalizeAssistant();
    }

    setIsStreaming(false);
  };

  // Minimal SSE parser for fetch Response.body (not EventSource)
  const consumeSse = async (resp) => {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buf = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const rawEvent = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const evt = parseSseEvent(rawEvent);
        if (evt) handleSseEvent(evt);
      }
    }

    if (buf.trim().length) {
      const evt = parseSseEvent(buf);
      if (evt) handleSseEvent(evt);
    }

    finalizeAssistant();
  };

  const parseSseEvent = (raw) => {
    const lines = raw.split(/\r?\n/);
    let event = "message";
    let data = "";
    for (const line of lines) {
      if (line.startsWith(":")) continue; // comment/heartbeat
      const [k, ...rest] = line.split(":");
      const v = rest.join(":").trimStart();
      if (k === "event") event = v;
      if (k === "data") data += v + "\n";
    }
    try {
      const parsed = data ? JSON.parse(data) : {};
      return { event, data: parsed };
    } catch {
      return { event, data: { text: data } };
    }
  };

  const handleSseEvent = ({ event, data }) => {
    switch (event) {
      case "token": {
        updateLastAssistant(data?.content || data?.text || "");
        break;
      }
      case "message": {
        updateLastAssistant(data?.content || data?.text || "");
        break;
      }
      case "tool_call": {
        addToolCall({
          toolName: data?.toolName || "toolCall",
          toolCallId: data?.toolCallId,
          args: data?.args ?? {},
        });
        break;
      }
      case "tool_result": {
        addToolResult({
          toolName: data?.toolName || "toolResult",
          toolCallId: data?.toolCallId,
          result: data?.result,
        });
        break;
      }
      case "error": {
        setLastError(typeof data === "string" ? data : data?.message || "error");
        break;
      }
      case "done": {
        finalizeAssistant();
        break;
      }
      default: {
        updateLastAssistant(typeof data === "string" ? data : data?.text || "");
      }
    }
  };

  return (
    <Box sx={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      <AppBar position="static" color="default" elevation={1}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Chat Frontend
          </Typography>
          {isStreaming ? (
            <Chip label={`Streaming via ${transport}`} size="small" />
          ) : (
            <Chip label="Idle" size="small" />
          )}
        </Toolbar>
      </AppBar>

      {isStreaming && <LinearProgress />}

      <Container maxWidth="md" sx={{ flex: 1, py: 2, overflow: "hidden" }}>
        <Paper variant="outlined" sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
          <List sx={{ flex: 1, overflowY: "auto", p: 2 }}>
            {messages.map((m) => (
              <ListItem key={m.id} sx={{ display: "block" }}>
                <Stack spacing={1} alignItems={m.role === "user" ? "flex-end" : "flex-start"}>
                  <Typography variant="caption" color="text.secondary">
                    {m.role.toUpperCase()} {m.meta?.toolName ? `· ${m.meta.toolName}` : ""}
                  </Typography>

                  {m.role === "tool" ? (
                    <Card variant="outlined" sx={{ maxWidth: "100%", bgcolor: "#f9fafb" }}>
                      <CardContent>
                        <Typography variant="body2" component="pre" style={{ whiteSpace: "pre-wrap" }}>
                          {m.content}
                        </Typography>
                      </CardContent>
                    </Card>
                  ) : (
                    <Paper
                      elevation={0}
                      sx={{
                        px: 2,
                        py: 1.25,
                        maxWidth: "100%",
                        bgcolor: m.role === "user" ? "#e3f2fd" : "#fff",
                        border: "1px solid",
                        borderColor: "divider",
                      }}
                    >
                      <Typography variant="body1" sx={{ wordBreak: "break-word" }}>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                      </Typography>
                    </Paper>
                  )}
                </Stack>
              </ListItem>
            ))}
            <div ref={listBottomRef} />
          </List>

          <Divider />

          <Box component="form" onSubmit={handleSubmit} sx={{ p: 2, display: "flex", gap: 1 }}>
            <TextField
              fullWidth
              disabled={isStreaming}
              placeholder="Type a message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  handleSubmit(e);
                }
              }}
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    {isStreaming ? (
                      <IconButton onClick={stopStreaming} aria-label="stop" size="small">
                        <StopIcon />
                      </IconButton>
                    ) : (
                      <IconButton type="submit" aria-label="send" disabled={!canSend} size="small">
                        <SendIcon />
                      </IconButton>
                    )}
                  </InputAdornment>
                ),
              }}
            />
            {!isStreaming && (
              <Button variant="outlined" onClick={() => setTransport("auto")} startIcon={<ReplayIcon />}>Reset</Button>
            )}
          </Box>
        </Paper>

        {lastError && (
          <Typography color="error" sx={{ mt: 1 }}>
            {lastError}
          </Typography>
        )}
      </Container>
    </Box>
  );
}
