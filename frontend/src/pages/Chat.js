import React, { useCallback, useEffect, useRef, useState } from "react";
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
import MicIcon from "@mui/icons-material/Mic";
import MicOffIcon from "@mui/icons-material/MicOff";
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
    const [isRecording, setIsRecording] = useState(false);
    const [isTranscribing, setIsTranscribing] = useState(false);
    const [liveTranscript, setLiveTranscript] = useState("");
    const [recorderError, setRecorderError] = useState("");
    const controllerRef = useRef(null);
    const listBottomRef = useRef(null);
    const audioContextRef = useRef(null);
    const processorRef = useRef(null);
    const mediaStreamRef = useRef(null);
    const bufferedChunksRef = useRef([]);
    const flushTimerRef = useRef(null);
    const voiceSessionIdRef = useRef(null);
    const isRecordingRef = useRef(false);
    const isSendingRef = useRef(false);
    const voiceBaseInputRef = useRef("");
    const historyClearTimerRef = useRef(null);

    const canSend = input.trim().length > 0 && !isStreaming && !isRecording;

    useEffect(() => {
        listBottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, isStreaming]);

    useEffect(() => {
        if (!voiceSessionIdRef.current) {
            voiceSessionIdRef.current = crypto.randomUUID();
        }
    }, []);

    useEffect(() => {
        isRecordingRef.current = isRecording;
    }, [isRecording]);

    const clearHistoryTimer = useCallback(() => {
        if (historyClearTimerRef.current) {
            window.clearTimeout(historyClearTimerRef.current);
            historyClearTimerRef.current = null;
        }
    }, []);

    const scheduleHistoryClear = useCallback(() => {
        clearHistoryTimer();
        historyClearTimerRef.current = window.setTimeout(() => {
            setMessages([]);
            historyClearTimerRef.current = null;
        }, 10000);
    }, [clearHistoryTimer, setMessages]);

    useEffect(() => {
        return () => {
            clearHistoryTimer();
        };
    }, [clearHistoryTimer]);

    const addMessage = (msg) => {
        setMessages((prev) => [...prev, { ...msg, id: crypto.randomUUID() }]);
    };

    const updateLastAssistant = (chunk) => {
        setMessages((prev) => {
            const next = [...prev];
            for (let i = next.length - 1; i >= 0; i--) {
                if (next[i].role === "assistant" && !next[i].finalized) {
                    next[i] = {
                        ...next[i],
                        content: (next[i].content || "") + chunk,
                    };
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
                content:
                    typeof result === "string"
                        ? result
                        : JSON.stringify(result, null, 2),
                meta: { toolName: toolName || "toolResult", toolCallId },
            },
        ]);
    };

    const floatToInt16 = (float32) => {
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i += 1) {
            const s = Math.max(-1, Math.min(1, float32[i]));
            int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        return int16;
    };

    const cleanupRecording = useCallback(async () => {
        if (flushTimerRef.current) {
            window.clearInterval(flushTimerRef.current);
            flushTimerRef.current = null;
        }
        if (processorRef.current) {
            try {
                processorRef.current.disconnect();
            } catch (err) {
                console.warn("processor disconnect failed", err);
            }
            processorRef.current.onaudioprocess = null;
            processorRef.current = null;
        }
        if (audioContextRef.current) {
            try {
                await audioContextRef.current.close();
            } catch (err) {
                console.warn("audio context close failed", err);
            }
            audioContextRef.current = null;
        }
        if (mediaStreamRef.current) {
            mediaStreamRef.current.getTracks().forEach((track) => track.stop());
            mediaStreamRef.current = null;
        }
    }, []);

    const sendAudioChunk = useCallback(
        async (arrayBuffer, finalize = false) => {
            if (!voiceSessionIdRef.current) {
                voiceSessionIdRef.current = crypto.randomUUID();
            }

            const sessionId = voiceSessionIdRef.current;
            const sampleRate = Math.round(
                audioContextRef.current?.sampleRate || 48000
            );

            try {
                const resp = await fetch(`${apiBase}/speech/stream`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/octet-stream",
                        "X-Session-Id": sessionId,
                        "X-Sample-Rate": String(sampleRate),
                        "X-Finalize": finalize ? "true" : "false",
                    },
                    body: arrayBuffer,
                });

                if (!resp.ok) {
                    throw new Error(`STT request failed (${resp.status})`);
                }

                const data = await resp.json();
                const text = data?.text ?? "";
                setLiveTranscript(text);

                if (isRecordingRef.current || finalize) {
                    setInput(() => {
                        const base = voiceBaseInputRef.current || "";
                        const combined = text
                            ? base
                                ? `${base} ${text}`
                                : text
                            : base;
                        return combined.trimStart();
                    });
                }

                if (finalize) {
                    setIsTranscribing(false);
                    voiceBaseInputRef.current = "";
                    voiceSessionIdRef.current = crypto.randomUUID();
                }
            } catch (err) {
                console.error("speech chunk error", err);
                setRecorderError(err?.message || String(err));
                await cleanupRecording();
                bufferedChunksRef.current = [];
                isRecordingRef.current = false;
                setIsRecording(false);
                setIsTranscribing(false);
            } finally {
                isSendingRef.current = false;
            }
        },
        [cleanupRecording]
    );

    const waitFor = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    const flushAudioBuffer = useCallback(
        async (finalize = false) => {
            if (!finalize && isSendingRef.current) {
                return;
            }

            if (finalize) {
                while (isSendingRef.current) {
                    // Wait for in-flight request before sending final chunk.
                    // eslint-disable-next-line no-await-in-loop
                    await waitFor(25);
                }
            }

            const chunks = bufferedChunksRef.current;
            if (!chunks.length && !finalize) {
                return;
            }

            let payload;
            if (chunks.length) {
                const totalLength = chunks.reduce(
                    (acc, chunk) => acc + chunk.length,
                    0
                );
                const combined = new Int16Array(totalLength);
                let offset = 0;
                chunks.forEach((chunk) => {
                    combined.set(chunk, offset);
                    offset += chunk.length;
                });
                payload = combined.buffer;
            } else {
                payload = new ArrayBuffer(0);
            }

            bufferedChunksRef.current = [];
            isSendingRef.current = true;
            await sendAudioChunk(payload, finalize);
        },
        [sendAudioChunk]
    );

    const startRecording = useCallback(async () => {
        if (isRecordingRef.current || isStreaming) {
            return;
        }

        setRecorderError("");
        setLastError("");
        voiceBaseInputRef.current = input;

        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: true,
            });
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtx) {
                throw new Error(
                    "Web Audio API is not supported in this browser."
                );
            }

            const audioContext = new AudioCtx();
            const source = audioContext.createMediaStreamSource(mediaStream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);

            bufferedChunksRef.current = [];
            processor.onaudioprocess = (event) => {
                const inputBuffer = event.inputBuffer.getChannelData(0);
                const pcm = floatToInt16(inputBuffer);
                bufferedChunksRef.current.push(pcm);
            };

            source.connect(processor);
            processor.connect(audioContext.destination);

            mediaStreamRef.current = mediaStream;
            audioContextRef.current = audioContext;
            processorRef.current = processor;
            voiceSessionIdRef.current = crypto.randomUUID();

            flushTimerRef.current = window.setInterval(() => {
                flushAudioBuffer(false).catch((err) =>
                    console.error("flush audio", err)
                );
            }, 750);

            setIsRecording(true);
            isRecordingRef.current = true;
            setIsTranscribing(true);
            setLiveTranscript("");

            if (audioContext.state === "suspended") {
                await audioContext.resume();
            }
        } catch (err) {
            console.error("start recording failed", err);
            setRecorderError(err?.message || String(err));
            await cleanupRecording();
            bufferedChunksRef.current = [];
            isRecordingRef.current = false;
            setIsRecording(false);
            setIsTranscribing(false);
        }
    }, [cleanupRecording, flushAudioBuffer, input, isStreaming]);

    const stopRecording = useCallback(async () => {
        if (!isRecordingRef.current) {
            return;
        }

        isRecordingRef.current = false;
        setIsRecording(false);

        if (flushTimerRef.current) {
            window.clearInterval(flushTimerRef.current);
            flushTimerRef.current = null;
        }

        if (processorRef.current) {
            processorRef.current.onaudioprocess = null;
        }

        try {
            await flushAudioBuffer(true);
        } catch (err) {
            console.error("finalize audio failed", err);
            setRecorderError(err?.message || String(err));
        } finally {
            bufferedChunksRef.current = [];
            await cleanupRecording();
            setIsTranscribing(false);
        }
    }, [cleanupRecording, flushAudioBuffer]);

    useEffect(() => {
        return () => {
            if (isRecordingRef.current) {
                stopRecording().catch(() => {});
            } else {
                cleanupRecording().catch(() => {});
            }
        };
    }, [cleanupRecording, stopRecording]);

    const handleSubmit = async (e) => {
        e?.preventDefault();
        if (!canSend) return;

        clearHistoryTimer();
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

        controllerRef.current = new AbortController();

        const resp = await fetch(`${apiBase}/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
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
            if (data.audio && data.audioMimeType) {
                const audio = new Audio(
                    `data:${data.audioMimeType};base64,${data.audio}`
                );
                audio
                    .play()
                    .catch((err) => console.error("Audio play failed", err));
            }
            updateLastAssistant(data?.reply ?? data?.content ?? "");
            finalizeAssistant();
            if (data && typeof data === "object" && data.historyCleared) {
                scheduleHistoryClear();
            }
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
        if (data && typeof data === "object" && data.historyCleared) {
            scheduleHistoryClear();
        }
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
                setLastError(
                    typeof data === "string" ? data : data?.message || "error"
                );
                break;
            }
            case "done": {
                finalizeAssistant();
                break;
            }
            default: {
                updateLastAssistant(
                    typeof data === "string" ? data : data?.text || ""
                );
            }
        }
    };

    return (
        <Box sx={{ height: "100vh", display: "flex", flexDirection: "column" }}>
            <AppBar position="static" color="default" elevation={1}>
                <Toolbar>
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                        Chat
                    </Typography>
                    {isStreaming ? (
                        <Chip
                            label={`Streaming via ${transport}`}
                            size="small"
                        />
                    ) : isRecording ? (
                        <Chip label="Listening" color="primary" size="small" />
                    ) : isTranscribing ? (
                        <Chip label="Transcribing" size="small" />
                    ) : (
                        <Chip label="Idle" size="small" />
                    )}
                </Toolbar>
            </AppBar>

            {(isStreaming || isTranscribing) && <LinearProgress />}

            <Container
                maxWidth="md"
                sx={{ flex: 1, py: 2, overflow: "scroll" }}
            >
                <Paper
                    variant="outlined"
                    sx={{
                        height: "100%",
                        display: "flex",
                        flexDirection: "column",
                    }}
                >
                    <List sx={{ flex: 1, overflowY: "auto", p: 2 }}>
                        {messages.map((m) => (
                            <ListItem key={m.id} sx={{ display: "block" }}>
                                <Stack
                                    spacing={1}
                                    alignItems={
                                        m.role === "user"
                                            ? "flex-end"
                                            : "flex-start"
                                    }
                                >
                                    <Typography
                                        variant="caption"
                                        color="text.secondary"
                                    >
                                        {m.role.toUpperCase()}{" "}
                                        {m.meta?.toolName
                                            ? `· ${m.meta.toolName}`
                                            : ""}
                                    </Typography>

                                    {m.role === "tool" ? (
                                        <Card
                                            variant="outlined"
                                            sx={{
                                                maxWidth: "100%",
                                                bgcolor: "#f9fafb",
                                            }}
                                        >
                                            <CardContent>
                                                <Typography
                                                    variant="body2"
                                                    component="pre"
                                                    style={{
                                                        whiteSpace: "pre-wrap",
                                                    }}
                                                >
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
                                                bgcolor:
                                                    m.role === "user"
                                                        ? "#e3f2fd"
                                                        : "#fff",
                                                border: "1px solid",
                                                borderColor: "divider",
                                            }}
                                        >
                                            <Typography
                                                variant="body1"
                                                sx={{ wordBreak: "break-word" }}
                                            >
                                                <ReactMarkdown
                                                    remarkPlugins={[remarkGfm]}
                                                >
                                                    {m.content}
                                                </ReactMarkdown>
                                            </Typography>
                                        </Paper>
                                    )}
                                </Stack>
                            </ListItem>
                        ))}
                        <div ref={listBottomRef} />
                    </List>

                    <Divider />

                    <Box
                        component="form"
                        onSubmit={handleSubmit}
                        sx={{ p: 2, display: "flex", gap: 1 }}
                    >
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
                            helperText={
                                isRecording
                                    ? `🎙 ${liveTranscript || "Listening..."}`
                                    : undefined
                            }
                            FormHelperTextProps={{ sx: { marginLeft: 0 } }}
                            InputProps={{
                                endAdornment: (
                                    <InputAdornment
                                        position="end"
                                        sx={{ gap: 0.5 }}
                                    >
                                        {isRecording ? (
                                            <IconButton
                                                onClick={() => stopRecording()}
                                                aria-label="stop recording"
                                                color="error"
                                                size="small"
                                            >
                                                <MicOffIcon />
                                            </IconButton>
                                        ) : (
                                            <IconButton
                                                onClick={() => startRecording()}
                                                aria-label="start recording"
                                                disabled={
                                                    isStreaming ||
                                                    isTranscribing
                                                }
                                                size="small"
                                            >
                                                <MicIcon />
                                            </IconButton>
                                        )}
                                        {isStreaming ? (
                                            <IconButton
                                                onClick={stopStreaming}
                                                aria-label="stop"
                                                size="small"
                                            >
                                                <StopIcon />
                                            </IconButton>
                                        ) : (
                                            <IconButton
                                                type="submit"
                                                aria-label="send"
                                                disabled={!canSend}
                                                size="small"
                                            >
                                                <SendIcon />
                                            </IconButton>
                                        )}
                                    </InputAdornment>
                                ),
                            }}
                        />
                        {!isStreaming && (
                            <Button
                                variant="outlined"
                                onClick={() => setTransport("auto")}
                                startIcon={<ReplayIcon />}
                            >
                                Reset
                            </Button>
                        )}
                    </Box>
                </Paper>

                {lastError && (
                    <Typography color="error" sx={{ mt: 1 }}>
                        {lastError}
                    </Typography>
                )}
                {recorderError && (
                    <Typography color="error" sx={{ mt: lastError ? 0.5 : 1 }}>
                        {recorderError}
                    </Typography>
                )}
            </Container>
        </Box>
    );
}
