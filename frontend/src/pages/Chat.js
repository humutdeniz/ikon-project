import React, { useEffect, useMemo, useRef, useState } from "react";
import "../App.css";
import { useNavigate } from "react-router-dom";
import {
    Box,
    Button,
    Container,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Paper,
    Stack,
    TextField,
    Typography,
    CircularProgress,
    Divider,
} from "@mui/material";

export function Chat() {
    const [history, setHistory] = useState([]);
    const [context, setContext] = useState({});
    const [message, setMessage] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const [pwdOpen, setPwdOpen] = useState(false);
    const [pwd, setPwd] = useState("");
    const [pwdErr, setPwdErr] = useState("");

    const navigate = useNavigate();
    const listRef = useRef(null);

    const apiBase = useMemo(() => "http://localhost:5000", []);

    useEffect(() => {
        if (listRef.current) {
            listRef.current.scrollTop = listRef.current.scrollHeight;
        }
    }, [history, loading]);

    const sendMessage = async (e) => {
        e?.preventDefault?.();
        const text = message.trim();
        if (!text || loading) return;
        setLoading(true);
        setError(null);
        try {
            const res = await fetch(`${apiBase}/api/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text, history, context }),
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            setHistory(data.history || []);
            if (data.context) setContext(data.context);
            setMessage("");
        } catch (err) {
            console.error(err);
            setError("Mesaj gönderilirken bir hata oluştu.");
        } finally {
            setLoading(false);
        }
    };

    const openAdminDialog = () => {
        setPwd("");
        setPwdErr("");
        setPwdOpen(true);
    };
    const closeAdminDialog = () => setPwdOpen(false);
    const submitAdmin = () => {
        if (pwd === "2728") {
            setPwdOpen(false);
            navigate("/admin");
        } else {
            setPwdErr("Hatalı şifre");
        }
    };

    return (
        <Container maxWidth="md" sx={{ py: 2, height: "calc(100vh - 70px)" }}>
            <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
                <Stack direction="row" alignItems="center" justifyContent="space-between">
                    <Typography variant="h5" component="h1">AI Concierge Chat</Typography>
                    <Button variant="outlined" onClick={openAdminDialog}>Admin Panel</Button>
                </Stack>
            </Paper>

            <Paper elevation={0} sx={{ p: 1, bgcolor: "background.default", height: "calc(100% - 120px)", display: "flex", flexDirection: "column", border: "1px solid", borderColor: "divider", borderRadius: 2 }}>
                <Box ref={listRef} sx={{ flex: 1, overflowY: "auto", p: 2 }}>
                    {history.length === 0 && (
                        <Typography variant="body2" color="text.secondary">
                            Sohbete başlamak için aşağıya yazın.
                        </Typography>
                    )}

                    <Stack spacing={1.2}>
                        {history.map((m, idx) => (
                            <Box key={idx} sx={{ display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start" }}>
                                <Box sx={{
                                    maxWidth: "75%",
                                    bgcolor: m.role === "user" ? "primary.main" : "grey.800",
                                    color: "common.white",
                                    px: 1.5,
                                    py: 1,
                                    borderRadius: 1.5,
                                    whiteSpace: "pre-wrap",
                                }}>
                                    <Typography variant="body2">{m.content}</Typography>
                                </Box>
                            </Box>
                        ))}
                    </Stack>

                    {loading && (
                        <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 2 }}>
                            <CircularProgress size={16} />
                            <Typography variant="caption" color="text.secondary">Yanıt bekleniyor...</Typography>
                        </Stack>
                    )}

                    {error && (
                        <Typography variant="caption" color="error" sx={{ mt: 1, display: "block" }}>{error}</Typography>
                    )}
                </Box>

                <Divider sx={{ my: 1 }} />

                <Box component="form" onSubmit={sendMessage} sx={{ display: "flex", gap: 1, p: 1 }}>
                    <TextField
                        fullWidth
                        placeholder="Mesajınızı yazın..."
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        size="small"
                        disabled={loading}
                    />
                    <Button type="submit" variant="contained" disabled={loading || !message.trim()}>
                        Gönder
                    </Button>
                </Box>
            </Paper>

            <Dialog open={pwdOpen} onClose={closeAdminDialog} fullWidth maxWidth="xs">
                <DialogTitle>Admin Paneli</DialogTitle>
                <DialogContent>
                    <TextField
                        autoFocus
                        margin="dense"
                        label="Şifre"
                        type="password"
                        fullWidth
                        value={pwd}
                        onChange={(e) => { setPwd(e.target.value); setPwdErr(""); }}
                        error={!!pwdErr}
                        helperText={pwdErr || "Giriş için şifreyi giriniz"}
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={closeAdminDialog}>İptal</Button>
                    <Button onClick={submitAdmin} variant="contained">Giriş</Button>
                </DialogActions>
            </Dialog>
        </Container>
    );
}

export default Chat;
