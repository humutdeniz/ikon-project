import React, { useEffect, useState } from "react";
import { Box, Button, Paper, Stack, TextField, Typography } from "@mui/material";
import { buildApiUrl } from "../../config";
import "../../App.css";

export function Meetings() {
    const [meetings, setMeetings] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const [newMeeting, setNewMeeting] = useState({
        host: "",
        guest: "",
        date: "",
    });

    const fetchMeetings = async (host, guest, date) => {
        try {
            setLoading(true);

            let url = buildApiUrl("/meetings");
            const params = [];
            if (host !== undefined) params.push(`host=${encodeURIComponent(host)}`);
            if (guest !== undefined) params.push(`guest=${encodeURIComponent(guest)}`);
            if (date !== undefined) params.push(`date=${encodeURIComponent(date)}`);
            if (params.length) url += "?" + params.join("&");

            const response = await fetch(url, { method: "GET" });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            setMeetings(data.data);
            console.log(data);
        } catch (e) {
            setError("Could not fetch data.");
            console.error("Fetching error: ", e);
        } finally {
            setLoading(false);
        }
    };
    useEffect(() => {
        fetchMeetings();
    }, []);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setNewMeeting({ ...newMeeting, [name]: value });
    };

    const handleAdd = async (e) => {
        e.preventDefault();

        if (!newMeeting.host || !newMeeting.guest || !newMeeting.date) {
            alert("Please fill out all fields.");
            return;
        }

        try {
            const response = await fetch(buildApiUrl("/meetings"), {
                method: "POST",
                body: JSON.stringify(newMeeting),
                headers: {
                    "Content-Type": "application/json",
                },
            });

            if (!response.ok) {
                throw new Error("Failed to add meeting");
            }

            setNewMeeting({ host: "", guest: "", date: "" });
            fetchMeetings();
        } catch (e) {
            setError("Could not add meeting.");
            console.error("Adding meeting error: ", e);
        }
    };

    if (loading) {
        return <div>Loading meetings...</div>;
    }

    if (error) {
        return <div>Error: {error}</div>;
    }
    return (
        <div className="App">
            <header className="App-header">
                <h1>AI Concierge</h1>
                <h2>Meetings</h2>
                <div className="content-container">
                    <Paper elevation={3} sx={{ p: 2, mb: 3 }} className="add-user-form">
                        <Typography variant="h6" component="h3" gutterBottom>
                            Add New Meeting
                        </Typography>
                        <Box component="form" onSubmit={handleAdd} noValidate>
                            <Stack spacing={2}>
                                <TextField
                                    label="Host"
                                    name="host"
                                    value={newMeeting.host}
                                    onChange={handleInputChange}
                                    required
                                    fullWidth
                                />
                                <TextField
                                    label="Guest"
                                    name="guest"
                                    value={newMeeting.guest}
                                    onChange={handleInputChange}
                                    required
                                    fullWidth
                                />
                                <TextField
                                    label="Date and Time"
                                    name="date"
                                    type="datetime-local"
                                    value={newMeeting.date}
                                    onChange={handleInputChange}
                                    InputLabelProps={{ shrink: true }}
                                    required
                                    fullWidth
                                />
                                <Button type="submit" variant="contained">
                                    Add Meeting
                                </Button>
                            </Stack>
                        </Box>
                    </Paper>

                    <div className="user-list">
                        <h3>Current Meetings</h3>
                        {meetings.length > 0 ? (
                            meetings.map((meeting) => (
                                <p key={meeting.id}>
                                    <strong>{meeting.host}</strong> meets{" "}
                                    <strong>{meeting.guest}</strong> on{" "}
                                    {meeting.date}
                                </p>
                            ))
                        ) : (
                            <p>No meetings found.</p>
                        )}
                    </div>
                </div>
            </header>
        </div>
    );
}
