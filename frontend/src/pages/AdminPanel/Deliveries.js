import React, { useEffect, useState } from "react";
import { Box, Button, Paper, Stack, TextField, Typography } from "@mui/material";
import { buildApiUrl } from "../../config";
import "../../App.css";

export function Deliveries() {
    const [deliveries, setDeliveries] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const [newDelivery, setNewDelivery] = useState({
        recipient: "",
        company: "",
    });

    const fetchDeliveries = async (recipient = "", company = "") => {
        try {
            setLoading(true);
            let url = buildApiUrl("/deliveries");
            const params = [];
            if (recipient)
                params.push(`recipient=${encodeURIComponent(recipient)}`);
            if (company) params.push(`company=${encodeURIComponent(company)}`);
            if (params.length) url += "?" + params.join("&");

            const response = await fetch(url, { method: "GET" });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            setDeliveries(data.data);
        } catch (e) {
            setError("Could not fetch data.");
            console.error("Fetching error: ", e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchDeliveries();
    }, []);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setNewDelivery({ ...newDelivery, [name]: value });
    };

    const handleAdd = async (e) => {
        e.preventDefault();

        if (!newDelivery.recipient || !newDelivery.company) {
            alert("Please fill out all fields.");
            return;
        }

        try {
            const response = await fetch(buildApiUrl("/deliveries"), {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(newDelivery),
            });

            if (!response.ok) {
                throw new Error("Failed to add delivery");
            }

            setNewDelivery({ recipient: "", company: "" });
            fetchDeliveries();
        } catch (e) {
            setError("Could not add delivery.");
            console.error("Adding delivery error: ", e);
        }
    };

    if (loading) {
        return <div>Loading deliveries...</div>;
    }

    if (error) {
        return <div>Error: {error}</div>;
    }
    return (
        <div className="App">
            <header className="App-header">
                <h1>AI Concierge</h1>
                <h2>Deliveries</h2>
                <div className="content-container">
                    <Paper elevation={3} sx={{ p: 2, mb: 3 }} className="add-user-form">
                        <Typography variant="h6" component="h3" gutterBottom>
                            Add New Delivery
                        </Typography>
                        <Box component="form" onSubmit={handleAdd} noValidate>
                            <Stack spacing={2}>
                                <TextField
                                    label="Recipient"
                                    name="recipient"
                                    value={newDelivery.recipient}
                                    onChange={handleInputChange}
                                    required
                                    fullWidth
                                />
                                <TextField
                                    label="Company"
                                    name="company"
                                    value={newDelivery.company}
                                    onChange={handleInputChange}
                                    required
                                    fullWidth
                                />
                                <Button type="submit" variant="contained">
                                    Add Delivery
                                </Button>
                            </Stack>
                        </Box>
                    </Paper>

                    <div className="user-list">
                        <h3>Current Deliveries</h3>
                        {deliveries.length > 0 ? (
                            deliveries.map((delivery) => (
                                <p key={delivery.id}>
                                    <strong>{delivery.recipient}</strong> -
                                    Company: {delivery.company}
                                </p>
                            ))
                        ) : (
                            <p>No deliveries found.</p>
                        )}
                    </div>
                </div>
            </header>
        </div>
    );
}
