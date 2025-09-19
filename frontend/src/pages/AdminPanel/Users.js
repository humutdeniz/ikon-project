import React, { useEffect, useState } from "react";
import { Box, Button, Paper, Stack, TextField, Typography } from "@mui/material";
import { buildApiUrl } from "../../config";
import "../../App.css";

export function Users() {
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const [newUser, setNewUser] = useState({
        name: "",
        status: "",
        password: "",
    });

    const fetchUsers = async () => {
        try {
            setLoading(true);
            const response = await fetch(buildApiUrl("/users"));
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            setUsers(data.data);
        } catch (e) {
            setError("Could not fetch data.");
            console.error("Fetching error: ", e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchUsers();
    }, []);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setNewUser({ ...newUser, [name]: value });
    };

    const handleAdd = async (e) => {
        e.preventDefault();

        if (!newUser.name || !newUser.status || !newUser.password) {
            alert("Please fill out all fields.");
            return;
        }

        try {
            const response = await fetch(buildApiUrl("/users"), {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(newUser),
            });

            if (!response.ok) {
                throw new Error("Failed to add user");
            }

            setNewUser({ name: "", status: "", password: "" });
            fetchUsers();
        } catch (e) {
            setError("Could not add user.");
            console.error("Adding user error: ", e);
        }
    };
    if (loading) {
        return <div>Loading users...</div>;
    }

    if (error) {
        return <div>Error: {error}</div>;
    }
    return (
        <div className="App">
            <header className="App-header">
                <h1>AI Concierge</h1>
                <h2>Workplace Directory</h2>
                <div className="content-container">
                    <Paper elevation={3} sx={{ p: 2, mb: 3 }} className="add-user-form">
                        <Typography variant="h6" component="h3" gutterBottom>
                            Add New User
                        </Typography>
                        <Box component="form" onSubmit={handleAdd} noValidate>
                            <Stack spacing={2}>
                                <TextField
                                    label="Name"
                                    name="name"
                                    value={newUser.name}
                                    onChange={handleInputChange}
                                    required
                                    fullWidth
                                />
                                <TextField
                                    label="Status"
                                    name="status"
                                    value={newUser.status}
                                    onChange={handleInputChange}
                                    required
                                    fullWidth
                                />
                                <TextField
                                    label="Password"
                                    name="password"
                                    type="password"
                                    value={newUser.password}
                                    onChange={handleInputChange}
                                    required
                                    fullWidth
                                />
                                <Button type="submit" variant="contained">
                                    Add User
                                </Button>
                            </Stack>
                        </Box>
                    </Paper>

                    <div className="user-list">
                        <h3>Current Users</h3>
                        {users.length > 0 ? (
                            users.map((user) => (
                                <p key={user.id}>
                                    <strong>{user.name}</strong> - Status:{" "}
                                    {user.status}
                                </p>
                            ))
                        ) : (
                            <p>No users found.</p>
                        )}
                    </div>
                </div>
            </header>
        </div>
    );
}
