import "./App.css";
import AppRoutes from "./routes";
import { BrowserRouter as Router, useLocation } from "react-router-dom";

function Layout() {
    const location = useLocation();
    const isAdmin = location.pathname.startsWith("/admin");

    return (
        <div style={{ maxHeight: "100vh", overflow: "hidden" }}>
            {isAdmin && (
                <div
                    style={{
                        width: "100%",
                        background: "#282c34",
                        color: "white",
                        padding: "10px 0",
                        textAlign: "center",
                        fontSize: "1.2em",
                        marginBottom: "3px",
                    }}
                >
                    <a
                        href="/"
                        style={{
                            color: "white",
                            margin: "0 20px",
                            textDecoration: "none",
                        }}
                    >
                        Home
                    </a>
                    <a
                        href="/admin/users"
                        style={{
                            color: "white",
                            margin: "0 20px",
                            textDecoration: "none",
                        }}
                    >
                        Users
                    </a>
                    <a
                        href="/admin/meetings"
                        style={{
                            color: "white",
                            margin: "0 20px",
                            textDecoration: "none",
                        }}
                    >
                        Meetings
                    </a>
                    <a
                        href="/admin/deliveries"
                        style={{
                            color: "white",
                            margin: "0 20px",
                            textDecoration: "none",
                        }}
                    >
                        Deliveries
                    </a>
                </div>
            )}
            <AppRoutes />
        </div>
    );
}

function App() {
    return (
        <Router>
            <Layout />
        </Router>
    );
}

export default App;

