import "../../App.css";

export function AdminPanel() {
    return (
        <div className="App">
            <header className="App-header">
                <h1>Welcome to the Admin Panel</h1>
                <nav style={{ margin: "20px 0" }}>
                    <a
                        href="/users"
                        style={{ marginRight: "20px", color: "white" }}
                    >
                        Users
                    </a>
                    <a
                        href="/meetings"
                        style={{ marginRight: "20px", color: "white" }}
                    >
                        Meetings
                    </a>
                    <a href="/deliveries" style={{ color: "white" }}>
                        Deliveries
                    </a>
                </nav>
                <p>Select a section from the navigation menu.</p>
            </header>
        </div>
    );
}
