import { Routes, Route } from "react-router-dom";
import { AdminPanel } from "./pages/AdminPanel";
import { Chat } from "./pages/Chat";
import { Users } from "./pages/AdminPanel/Users";
import { Meetings } from "./pages/AdminPanel/Meetings";
import { Deliveries } from "./pages/AdminPanel/Deliveries";

const AppRoutes = () => (
    <Routes>
        <Route path="/" element={<Chat />} />
        <Route path="/admin" element={<AdminPanel />} />
        <Route path="/admin/users" element={<Users />} />
        <Route path="/admin/meetings" element={<Meetings />} />
        <Route path="/admin/deliveries" element={<Deliveries />} />
    </Routes>
);

export default AppRoutes;
