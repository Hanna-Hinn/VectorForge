import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import Toast from "../ui/Toast";

export default function Layout() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-y-auto">
        <Outlet />
      </main>
      <Toast />
    </div>
  );
}
