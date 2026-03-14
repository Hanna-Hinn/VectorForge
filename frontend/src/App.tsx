import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/layout/Layout";
import CollectionsPage from "./pages/CollectionsPage";
import DocumentsPage from "./pages/DocumentsPage";
import QueryPage from "./pages/QueryPage";
import AnalyticsPage from "./pages/AnalyticsPage";
import SettingsPage from "./pages/SettingsPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/collections" replace />} />
          <Route path="collections" element={<CollectionsPage />} />
          <Route
            path="collections/:collectionId/documents"
            element={<DocumentsPage />}
          />
          <Route
            path="collections/:collectionId/query"
            element={<QueryPage />}
          />
          <Route
            path="collections/:collectionId/analytics"
            element={<AnalyticsPage />}
          />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
