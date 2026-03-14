import { NavLink } from "react-router-dom";
import { t } from "../../i18n";
import { useCollections } from "../../hooks/useCollections";

export default function Sidebar() {
  const { data: collections } = useCollections();

  return (
    <aside className="flex h-full w-60 flex-col border-r border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-900">
      <div className="flex items-center gap-2 px-4 py-5">
        <span className="text-xl font-bold text-brand-600">VectorForge</span>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-2" aria-label="Main navigation">
        <NavLink
          to="/collections"
          end
          className={({ isActive }) =>
            `mb-1 flex items-center rounded-md px-3 py-2 text-sm font-medium transition-colors ${
              isActive
                ? "bg-brand-50 text-brand-700 dark:bg-brand-900/20 dark:text-brand-400"
                : "text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
            }`
          }
        >
          {t("nav.collections")}
        </NavLink>

        {collections && collections.length > 0 && (
          <div className="ml-3 mt-1 space-y-0.5">
            {collections.map((col) => (
              <NavLink
                key={col.id}
                to={`/collections/${col.id}/query`}
                className={({ isActive }) =>
                  `block truncate rounded px-3 py-1.5 text-xs transition-colors ${
                    isActive
                      ? "bg-brand-50 text-brand-700 dark:bg-brand-900/20"
                      : "text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800"
                  }`
                }
              >
                {col.name}
              </NavLink>
            ))}
          </div>
        )}

        <NavLink
          to="/settings"
          className={({ isActive }) =>
            `mt-4 flex items-center rounded-md px-3 py-2 text-sm font-medium transition-colors ${
              isActive
                ? "bg-brand-50 text-brand-700 dark:bg-brand-900/20 dark:text-brand-400"
                : "text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
            }`
          }
        >
          {t("nav.settings")}
        </NavLink>
      </nav>
    </aside>
  );
}
