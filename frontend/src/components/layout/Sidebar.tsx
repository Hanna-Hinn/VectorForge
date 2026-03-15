import { NavLink } from "react-router-dom";
import { t } from "../../i18n";
import { useCollections } from "../../hooks/useCollections";

function CollectionsIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 12.75V12A2.25 2.25 0 014.5 9.75h15A2.25 2.25 0 0121.75 12v.75m-8.69-6.44l-2.12-2.12a1.5 1.5 0 00-1.061-.44H4.5A2.25 2.25 0 002.25 6v12a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9a2.25 2.25 0 00-2.25-2.25h-5.379a1.5 1.5 0 01-1.06-.44z" />
    </svg>
  );
}

function EvaluationsIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
    </svg>
  );
}

function SettingsIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  );
}

function CollectionItemIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75C20.25 16.153 16.556 18 12 18s-8.25-1.847-8.25-4.125v-3.75m16.5 0c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125" />
    </svg>
  );
}

const navLinkBase = "flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-150";
const navLinkActive = "bg-brand-50 text-brand-700 shadow-sm dark:bg-brand-900/20 dark:text-brand-400";
const navLinkInactive = "text-gray-600 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-gray-200";

export default function Sidebar() {
  const { data: collections } = useCollections();

  return (
    <aside className="flex h-full w-64 flex-col border-r border-gray-200/80 bg-gradient-to-b from-gray-50 to-white dark:border-gray-700/80 dark:from-gray-900 dark:to-gray-900/95">
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-5 py-5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600 shadow-sm">
          <svg className="h-4 w-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
          </svg>
        </div>
        <span className="text-lg font-bold tracking-tight text-gray-900 dark:text-gray-100">
          VectorForge
        </span>
      </div>

      {/* Divider */}
      <div className="mx-4 border-t border-gray-200/60 dark:border-gray-700/60" />

      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1" aria-label="Main navigation">
        <NavLink
          to="/collections"
          end
          className={({ isActive }) =>
            `${navLinkBase} ${isActive ? navLinkActive : navLinkInactive}`
          }
        >
          <CollectionsIcon className="h-4.5 w-4.5 h-[18px] w-[18px] shrink-0" />
          {t("nav.collections")}
        </NavLink>

        {collections && collections.length > 0 && (
          <div className="ml-4 space-y-0.5 border-l border-gray-200 pl-3 dark:border-gray-700/60">
            {collections.map((col) => (
              <NavLink
                key={col.id}
                to={`/collections/${col.id}/query`}
                className={({ isActive }) =>
                  `flex items-center gap-2 truncate rounded-md px-2.5 py-1.5 text-xs transition-all duration-150 ${
                    isActive
                      ? "bg-brand-50 font-medium text-brand-700 dark:bg-brand-900/20 dark:text-brand-400"
                      : "text-gray-500 hover:bg-gray-100 hover:text-gray-700 dark:text-gray-500 dark:hover:bg-gray-800 dark:hover:text-gray-300"
                  }`
                }
              >
                <CollectionItemIcon className="h-3.5 w-3.5 shrink-0 opacity-60" />
                {col.name}
              </NavLink>
            ))}
          </div>
        )}

        <div className="pt-3">
          <p className="mb-2 px-3 text-[10px] font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-600">
            Tools
          </p>
          <NavLink
            to="/evaluations"
            className={({ isActive }) =>
              `${navLinkBase} ${isActive ? navLinkActive : navLinkInactive}`
            }
          >
            <EvaluationsIcon className="h-[18px] w-[18px] shrink-0" />
            {t("nav.evaluations")}
          </NavLink>

          <NavLink
            to="/settings"
            className={({ isActive }) =>
              `${navLinkBase} ${isActive ? navLinkActive : navLinkInactive}`
            }
          >
            <SettingsIcon className="h-[18px] w-[18px] shrink-0" />
            {t("nav.settings")}
          </NavLink>
        </div>
      </nav>

      {/* Footer */}
      <div className="border-t border-gray-200/60 px-5 py-3 dark:border-gray-700/60">
        <p className="text-[10px] text-gray-400 dark:text-gray-600">
          VectorForge RAG Engine
        </p>
      </div>
    </aside>
  );
}
