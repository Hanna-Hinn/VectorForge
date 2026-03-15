/** Lightweight i18n — English-only, ready for additional locales. */

const en: Record<string, string> = {
  // Common
  "common.loading": "Loading…",
  "common.close": "Close",
  "common.delete": "Delete",
  "common.cancel": "Cancel",
  "common.save": "Save",
  "common.create": "Create",
  "common.confirm": "Confirm",
  "common.search": "Search",
  "common.noData": "No data available",
  "common.error": "Something went wrong",

  // Navigation
  "nav.collections": "Collections",
  "nav.query": "Query",
  "nav.analytics": "Analytics",
  "nav.settings": "Settings",

  // Collections
  "collections.title": "Collections",
  "collections.create.title": "Create Collection",
  "collections.create.name": "Collection Name",
  "collections.create.description": "Description",
  "collections.delete.title": "Delete Collection",
  "collections.delete.confirm": "Are you sure you want to delete this collection? This action cannot be undone.",
  "collections.empty": "No collections yet. Create one to get started.",

  // Documents
  "documents.title": "Documents",
  "documents.upload.title": "Upload Document",
  "documents.upload.chooseFile": "Choose a file",
  "documents.upload.submit": "Upload",
  "documents.upload.dragDrop": "Drag & drop your file here",
  "documents.upload.or": "or",
  "documents.upload.browse": "browse to choose",
  "documents.upload.remove": "Remove file",
  "documents.upload.accepted": "Accepted",
  "documents.empty": "No documents in this collection.",
  "documents.empty.hint": "Upload a document to get started with your knowledge base.",

  // Query
  "query.title": "Query",
  "query.placeholder": "Type your question…",
  "query.send": "Send",
  "query.sources": "Sources",
  "query.empty": "Ask a question to get started.",

  // Analytics
  "analytics.title": "Analytics",
  "analytics.totalQueries": "Total Queries",
  "analytics.uniqueQueries": "Unique Queries",
  "analytics.avgLatency": "Avg Latency",
  "analytics.p95Latency": "p95 Latency",
  "analytics.topQueries": "Top Queries",
  "analytics.noStats": "No analytics data yet.",

  // Evaluation
  "nav.evaluations": "Evaluations",
  "evaluation.title": "Evaluation Dashboard",
  "evaluation.noRuns": "No evaluation runs yet.",
  "evaluation.runNow": "Run Evaluation",
  "evaluation.status": "Status",
  "evaluation.samples": "Samples",
  "evaluation.completedAt": "Completed",
  "evaluation.recentRuns": "Recent Runs",
  "evaluation.runId": "Run ID",

  // Settings
  "settings.title": "Settings",
  "settings.providers": "Providers",
  "settings.health": "System Health",

  // Status
  "status.healthy": "Healthy",
  "status.degraded": "Degraded",
  "status.unhealthy": "Unhealthy",
};

/**
 * Look up a translation key. Returns the key itself if not found.
 */
export function t(key: string): string {
  return en[key] ?? key;
}
