# VectorForge — Frontend (React UI)

> Developer reference for the React single-page application that provides a GUI for VectorForge.

---

## Table of Contents

- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
  - [Routing](#routing)
  - [Data Fetching](#data-fetching)
  - [API Client](#api-client)
  - [State Management](#state-management)
- [Pages](#pages)
  - [Collections](#collections)
  - [Documents](#documents)
  - [Query](#query)
  - [Analytics](#analytics)
  - [Settings](#settings)
- [Component Library](#component-library)
  - [UI Components](#ui-components)
  - [Feature Components](#feature-components)
  - [Layout Components](#layout-components)
- [Hooks](#hooks)
- [Type System](#type-system)
- [Internationalization](#internationalization)
- [Styling](#styling)
- [Configuration](#configuration)
- [Build & Deploy](#build--deploy)

---

## Overview

The frontend is a standalone React SPA in the `frontend/` directory. It communicates with the VectorForge REST API and provides:

- **Collection management** — create, browse, and delete collections
- **Document ingestion** — upload documents into collections
- **RAG query interface** — chat-style streaming queries with source citations
- **Analytics dashboard** — query volume, latency charts, top queries
- **System health** — live status checks and provider information

---

## Technology Stack

| Category | Technology | Version |
|----------|-----------|---------|
| Framework | React | 19.0 |
| Language | TypeScript | 5.7 |
| Build | Vite | 6.0 |
| Routing | React Router | 7.1 |
| Server State | TanStack React Query | 5.60 |
| Styling | Tailwind CSS | 3.4 |
| Linting | ESLint | 9.17 |

---

## Getting Started

```bash
# Prerequisites: Node.js 18+
cd frontend
npm install
npm run dev          # http://localhost:5173
```

The Vite dev server proxies `/api` requests to `http://127.0.0.1:8000` (the Python backend).

### Available Scripts

| Script | Command | Description |
|--------|---------|-------------|
| `dev` | `vite` | Start dev server with HMR |
| `build` | `tsc -b && vite build` | Type-check + production build |
| `preview` | `vite preview` | Preview production build |
| `lint` | `eslint .` | Run ESLint |

---

## Project Structure

```
frontend/
├── index.html                # HTML entry point
├── package.json
├── vite.config.ts            # Vite + proxy config
├── tsconfig.json             # TypeScript strict config
├── tailwind.config.ts        # Brand colors, content paths
├── postcss.config.js         # Tailwind + autoprefixer
└── src/
    ├── main.tsx              # React root + QueryClientProvider
    ├── App.tsx               # Router + route definitions
    ├── index.css             # Tailwind directives + CSS variables
    ├── api/                  # HTTP client layer
    │   ├── client.ts         # Base fetch wrapper (get/post/del)
    │   ├── collections.ts    # Collection CRUD
    │   ├── documents.ts      # Document CRUD + ingest
    │   ├── query.ts          # Query + SSE streaming
    │   └── analytics.ts      # Analytics endpoints
    ├── hooks/                # React Query hooks
    │   ├── useCollections.ts # Collection queries & mutations
    │   ├── useDocuments.ts   # Document queries & mutations
    │   ├── useQuery.ts       # Streaming query hook
    │   └── useAnalytics.ts   # Analytics queries
    ├── types/                # TypeScript interfaces
    │   ├── models.ts         # Domain models (Collection, Document, etc.)
    │   └── api.ts            # Request/response DTOs
    ├── utils/
    │   └── format.ts         # formatDate, formatNumber, formatBytes, formatMs
    ├── i18n/
    │   └── index.ts          # Translation key lookup (t function)
    ├── components/
    │   ├── ui/               # Reusable primitives
    │   │   ├── Badge.tsx
    │   │   ├── Button.tsx
    │   │   ├── Card.tsx
    │   │   ├── Dialog.tsx
    │   │   ├── Input.tsx
    │   │   ├── Spinner.tsx
    │   │   ├── Table.tsx
    │   │   └── Toast.tsx
    │   ├── features/         # Domain-specific components
    │   │   ├── ChatMessage.tsx
    │   │   ├── CollectionCard.tsx
    │   │   ├── DocumentRow.tsx
    │   │   ├── LatencyChart.tsx
    │   │   ├── SourceCard.tsx
    │   │   └── UploadDialog.tsx
    │   └── layout/           # App shell
    │       ├── Header.tsx
    │       ├── Layout.tsx
    │       └── Sidebar.tsx
    └── pages/                # Route-level views
        ├── CollectionsPage.tsx
        ├── DocumentsPage.tsx
        ├── QueryPage.tsx
        ├── AnalyticsPage.tsx
        └── SettingsPage.tsx
```

---

## Architecture

### Routing

The app uses React Router v7 with a nested layout:

```
/                               → Redirect to /collections
/collections                    → CollectionsPage
/collections/:collectionId/documents → DocumentsPage
/collections/:collectionId/query     → QueryPage
/collections/:collectionId/analytics → AnalyticsPage
/settings                       → SettingsPage
```

All routes are wrapped in the `<Layout>` component that renders the sidebar and header.

### Data Fetching

TanStack React Query manages all server state:

- **Stale time**: 30 seconds (`staleTime: 30_000`)
- **Retry**: 1 attempt on failure
- **Refetch**: Analytics page refreshes every 30 seconds
- **Cache invalidation**: Mutations automatically invalidate related query keys

### API Client

The base client (`src/api/client.ts`) wraps `fetch()` with:

- Automatic `Content-Type: application/json` headers
- Optional `X-Api-Key` header (set via `setApiKey()`)
- Typed `ApiError` class with status code and detail message
- Separate `get<T>()`, `post<T>()`, `del<T>()` convenience functions

All requests go to `/api/*` which Vite proxies to the backend in development.

### State Management

- **Server state**: Entirely managed by React Query (collections, documents, analytics)
- **Local state**: React `useState` for UI concerns (dialogs, form inputs, streaming content)
- **No global client state store** — React Query's cache serves as the single source of truth

---

## Pages

### Collections

**Route**: `/collections`

- Lists all collections as cards in a responsive grid
- "Create Collection" button opens a dialog with fields for name, description, embedding provider/model, and chunking settings
- Each card shows collection name, description, creation date, and action buttons
- "Documents", "Query", and "Analytics" links navigate to collection sub-pages
- Delete triggers a confirmation dialog

### Documents

**Route**: `/collections/:collectionId/documents`

- Paginated table of documents in the collection
- "Ingest Document" button opens upload dialog (source URI based)
- Each row shows source URI, content type, status badge, size, and creation date
- Delete removes a document and its embeddings

### Query

**Route**: `/collections/:collectionId/query`

- Chat-style interface for RAG queries
- Messages stream in via SSE (Server-Sent Events)
- User messages appear on the right, assistant responses on the left
- Source citations displayed below answers with relevance scores
- Real-time latency indicator after generation completes
- "Clear" button resets the conversation

### Analytics

**Route**: `/collections/:collectionId/analytics`

- Summary cards: total queries, unique queries, average latency, p95 latency
- Latency chart with statistical breakdown (min, max, p50, p95)
- Top queries table with frequency counts
- Auto-refreshes every 30 seconds

### Settings

**Route**: `/settings`

- System health panel — deep probe results for each component (database, pgvector, etc.)
- Provider information — lists registered embedding and LLM providers
- API key configuration

---

## Component Library

### UI Components

Generic, reusable primitives with no domain logic:

| Component | Props | Description |
|-----------|-------|-------------|
| `Button` | `variant`, `size`, `loading` | Primary/secondary/danger/ghost with loading spinner |
| `Input` | `label`, `error` | Text input with label and error message |
| `Card` | `title`, `className` | Container with optional title |
| `Dialog` | `open`, `onClose`, `title`, `actions` | Modal overlay with backdrop click to close |
| `Badge` | `variant` | Status indicator (default/success/warning/error) |
| `Spinner` | `size`, `className` | Animated loading spinner (sm/md/lg) |
| `Table` | `columns`, `data`, `keyExtractor` | Generic typed table with column renderers |
| `Toast` | — | Toast notification system via `showToast()` function |

### Feature Components

Domain-specific components composed from UI primitives:

| Component | Props | Description |
|-----------|-------|-------------|
| `CollectionCard` | `collection`, `onDelete` | Card displaying collection metadata and actions |
| `DocumentRow` | `document`, `onDelete` | Table row with document info and status badge |
| `ChatMessage` | `message` | User or assistant chat bubble |
| `SourceCard` | `sources` | Expandable source citations from RAG results |
| `UploadDialog` | `open`, `onClose`, `onSubmit`, `loading` | Dialog for document ingestion |
| `LatencyChart` | `stats` | Visual latency statistics breakdown |

### Layout Components

| Component | Props | Description |
|-----------|-------|-------------|
| `Layout` | — | Root shell: sidebar + header + `<Outlet>` |
| `Sidebar` | — | Navigation links and collection list |
| `Header` | `title` | Page title bar |

---

## Hooks

Custom hooks encapsulate all data fetching logic:

| Hook | Returns | Description |
|------|---------|-------------|
| `useCollections()` | `Collection[]` | List all collections |
| `useCollection(id)` | `Collection` | Single collection detail |
| `useCreateCollection()` | Mutation | Create a new collection |
| `useDeleteCollection()` | Mutation | Delete a collection |
| `useDocuments(collectionId, limit, offset)` | `DocumentListResponse` | Paginated documents |
| `useDocument(id)` | `DocumentDetailResponse` | Single document detail |
| `useIngestDocument(collectionId)` | Mutation | Ingest a document |
| `useDeleteDocument(collectionId)` | Mutation | Delete a document |
| `useStreamQuery()` | `{ messages, isStreaming, sendQuery, clearMessages }` | SSE streaming chat |
| `useAnalyticsSummary(collectionId, since?)` | `AnalyticsSummary` | Full analytics summary |
| `useTopQueries(collectionId, limit?, since?)` | `QueryFrequency[]` | Most frequent queries |
| `useLatencyStats(collectionId, since?)` | `LatencyStats \| null` | Latency statistics |

---

## Type System

Types are split into two files:

### `types/models.ts` — Domain Models

| Type | Description |
|------|-------------|
| `Collection` | Collection with id, name, description, configs, timestamps |
| `Document` | Document with id, collection_id, source_uri, status, metadata |
| `DocumentStatus` | `"pending" \| "processing" \| "indexed" \| "failed"` |
| `SourceCitation` | RAG source with document_source, chunk_index, score, snippet |
| `LatencyStats` | avg/min/max/p50/p95 ms and sample_count |
| `QueryFrequency` | query_text + count |
| `VolumeDataPoint` | date + count |
| `AnalyticsSummary` | Aggregated analytics data |
| `SystemHealth` | Component health statuses |
| `ProvidersInfo` | Available embedding and LLM providers |

### `types/api.ts` — Request/Response DTOs

| Type | Description |
|------|-------------|
| `CreateCollectionRequest` | Name, description, embedding/chunking options |
| `CollectionListResponse` | `{ collections: Collection[] }` |
| `IngestDocumentRequest` | Source URI, metadata, chunking options |
| `DocumentListResponse` | `{ documents: Document[], total: number }` |
| `BatchIngestResponse` | Per-document results with success/fail counts |
| `QueryRequest` | Query text, collection_id, retrieval/generation options |
| `QueryResponse` | Answer, sources, latency breakdown |
| `ErrorResponse` | `{ error: string, detail: string }` |
| `MessageResponse` | `{ message: string }` |

---

## Internationalization

The `src/i18n/index.ts` module provides a lightweight `t(key)` function:

```typescript
import { t } from "@/i18n";

<h1>{t("collections.title")}</h1>
```

- All user-facing strings are externalized as translation keys
- Currently English-only
- Adding a new locale: create a parallel record and select by `navigator.language`
- Keys follow the pattern `{section}.{context}` (e.g., `"query.placeholder"`, `"analytics.avgLatency"`)

---

## Styling

### Tailwind CSS

- **Configuration**: `tailwind.config.ts` with custom `brand` color palette (50–900)
- **Scanned paths**: `./index.html` and `./src/**/*.{ts,tsx}`
- **PostCSS**: Tailwind + autoprefixer

### CSS Variables

Defined in `src/index.css` with automatic dark mode:

| Variable | Light | Dark |
|----------|-------|------|
| `--color-bg` | `#ffffff` | `#111827` |
| `--color-text` | `#111827` | `#f9fafb` |
| `--color-border` | `#e5e7eb` | `#374151` |
| `--color-muted` | `#6b7280` | `#9ca3af` |

Dark mode activates automatically via `prefers-color-scheme: dark`.

### Accessibility

- `prefers-reduced-motion` support — animations are disabled when the user preference is set
- Semantic HTML throughout (buttons, headings, form labels)
- Focus states on interactive elements
- Color contrast meets WCAG AA standards

---

## Configuration

### Vite Config (`vite.config.ts`)

| Setting | Value | Description |
|---------|-------|-------------|
| `resolve.alias.@` | `./src` | Path alias for imports (`@/hooks/...`) |
| `server.port` | `5173` | Dev server port |
| `server.proxy./api` | `http://127.0.0.1:8000` | Proxy API calls to backend |

### TypeScript (`tsconfig.json`)

- **Strict mode** enabled (`strict: true`)
- `noUnusedLocals`, `noUnusedParameters`, `noUncheckedIndexedAccess`
- Path alias: `@/*` → `src/*`
- Target: ES2020, JSX: react-jsx

---

## Build & Deploy

```bash
# Production build (outputs to frontend/dist/)
cd frontend
npm run build

# Preview the production build locally
npm run preview
```

The `dist/` directory contains static files ready to be served by any HTTP server or CDN. In production, configure your reverse proxy to:

1. Serve `dist/` for all non-API routes
2. Proxy `/api/*` to the VectorForge server
