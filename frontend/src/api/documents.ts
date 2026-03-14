/** Documents API calls. */

import { get, post, del } from "./client";
import type {
  DocumentListResponse,
  DocumentDetailResponse,
  IngestDocumentRequest,
  BatchIngestResponse,
  MessageResponse,
} from "../types/api";
import type { Document } from "../types/models";

export function listDocuments(
  collectionId: string,
  limit = 50,
  offset = 0,
): Promise<DocumentListResponse> {
  return get<DocumentListResponse>(
    `/collections/${collectionId}/documents?limit=${limit}&offset=${offset}`,
  );
}

export function getDocument(id: string): Promise<DocumentDetailResponse> {
  return get<DocumentDetailResponse>(`/documents/${id}`);
}

export function ingestDocument(
  collectionId: string,
  data: IngestDocumentRequest,
): Promise<Document> {
  return post<Document>(`/collections/${collectionId}/documents`, data);
}

export function batchIngest(
  collectionId: string,
  documents: IngestDocumentRequest[],
): Promise<BatchIngestResponse> {
  return post<BatchIngestResponse>(
    `/collections/${collectionId}/documents/batch`,
    documents,
  );
}

export function deleteDocument(id: string): Promise<MessageResponse> {
  return del<MessageResponse>(`/documents/${id}`);
}
