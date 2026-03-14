/** Collections API calls. */

import { get, post, del } from "./client";
import type { Collection } from "../types/models";
import type {
  CollectionListResponse,
  CreateCollectionRequest,
  MessageResponse,
} from "../types/api";

export function listCollections(): Promise<CollectionListResponse> {
  return get<CollectionListResponse>("/collections");
}

export function getCollection(id: string): Promise<Collection> {
  return get<Collection>(`/collections/${id}`);
}

export function createCollection(
  data: CreateCollectionRequest,
): Promise<Collection> {
  return post<Collection>("/collections", data);
}

export function deleteCollection(id: string): Promise<MessageResponse> {
  return del<MessageResponse>(`/collections/${id}`);
}
