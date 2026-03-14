/** React Query hooks for document management. */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import * as documentsApi from "../api/documents";
import type { IngestDocumentRequest } from "../types/api";

function queryKey(collectionId: string) {
  return ["documents", collectionId] as const;
}

export function useDocuments(
  collectionId: string,
  limit = 50,
  offset = 0,
) {
  return useQuery({
    queryKey: [...queryKey(collectionId), limit, offset],
    queryFn: () => documentsApi.listDocuments(collectionId, limit, offset),
    enabled: !!collectionId,
  });
}

export function useDocument(id: string) {
  return useQuery({
    queryKey: ["document", id],
    queryFn: () => documentsApi.getDocument(id),
    enabled: !!id,
  });
}

export function useIngestDocument(collectionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: IngestDocumentRequest) =>
      documentsApi.ingestDocument(collectionId, data),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: queryKey(collectionId),
      });
    },
  });
}

export function useDeleteDocument(collectionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => documentsApi.deleteDocument(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: queryKey(collectionId),
      });
    },
  });
}
