/** React Query hooks for collections CRUD. */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import * as collectionsApi from "../api/collections";
import type { CreateCollectionRequest } from "../types/api";

const QUERY_KEY = ["collections"] as const;

export function useCollections() {
  return useQuery({
    queryKey: QUERY_KEY,
    queryFn: () => collectionsApi.listCollections(),
    select: (data) => data.collections,
  });
}

export function useCollection(id: string) {
  return useQuery({
    queryKey: [...QUERY_KEY, id],
    queryFn: () => collectionsApi.getCollection(id),
    enabled: !!id,
  });
}

export function useCreateCollection() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: CreateCollectionRequest) =>
      collectionsApi.createCollection(data),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: QUERY_KEY });
    },
  });
}

export function useDeleteCollection() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => collectionsApi.deleteCollection(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: QUERY_KEY });
    },
  });
}
