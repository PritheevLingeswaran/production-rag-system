import { api } from "./client";
import type { Document, DocumentSummary, UploadResponseItem } from "../types";

export const documentsApi = {
  list: () =>
    api.get<{ documents: Document[] }>("/documents").then((r) => r.documents),

  upload: (file: File) => {
    const form = new FormData();
    form.append("files", file);
    return api
      .upload<{ documents: UploadResponseItem[] }>("/documents/upload", form)
      .then((response) => response.documents[0]);
  },

  get: (id: string) => api.get<Document>(`/documents/${id}`),

  summary: (id: string) => api.get<DocumentSummary>(`/documents/${id}/summary`),

  delete: (id: string) => api.delete<Document>(`/documents/${id}`),

  waitForIndexing: async (id: string, timeoutMs = 20_000) => {
    const startedAt = Date.now();

    while (Date.now() - startedAt < timeoutMs) {
      const document = await api.get<Document>(`/documents/${id}`);
      if (document.indexing_status === "ready" || document.indexing_status === "failed") {
        return document;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 900));
    }

    return api.get<Document>(`/documents/${id}`);
  },
};
