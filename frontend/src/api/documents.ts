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
};
