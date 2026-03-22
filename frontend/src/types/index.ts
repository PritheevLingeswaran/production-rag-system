export interface Citation {
  id?: string;
  document_id: string | null;
  chunk_id: string;
  source: string;
  page: number;
  excerpt: string;
  score?: number | null;
  created_at?: string;
}

export interface Source {
  chunk_id: string;
  source: string;
  page: number;
  score: number;
  text: string;
}

export interface Message {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  confidence?: number | null;
  refusal?: boolean;
  latency_ms?: number | null;
  created_at: string;
  metadata?: Record<string, unknown>;
  citations?: Citation[];
  sources?: Source[];
}

export interface Session {
  id: string;
  owner_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface SessionDetail extends Session {
  messages: Message[];
}

export interface Document {
  id: string;
  filename: string;
  stored_path: string;
  file_type: string;
  size_bytes: number;
  pages: number;
  chunks_created: number;
  upload_time: string;
  indexing_status: string;
  summary_status: string;
  collection_name?: string | null;
  error_message?: string | null;
  metadata: Record<string, unknown>;
}

export interface UploadResponseItem {
  id: string;
  filename: string;
  stored_path: string;
  file_type: string;
  size_bytes: number;
  indexing_status: string;
  summary_status: string;
  upload_time: string;
}

export interface DocumentSummary {
  document_id: string;
  status: string;
  title?: string | null;
  summary?: string | null;
  key_insights: string[];
  important_points: string[];
  topics: string[];
  keywords: string[];
  error_message?: string | null;
  method?: string | null;
  generated_at?: string | null;
}

export interface Settings {
  app_name: string;
  environment: string;
  default_generation_model: string;
  default_embedding_model: string;
  vector_store_provider: string;
  auth_enabled: boolean;
  auth_provider: string;
  summaries_enabled: boolean;
  default_retrieval_mode: string;
}

export interface QueryResponse {
  answer: string;
  session_id: string;
  confidence: number;
  refusal: {
    is_refusal: boolean;
    reason: string;
  };
  citations: Citation[];
  sources: Source[];
  timing: {
    total_latency_ms?: number | null;
    retrieval_latency_ms?: number | null;
    rerank_latency_ms?: number | null;
    generation_latency_ms?: number | null;
    embedding_tokens?: number | null;
    embedding_cost_usd?: number | null;
    llm_tokens_in?: number | null;
    llm_tokens_out?: number | null;
    llm_cost_usd?: number | null;
  };
}

export type View = "chat" | "documents" | "settings" | "login" | "signup";
