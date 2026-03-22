import { useEffect, useRef, useState } from "react";
import { documentsApi } from "../../api/documents";
import type { Document, DocumentSummary, UploadResponseItem } from "../../types";
import "./DocumentsView.css";

interface Props {
  refreshKey?: number;
}

export default function DocumentsView({ refreshKey = 0 }: Props) {
  const [docs, setDocs] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);
  const [summary, setSummary] = useState<DocumentSummary | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const load = async () => {
    setLoading(true);
    try {
      const data = await documentsApi.list();
      setDocs(data);
    } catch {
      // non-fatal
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [refreshKey]);

  const uploadFile = async (file: File) => {
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      const upload = await documentsApi.upload(file);
      setDocs((current) => [toDocument(upload), ...current]);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const handleFiles = (files: FileList | null) => {
    if (!files?.length) return;
    Array.from(files).forEach(uploadFile);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  const openSummary = async (doc: Document) => {
    if (selectedDoc?.id === doc.id) {
      setSelectedDoc(null);
      setSummary(null);
      return;
    }
    setSelectedDoc(doc);
    setSummary(null);
    setSummaryLoading(true);
    try {
      const s = await documentsApi.summary(doc.id);
      setSummary(s);
    } catch {
      setSummary(null);
    } finally {
      setSummaryLoading(false);
    }
  };

  const deleteDocument = async (event: React.MouseEvent, doc: Document) => {
    event.stopPropagation();
    setUploadError(null);
    setDeletingId(doc.id);
    try {
      await documentsApi.delete(doc.id);
      setDocs((current) => current.filter((item) => item.id !== doc.id));
      if (selectedDoc?.id === doc.id) {
        setSelectedDoc(null);
        setSummary(null);
      }
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Delete failed.");
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="documents-view">
      <div className="documents-header">
        <div>
          <h1 className="documents-title">Documents</h1>
          <p className="documents-sub">
            {docs.length > 0
              ? `${docs.length} uploaded document${docs.length !== 1 ? "s" : ""} available`
              : "Upload documents to start asking questions"}
          </p>
        </div>
      </div>

      <div className="documents-body">
        <div className="documents-main">
          <div
            className={`drop-zone ${dragging ? "drop-zone--active" : ""} ${uploading ? "drop-zone--uploading" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".pdf,.txt,.md,.html,.htm"
              multiple
              className="visually-hidden"
              onChange={(e) => handleFiles(e.target.files)}
              disabled={uploading}
            />
            {uploading ? (
              <>
                <UploadingIcon />
                <p className="drop-zone-label">Uploading…</p>
              </>
            ) : (
              <>
                <UploadIcon />
                <p className="drop-zone-label">
                  {dragging ? "Drop to upload" : "Drop files here or click to browse"}
                </p>
                <p className="drop-zone-hint">PDF, TXT, MD, HTML</p>
              </>
            )}
          </div>

          {uploadError && (
            <div className="upload-error">
              <AlertIcon /> {uploadError}
            </div>
          )}

          {loading ? (
            <div className="doc-list">
              {[1, 2, 3].map((i) => (
                <div key={i} className="doc-skeleton" />
              ))}
            </div>
          ) : docs.length === 0 ? (
            <div className="doc-empty">
              <FolderIcon />
              <p>No documents yet. Upload one to get started.</p>
            </div>
          ) : (
            <ul className="doc-list">
              {docs.map((doc) => (
                <li key={doc.id}>
                  <button
                    className={`doc-item ${selectedDoc?.id === doc.id ? "doc-item--selected" : ""}`}
                    onClick={() => openSummary(doc)}
                  >
                    <div className="doc-icon">
                      <FileIcon ext={fileExt(doc.filename)} />
                    </div>
                    <div className="doc-info">
                      <span className="doc-name">{doc.filename}</span>
                      <span className="doc-meta">
                        {formatSize(doc.size_bytes)}
                        {doc.pages ? ` · ${doc.pages} pages` : ""}
                        {" · "}
                        {formatDate(doc.upload_time)}
                      </span>
                      {doc.error_message && (
                        <span className="doc-error-text">{doc.error_message}</span>
                      )}
                    </div>
                    <StatusBadge status={doc.indexing_status} />
                    <button
                      className="doc-delete"
                      type="button"
                      onClick={(event) => deleteDocument(event, doc)}
                      disabled={deletingId === doc.id}
                      title="Delete document"
                      aria-label={`Delete ${doc.filename}`}
                    >
                      {deletingId === doc.id ? <SpinnerIcon /> : <TrashIcon />}
                    </button>
                    <ChevronIcon rotated={selectedDoc?.id === doc.id} />
                  </button>

                  {selectedDoc?.id === doc.id && (
                    <div className="doc-summary-panel">
                      {summaryLoading ? (
                        <div className="summary-loading">
                          <span className="status-dot" />
                          <span className="status-dot" />
                          <span className="status-dot" />
                          <span style={{ marginLeft: 8, fontSize: 13, color: "var(--text-muted)" }}>
                            Generating summary…
                          </span>
                        </div>
                      ) : summary ? (
                        <>
                          {summary.title && (
                            <p className="summary-text" style={{ fontWeight: 600 }}>{summary.title}</p>
                          )}
                          {summary.summary && (
                            <p className="summary-text">{summary.summary}</p>
                          )}
                          {summary.key_insights.length > 0 && (
                            <div className="summary-topics">
                              {summary.key_insights.map((item) => (
                                <span key={item} className="topic-tag">{item}</span>
                              ))}
                            </div>
                          )}
                          {summary.topics.length > 0 && (
                            <div className="summary-topics">
                              {summary.topics.map((t) => (
                                <span key={t} className="topic-tag">{t}</span>
                              ))}
                            </div>
                          )}
                        </>
                      ) : (
                        <p className="summary-unavailable">Summary not available.</p>
                      )}
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: Document["indexing_status"] }) {
  const tone =
    status === "ready" ? "ready" : status === "failed" ? "failed" : "processing";
  return (
    <span className={`status-badge status-badge--${tone}`}>
      {status === "ready" ? "Ready" : status === "failed" ? "Failed" : "Processing"}
    </span>
  );
}

function ChevronIcon({ rotated }: { rotated: boolean }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 14 14"
      fill="none"
      className="doc-chevron"
      style={{ transform: rotated ? "rotate(180deg)" : "none", transition: "transform 180ms ease" }}
    >
      <path d="M3 5.5L7 9.5l4-4" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function FileIcon({ ext }: { ext: string }) {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
      <path d="M4 2h7l4 4v10a1 1 0 01-1 1H4a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="var(--text-muted)" strokeWidth="1.2" />
      <path d="M11 2v4h4" stroke="var(--text-muted)" strokeWidth="1.2" strokeLinejoin="round" />
      <text x="5" y="14" fontSize="4.5" fill="var(--accent)" fontFamily="var(--font-ui)" fontWeight="600">{ext.toUpperCase()}</text>
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <path d="M14 18V10M14 10l-4 4M14 10l4 4" stroke="var(--text-muted)" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M6 20c-2 0-3.5-1.5-3.5-3.5 0-1.7 1.2-3.1 2.8-3.4A5 5 0 0114 9a5 5 0 014.7 3.1A3.5 3.5 0 0125.5 15.5 3.5 3.5 0 0122 20" stroke="var(--text-muted)" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function UploadingIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" className="spin">
      <circle cx="14" cy="14" r="10" stroke="var(--border)" strokeWidth="2" />
      <path d="M14 4a10 10 0 0110 10" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function FolderIcon() {
  return (
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
      <path d="M4 10a2 2 0 012-2h6l2 2h12a2 2 0 012 2v12a2 2 0 01-2 2H6a2 2 0 01-2-2V10z" stroke="var(--border)" strokeWidth="1.5" />
    </svg>
  );
}

function AlertIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <circle cx="7" cy="7" r="5.5" stroke="currentColor" strokeWidth="1.2" />
      <path d="M7 4.5v3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
      <circle cx="7" cy="9.5" r="0.6" fill="currentColor" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path d="M3 4h8M5.2 4V3h3.6v1M4 4l.45 6.2a1 1 0 001 .8h2.1a1 1 0 001-.8L9.2 4M5.5 6v3M8.5 6v3" stroke="currentColor" strokeWidth="1.15" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="spin">
      <circle cx="7" cy="7" r="5" stroke="currentColor" strokeOpacity="0.22" strokeWidth="1.2" />
      <path d="M7 2a5 5 0 015 5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  );
}

function fileExt(name: string) {
  return name.split(".").pop() ?? "doc";
}

function toDocument(upload: UploadResponseItem): Document {
  return {
    ...upload,
    pages: 0,
    chunks_created: 0,
    collection_name: null,
    error_message: null,
    metadata: {},
  };
}

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" });
}
