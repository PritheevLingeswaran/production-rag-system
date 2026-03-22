import { useRef, useState, useEffect } from "react";
import "./Composer.css";

interface ComposerAttachment {
  id: string;
  fileName: string;
  fileType: string;
  status: "processing" | "ready" | "failed";
  note?: string;
  previewUrl?: string;
}

interface Props {
  onSend: (text: string) => void;
  onUpload: (file: File) => Promise<void>;
  disabled?: boolean;
  uploadDisabled?: boolean;
}

export default function Composer({
  onSend,
  onUpload,
  disabled,
  uploadDisabled = false,
}: Props) {
  const [value, setValue] = useState("");
  const [attachments, setAttachments] = useState<ComposerAttachment[]>([]);
  const attachmentsRef = useRef<ComposerAttachment[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!disabled) textareaRef.current?.focus();
  }, [disabled]);

  useEffect(() => {
    attachmentsRef.current = attachments;
  }, [attachments]);

  useEffect(() => {
    return () => {
      attachmentsRef.current.forEach((attachment) => {
        if (attachment.previewUrl) {
          URL.revokeObjectURL(attachment.previewUrl);
        }
      });
    };
  }, []);

  const autoResize = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    autoResize();
  };

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const canSend = value.trim().length > 0 && !disabled;

  const handleUploadChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const attachmentId = crypto.randomUUID();
    const isImage = file.type.startsWith("image/");
    const previewUrl = isImage ? URL.createObjectURL(file) : undefined;

    setAttachments((current) => [
      {
        id: attachmentId,
        fileName: file.name,
        fileType: file.type || file.name.split(".").pop() || "file",
        status: "processing",
        previewUrl,
      },
      ...current,
    ]);

    try {
      await onUpload(file);
      setAttachments((current) =>
        current.map((item) =>
          item.id === attachmentId
            ? { ...item, status: "ready", note: "Indexed and ready for questions" }
            : item
        )
      );
    } catch (error) {
      setAttachments((current) =>
        current.map((item) =>
          item.id === attachmentId
            ? {
                ...item,
                status: "failed",
                note:
                  error instanceof Error
                    ? error.message
                    : "This file could not be processed.",
              }
            : item
        )
      );
    }
    event.target.value = "";
  };

  return (
    <div className="composer-wrap">
      {attachments.length > 0 && (
        <div className="composer-attachments">
          {attachments.map((attachment) => (
            <div key={attachment.id} className="attachment-card">
              <div className="attachment-icon-shell">
                {attachment.previewUrl ? (
                  <img
                    className="attachment-image"
                    src={attachment.previewUrl}
                    alt={attachment.fileName}
                  />
                ) : (
                  <span className="attachment-badge">
                    {formatFileBadge(attachment.fileName)}
                  </span>
                )}
              </div>
              <div className="attachment-meta">
                <span className="attachment-name">{attachment.fileName}</span>
                <span className={`attachment-status attachment-status--${attachment.status}`}>
                  {attachment.note ||
                    (attachment.status === "processing"
                      ? "Processing document..."
                      : attachment.status === "ready"
                        ? "Indexed and ready"
                        : "Upload failed")}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="composer">
        <input
          ref={fileInputRef}
          type="file"
          className="visually-hidden"
          accept=".pdf,.txt,.md,.html,.htm"
          onChange={handleUploadChange}
        />
        <button
          className="composer-upload"
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploadDisabled}
          aria-label="Upload document"
          title="Upload document"
        >
          <PlusIcon />
        </button>
        <textarea
          ref={textareaRef}
          className="composer-input"
          placeholder="Ask a question about your documents…"
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={disabled}
          aria-label="Message input"
        />
        <button
          className={`composer-send ${canSend ? "composer-send--active" : ""}`}
          onClick={submit}
          disabled={!canSend}
          aria-label="Send"
        >
          <SendIcon />
        </button>
      </div>
      <p className="composer-hint">
        Enter to send · Shift+Enter for new line
      </p>
    </div>
  );
}

function SendIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
      <path
        d="M13 7.5L2 2l2.5 5.5L2 13l11-5.5z"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
      <path d="M7.5 3v9M3 7.5h9" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}

function formatFileBadge(fileName: string) {
  return (fileName.split(".").pop() || "file").slice(0, 4).toUpperCase();
}
