import { useEffect, useState } from "react";
import { settingsApi } from "../../api/settings";
import type { Settings } from "../../types";
import "./SettingsView.css";

const LABELS: Record<string, string> = {
  app_name: "Application",
  environment: "Environment",
  default_generation_model: "Generation model",
  default_embedding_model: "Embedding model",
  vector_store_provider: "Vector store",
  auth_enabled: "Authentication enabled",
  auth_provider: "Auth provider",
  summaries_enabled: "Summaries enabled",
  default_retrieval_mode: "Default retrieval mode",
};

const DESCRIPTIONS: Record<string, string> = {
  app_name: "Public-facing service name reported by the backend.",
  environment: "Active runtime environment for this deployment.",
  default_generation_model: "Model used to generate grounded answers.",
  default_embedding_model: "Model used for document and query embeddings.",
  vector_store_provider: "Retrieval backend used for vector search.",
  auth_enabled: "Whether API authentication is currently enforced.",
  auth_provider: "Configured authentication strategy.",
  summaries_enabled: "Whether document summaries are available.",
  default_retrieval_mode: "Fallback retrieval mode used by the API.",
};

const PRIORITY = [
  "app_name",
  "environment",
  "default_generation_model",
  "default_embedding_model",
  "default_retrieval_mode",
  "vector_store_provider",
  "auth_enabled",
  "auth_provider",
  "summaries_enabled",
];

export default function SettingsView() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    settingsApi.get()
      .then(setSettings)
      .catch((err) => setError(err.message ?? "Failed to load settings."))
      .finally(() => setLoading(false));
  }, []);

  const entries = settings ? Object.entries(settings) as Array<[keyof Settings, Settings[keyof Settings]]> : [];
  const sortedKeys = settings
    ? [
        ...PRIORITY.filter((k): k is keyof Settings => k in settings),
        ...entries.map(([key]) => key).filter((key) => !PRIORITY.includes(key)),
      ]
    : [];

  return (
    <div className="settings-view">
      <div className="settings-header">
        <h1 className="settings-title">Settings</h1>
        <p className="settings-sub">
          Current backend configuration — read-only.
        </p>
      </div>

      <div className="settings-body">
        {loading && (
          <div className="settings-loading">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="settings-skeleton" />
            ))}
          </div>
        )}

        {error && (
          <div className="settings-error">
            <AlertIcon />
            {error}
          </div>
        )}

        {settings && (
          <div className="settings-grid">
            {sortedKeys.map((key) => {
              const val = settings[key];
              if (val === undefined || val === null) return null;
              return (
                <div key={key} className="setting-row">
                  <div className="setting-info">
                    <span className="setting-label">{LABELS[key] ?? key}</span>
                    {DESCRIPTIONS[key] && (
                      <span className="setting-desc">{DESCRIPTIONS[key]}</span>
                    )}
                  </div>
                  <span className="setting-value">{String(val)}</span>
                </div>
              );
            })}
          </div>
        )}

        <div className="settings-note">
          <NoteIcon />
          <p>
            Settings are managed server-side. To change them, update your backend
            configuration and restart the service.
          </p>
        </div>
      </div>
    </div>
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

function NoteIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style={{ flexShrink: 0 }}>
      <circle cx="7" cy="7" r="5.5" stroke="currentColor" strokeWidth="1.2" />
      <path d="M7 6.5v4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
      <circle cx="7" cy="4.5" r="0.6" fill="currentColor" />
    </svg>
  );
}
