import { useState } from "react";
import type { Citation, Source } from "../../types";
import "./Sources.css";

interface Props {
  sources: Source[];
  citations?: Citation[];
}

export default function Sources({ sources, citations = [] }: Props) {
  const [expanded, setExpanded] = useState(false);

  if (!sources.length) return null;

  const visibleSources = expanded ? sources : sources.slice(0, 2);

  return (
    <div className="sources">
      <button
        className="sources-toggle"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
      >
        <BookmarkIcon />
        <span>
          {sources.length} {sources.length === 1 ? "source" : "sources"}
        </span>
        <ChevronIcon rotated={expanded} />
      </button>

      {expanded && (
        <ul className="sources-list">
          {visibleSources.map((s, i) => {
            const citation = citations.find((item) => item.chunk_id === s.chunk_id);
            return (
              <li key={`${s.chunk_id}-${i}`} className="source-item">
                <div className="source-header">
                  <span className="source-name">{formatSourceName(s.source)}</span>
                  {s.page !== undefined && (
                    <span className="source-page">p. {s.page}</span>
                  )}
                  {s.score !== undefined && (
                    <span className="source-score">{Math.round(s.score * 100)}%</span>
                  )}
                </div>
                <p className="source-excerpt">{citation?.excerpt ?? s.text}</p>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function formatSourceName(source: string) {
  const parts = source.split("/");
  return parts[parts.length - 1] || source;
}

function BookmarkIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
      <path d="M3 2h7a1 1 0 011 1v8l-4.5-2L2 11V3a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.1" strokeLinejoin="round" />
    </svg>
  );
}

function ChevronIcon({ rotated }: { rotated: boolean }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 12 12"
      fill="none"
      style={{
        transform: rotated ? "rotate(180deg)" : "rotate(0deg)",
        transition: "transform 180ms ease",
      }}
    >
      <path d="M2.5 4.5L6 8l3.5-3.5" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
