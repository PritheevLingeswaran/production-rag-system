import { useState } from "react";
import "./AuthView.css";

interface Props {
  mode: "login" | "signup";
  onModeChange: (mode: "login" | "signup") => void;
}

export default function AuthView({ mode, onModeChange }: Props) {
  const [submitted, setSubmitted] = useState(false);

  const title = mode === "login" ? "Welcome back" : "Create your workspace";
  const subtitle =
    mode === "login"
      ? "Sign in to continue working across your document sessions."
      : "Start a new SmartQA workspace for document search, summaries, and grounded chat.";
  const buttonLabel = mode === "login" ? "Log in" : "Create account";

  return (
    <section className="auth-view">
      <div className="auth-shell">
        <div className="auth-hero">
          <span className="auth-eyebrow">Workspace access</span>
          <h1>{title}</h1>
          <p>{subtitle}</p>
          <div className="auth-highlights">
            <div className="auth-highlight">
              <strong>Grounded answers</strong>
              <span>Search across uploaded files with verifiable citations.</span>
            </div>
            <div className="auth-highlight">
              <strong>Shared memory</strong>
              <span>Keep sessions, documents, and summaries organized in one place.</span>
            </div>
          </div>
        </div>

        <div className="auth-card">
          <div className="auth-card-header">
            <span className="auth-card-kicker">{mode === "login" ? "Account login" : "New account"}</span>
            <h2>{mode === "login" ? "Log in to SmartQA" : "Sign up for SmartQA"}</h2>
            <p>
              {mode === "login"
                ? "Use your email and password to access your workspace."
                : "Create a local account shell now. You can connect real auth later."}
            </p>
          </div>

          <form
            className="auth-form"
            onSubmit={(event) => {
              event.preventDefault();
              setSubmitted(true);
            }}
          >
            {mode === "signup" && (
              <label className="auth-field">
                <span>Full name</span>
                <input type="text" placeholder="Pritheev Lingeswaran" />
              </label>
            )}

            <label className="auth-field">
              <span>Email</span>
              <input type="email" placeholder="you@company.com" />
            </label>

            <label className="auth-field">
              <span>Password</span>
              <input type="password" placeholder={mode === "login" ? "Enter your password" : "Create a secure password"} />
            </label>

            {mode === "signup" && (
              <label className="auth-field">
                <span>Workspace name</span>
                <input type="text" placeholder="SmartQA Team" />
              </label>
            )}

            <button type="submit" className="auth-submit">
              {buttonLabel}
            </button>
          </form>

          <div className="auth-footer">
            <span>
              {mode === "login" ? "Need an account?" : "Already have an account?"}
            </span>
            <button
              type="button"
              className="auth-switch"
              onClick={() => {
                setSubmitted(false);
                onModeChange(mode === "login" ? "signup" : "login");
              }}
            >
              {mode === "login" ? "Sign up" : "Log in"}
            </button>
          </div>

          {submitted && (
            <p className="auth-note">
              {mode === "login"
                ? "UI is ready. Connect this form to your auth API when backend login is available."
                : "Account creation UI is ready. Hook this to a signup endpoint when auth is implemented."}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
