import { useState, type FormEvent } from "react";
import { api, clearKey, errorMessage, setKey } from "../api/client";

type Mode = "signin" | "register" | "key";

export function Login({ onSignedIn }: { onSignedIn: (user: string) => void }) {
  const [mode, setMode] = useState<Mode>("signin");
  const [name, setName] = useState("");
  const [pin, setPin] = useState("");
  const [key, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  function switchTo(next: Mode) {
    setMode(next);
    setError(null);
    setPin("");
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    if (mode === "key") {
      setKey(key.trim());
      const { data, error: err, response } = await api.GET("/me");
      setBusy(false);
      if (data) {
        onSignedIn(data.user);
      } else {
        clearKey();
        setError(response?.status === 401 ? "That key was not recognised." : errorMessage(err));
      }
      return;
    }
    const body = { name: name.trim(), pin };
    const { data, error: err } = await (mode === "register"
      ? api.POST("/auth/register", { body })
      : api.POST("/auth/login", { body }));
    setBusy(false);
    if (data) {
      setKey(data.token); // the sign-in token travels in the same header as an API key
      onSignedIn(data.user);
    } else {
      setPin("");
      setError(errorMessage(err));
    }
  }

  const pinOk = /^\d{4}$/.test(pin);
  return (
    <form className="login card" onSubmit={submit}>
      <h1>KVACK</h1>
      <p className="muted">Knowledge, Verification &amp; Automation Compliance Kit</p>
      {mode === "key" ? (
        <>
          <p className="muted">Sign in with an API key.</p>
          <label>
            API key
            <input type="password" value={key} onChange={(e) => setValue(e.target.value)} autoComplete="off" autoFocus />
          </label>
        </>
      ) : (
        <>
          <p className="muted">
            {mode === "register"
              ? "Create your account: a name and a 4-digit PIN. Approvals are recorded under your name."
              : "Sign in with your name and PIN."}
          </p>
          <label>
            Name
            <input value={name} onChange={(e) => setName(e.target.value)} autoComplete="username" autoFocus />
          </label>
          <label>
            PIN (4 digits)
            <input
              type="password"
              inputMode="numeric"
              maxLength={4}
              value={pin}
              onChange={(e) => setPin(e.target.value.replace(/\D/g, ""))}
              autoComplete={mode === "register" ? "new-password" : "current-password"}
            />
          </label>
        </>
      )}
      {error && <p role="alert" className="error">{error}</p>}
      <button
        type="submit"
        disabled={busy || (mode === "key" ? !key.trim() : !name.trim() || !pinOk)}
      >
        {busy ? "Checking…" : mode === "register" ? "Create account" : "Sign in"}
      </button>
      <p className="muted">
        {mode === "signin" && (
          <>
            New here? <button type="button" className="link" onClick={() => switchTo("register")}>Create an account</button>
            {" · "}
            <button type="button" className="link" onClick={() => switchTo("key")}>Use an API key</button>
          </>
        )}
        {mode !== "signin" && (
          <button type="button" className="link" onClick={() => switchTo("signin")}>Back to sign in</button>
        )}
      </p>
    </form>
  );
}
