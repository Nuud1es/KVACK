import { useCallback, useEffect, useState } from "react";
import { api, clearKey, getKey, setUnauthorizedHandler } from "./api/client";
import { Login } from "./components/Login";
import { Admin } from "./pages/Admin";
import { Approvals } from "./pages/Approvals";
import { Chat } from "./pages/Chat";
import { Documents } from "./pages/Documents";

const TABS = [
  { id: "chat", label: "Ask" },
  { id: "documents", label: "Documents" },
  { id: "approvals", label: "Approvals" },
  { id: "admin", label: "Accounts" },
] as const;
type Tab = (typeof TABS)[number]["id"];

function tabFromHash(): Tab {
  const id = window.location.hash.replace("#", "");
  return TABS.find((t) => t.id === id)?.id ?? "chat";
}

export function App() {
  const [user, setUser] = useState<string | null>(null);
  const [admin, setAdmin] = useState(false);
  const [checking, setChecking] = useState(Boolean(getKey()));
  const [tab, setTab] = useState<Tab>(tabFromHash);

  const signOut = useCallback(() => {
    clearKey();
    setUser(null);
    setAdmin(false);
  }, []);

  useEffect(() => {
    setUnauthorizedHandler(signOut);
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [signOut]);

  // A key kept from earlier in this tab: check it is still good.
  useEffect(() => {
    if (!getKey()) return;
    void api.GET("/me").then(({ data }) => {
      setUser(data?.user ?? null);
      setAdmin(data?.admin ?? false);
      setChecking(false);
    });
  }, []);

  // Right after signing in we only know the name: ask again for what this person may open.
  useEffect(() => {
    if (!user) return;
    void api.GET("/me").then(({ data }) => setAdmin(data?.admin ?? false));
  }, [user]);

  if (checking) return <p className="center muted">Checking your sign-in…</p>;
  if (!user) return <Login onSignedIn={setUser} />;

  return (
    <>
      <header className="top">
        <strong>KVACK</strong>
        <nav aria-label="Main">
          {TABS.filter((t) => t.id !== "admin" || admin).map((t) => (
            <a
              key={t.id}
              href={`#${t.id}`}
              className={t.id === tab ? "active" : ""}
              aria-current={t.id === tab ? "page" : undefined}
            >
              {t.label}
            </a>
          ))}
        </nav>
        <span className="who">
          {user} · <button className="link" onClick={signOut}>Sign out</button>
        </span>
      </header>
      <main>
        {tab === "chat" && <Chat />}
        {tab === "documents" && <Documents />}
        {tab === "approvals" && <Approvals me={user} />}
        {tab === "admin" && admin && <Admin me={user} />}
      </main>
    </>
  );
}
