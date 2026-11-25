import { useCallback, useEffect, useState } from "react";
import { api, errorMessage, type Schemas } from "../api/client";
import { NameReview } from "../components/NameReview";

type Account = Schemas["AdminUser"];

const when = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : "–");

/** Accounts: who has signed up, who is locked out, and the fixes (new PIN, unlock, delete). */
export function Admin({ me }: { me: string }) {
  const [users, setUsers] = useState<Account[] | null>(null);
  const [filter, setFilter] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [resetting, setResetting] = useState<number | null>(null);
  const [pin, setPin] = useState("");

  const load = useCallback(async () => {
    const { data, error: e } = await api.GET("/admin/users");
    if (data) {
      setUsers(data);
      setError(null);
    } else setError(errorMessage(e));
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function run(done: string, call: () => Promise<{ error?: unknown }>) {
    setError(null);
    setNotice(null);
    const { error: e } = await call();
    if (e) setError(errorMessage(e));
    else setNotice(done);
    setResetting(null);
    setPin("");
    await load();
  }

  const shown = (users ?? []).filter((u) => u.name.toLowerCase().includes(filter.trim().toLowerCase()));
  return (
    <section>
      <h1>Accounts</h1>
      <p className="muted">
        People create their own account on the sign-in page. Here you can set a new PIN for someone who forgot theirs,
        lift a lock, sign them out, or delete the account. Every change is recorded in the audit log.
      </p>
      {error && <p role="alert" className="error">{error}</p>}
      {notice && <p role="status" className="muted">{notice}</p>}
      <label>
        Find a person
        <input value={filter} onChange={(e) => setFilter(e.target.value)} />
      </label>
      {users === null ? (
        <p className="muted">Loading…</p>
      ) : shown.length === 0 ? (
        <p className="muted">{users.length ? "Nobody matches." : "Nobody has signed up yet."}</p>
      ) : (
        <table>
          <thead>
            <tr><th>Name</th><th>Signed up</th><th>Last sign-in</th><th>Status</th><th></th></tr>
          </thead>
          <tbody>
            {shown.map((u) => (
              <tr key={u.id}>
                <td>{u.name}{u.admin && <span className="badge good"> admin</span>}</td>
                <td>{when(u.created_at)}</td>
                <td>{when(u.last_sign_in)}</td>
                <td>
                  {u.locked ? <span className="badge bad">locked until {when(u.locked_until)}</span>
                    : u.failed_attempts > 0 ? <span className="badge warn">{u.failed_attempts} wrong PIN(s)</span>
                    : u.sessions > 0 ? <span className="badge good">signed in</span> : "–"}
                </td>
                <td>
                  {resetting === u.id ? (
                    <span className="row">
                      <input
                        aria-label={`New PIN for ${u.name}`}
                        type="password"
                        inputMode="numeric"
                        maxLength={4}
                        value={pin}
                        onChange={(e) => setPin(e.target.value.replace(/\D/g, ""))}
                      />
                      <button
                        disabled={pin.length !== 4}
                        onClick={() => void run(`New PIN set for ${u.name}.`, () => api.POST("/admin/users/{user_id}/pin", { params: { path: { user_id: u.id } }, body: { pin } }))}
                      >
                        Set PIN
                      </button>
                      <button className="link" onClick={() => { setResetting(null); setPin(""); }}>Cancel</button>
                    </span>
                  ) : (
                    <span className="row">
                      <button className="link" onClick={() => { setResetting(u.id); setPin(""); }}>New PIN</button>
                      {u.locked && (
                        <button className="link" onClick={() => void run(`${u.name} is unlocked.`, () => api.POST("/admin/users/{user_id}/unlock", { params: { path: { user_id: u.id } } }))}>Unlock</button>
                      )}
                      {u.sessions > 0 && (
                        <button className="link" onClick={() => void run(`${u.name} is signed out.`, () => api.POST("/admin/users/{user_id}/sign-out", { params: { path: { user_id: u.id } } }))}>Sign out</button>
                      )}
                      {u.name !== me && (
                        <button
                          className="link"
                          onClick={() => {
                            if (window.confirm(`Delete the account for ${u.name}? They will have to sign up again.`))
                              void run(`${u.name} was deleted.`, () => api.DELETE("/admin/users/{user_id}", { params: { path: { user_id: u.id } } }));
                          }}
                        >
                          Delete
                        </button>
                      )}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <NameReview />
    </section>
  );
}
