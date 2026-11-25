import { useCallback, useEffect, useState } from "react";
import { api, errorMessage, type Schemas } from "../api/client";

type Item = Schemas["NameReview"];

const KIND: Record<Item["kind"], string> = { customer: "Customer", supplier: "Packaging supplier", person: "Person" };
const keyOf = (i: Item) => `${i.kind}:${i.name}`;

/**
 * Names that need a person: a spelling that may be another name's alias, or a name that may be new.
 * Each row says what looks likely; confirming "same as" makes an alias at once, "separate" remembers
 * that it is its own name so it is not suggested again.
 */
export function NameReview() {
  const [items, setItems] = useState<Item[] | null>(null);
  const [pick, setPick] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    const { data, error: e } = await api.GET("/admin/names");
    if (data) {
      setItems(data);
      setError(null);
    } else setError(errorMessage(e));
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function decide(done: string, call: () => Promise<{ error?: unknown }>) {
    setError(null);
    setNotice(null);
    const { error: e } = await call();
    if (e) setError(errorMessage(e));
    else setNotice(done);
    await load();
  }

  return (
    <section aria-labelledby="names-heading">
      <h2 id="names-heading">Names to check</h2>
      <p className="muted">
        The same customer, supplier or person is sometimes typed in more than one way. Where a name looks like another,
        it is listed here with a recommendation: make it an alias (it then counts as that name everywhere) or keep it as
        its own name. Nothing changes until you choose.
      </p>
      {error && <p role="alert" className="error">{error}</p>}
      {notice && <p role="status" className="muted">{notice}</p>}
      {items === null ? (
        <p className="muted">Loading…</p>
      ) : items.length === 0 ? (
        <p className="muted">Nothing needs checking.</p>
      ) : (
        <table>
          <thead>
            <tr><th>Name</th><th>Looks like</th><th></th></tr>
          </thead>
          <tbody>
            {items.map((i) => {
              const chosen = pick[keyOf(i)] ?? i.suggested ?? i.candidates[0]?.name ?? "";
              return (
                <tr key={keyOf(i)}>
                  <td>
                    {i.name} <span className="muted">({i.specs} spec{i.specs === 1 ? "" : "s"}, {KIND[i.kind].toLowerCase()})</span>
                  </td>
                  <td>
                    {i.recommendation === "alias" ? (
                      <span className="badge good">Probably an alias{i.suggested ? ` of ${i.suggested}` : ""}</span>
                    ) : (
                      <span className="badge warn">Probably a new name</span>
                    )}
                    <div className="muted">{i.reason}</div>
                  </td>
                  <td>
                    <span className="row">
                      <label>
                        <span className="muted">Same as </span>
                        <select
                          aria-label={`Same as, for ${i.name}`}
                          value={chosen}
                          onChange={(e) => setPick({ ...pick, [keyOf(i)]: e.target.value })}
                        >
                          {i.candidates.map((c) => (
                            <option key={c.name} value={c.name}>{c.name} ({c.specs})</option>
                          ))}
                        </select>
                      </label>
                      <button
                        disabled={!chosen}
                        onClick={() =>
                          void decide(`${i.name} now counts as ${chosen}.`, () =>
                            api.POST("/admin/names/alias", { body: { kind: i.kind, name: i.name, canonical: chosen } }),
                          )
                        }
                      >
                        Make alias
                      </button>
                      <button
                        className="link"
                        onClick={() =>
                          void decide(`${i.name} is kept as its own name.`, () =>
                            api.POST("/admin/names/separate", { body: { kind: i.kind, name: i.name } }),
                          )
                        }
                      >
                        It's a separate name
                      </button>
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}
