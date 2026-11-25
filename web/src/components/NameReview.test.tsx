import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { NameReview } from "./NameReview";
import { json, mockApi } from "../test/mockApi";

const item = (over: object) => ({
  kind: "customer", name: "Sainsbury", specs: 12, recommendation: "alias", suggested: "Sainsbury's",
  confidence: "safe", reason: "same name, spelling or case",
  candidates: [{ name: "Sainsbury's", specs: 30 }], ...over,
});

afterEach(() => vi.restoreAllMocks());

describe("NameReview", () => {
  it("recommends an alias and makes it on confirmation", async () => {
    let items = [item({})];
    const api = mockApi({
      "GET /admin/names": () => json(items),
      "POST /admin/names/alias": () => {
        items = [];
        return new Response(null, { status: 204 });
      },
    });
    render(<NameReview />);
    expect(await screen.findByText("Probably an alias of Sainsbury's")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Make alias" }));
    expect(await screen.findByText("Sainsbury now counts as Sainsbury's.")).toBeInTheDocument();
    expect(api.calls.find((c) => c.key === "POST /admin/names/alias")?.body).toEqual({
      kind: "customer", name: "Sainsbury", canonical: "Sainsbury's",
    });
    expect(await screen.findByText("Nothing needs checking.")).toBeInTheDocument();
  });

  it("recommends a new name and keeps it separate, or lets you pick another match", async () => {
    const api = mockApi({
      "GET /admin/names": () =>
        json([
          item({
            name: "Smart Pack Systems", kind: "supplier", specs: 1, recommendation: "new_name", suggested: null,
            confidence: "unclear", reason: "used by few specs",
            candidates: [{ name: "Smart Packaging", specs: 20 }, { name: "Smart Labels", specs: 3 }],
          }),
        ]),
      "POST /admin/names/separate": () => new Response(null, { status: 204 }),
      "POST /admin/names/alias": () => new Response(null, { status: 204 }),
    });
    render(<NameReview />);
    expect(await screen.findByText("Probably a new name")).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("Same as, for Smart Pack Systems"), "Smart Labels");
    await userEvent.click(screen.getByRole("button", { name: "Make alias" }));
    expect(api.calls.find((c) => c.key === "POST /admin/names/alias")?.body).toMatchObject({ canonical: "Smart Labels" });
    await userEvent.click(screen.getByRole("button", { name: "It's a separate name" }));
    expect(api.calls.find((c) => c.key === "POST /admin/names/separate")?.body).toEqual({
      kind: "supplier", name: "Smart Pack Systems",
    });
  });
});
