import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Admin } from "./Admin";
import { json, mockApi } from "../test/mockApi";

const account = (over: object) => ({
  id: 1, name: "Joe Rudling", created_at: "2026-11-20T10:00:00Z", failed_attempts: 0, locked_until: null,
  locked: false, sessions: 0, last_sign_in: null, admin: false, ...over,
});
const people = [
  account({ id: 1, name: "Boss Person", admin: true, sessions: 1 }),
  account({ id: 2, name: "Sam Jones", locked: true, locked_until: "2026-11-20T12:05:00Z", failed_attempts: 0 }),
  account({ id: 3, name: "Pat Lee", sessions: 2 }),
];

afterEach(() => vi.restoreAllMocks());

describe("Admin", () => {
  it("lists accounts, filters them and shows who is locked", async () => {
    mockApi({ "GET /admin/users": () => json(people), "GET /admin/names": () => json([]) });
    render(<Admin me="Boss Person" />);
    expect(await screen.findByText("Sam Jones")).toBeInTheDocument();
    expect(screen.getByText(/locked until/)).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText("Find a person"), "pat");
    expect(screen.queryByText("Sam Jones")).not.toBeInTheDocument();
    expect(screen.getByText("Pat Lee")).toBeInTheDocument();
  });

  it("sets a new PIN, four digits only", async () => {
    const api = mockApi({
      "GET /admin/users": () => json(people),
      "GET /admin/names": () => json([]),
      "POST /admin/users/3/pin": () => new Response(null, { status: 204 }),
    });
    render(<Admin me="Boss Person" />);
    const row = (await screen.findByText("Pat Lee")).closest("tr") as HTMLElement;
    await userEvent.click(within(row).getByRole("button", { name: "New PIN" }));
    const box = within(row).getByLabelText("New PIN for Pat Lee");
    await userEvent.type(box, "12a3");
    expect(box).toHaveValue("123");
    expect(within(row).getByRole("button", { name: "Set PIN" })).toBeDisabled();
    await userEvent.type(box, "4");
    await userEvent.click(within(row).getByRole("button", { name: "Set PIN" }));
    expect(await screen.findByText("New PIN set for Pat Lee.")).toBeInTheDocument();
    expect(api.calls.find((c) => c.key === "POST /admin/users/3/pin")?.body).toEqual({ pin: "1234" });
  });

  it("unlocks only a locked account and cannot delete yourself", async () => {
    const api = mockApi({
      "GET /admin/users": () => json(people),
      "GET /admin/names": () => json([]),
      "POST /admin/users/2/unlock": () => new Response(null, { status: 204 }),
    });
    render(<Admin me="Boss Person" />);
    const sam = (await screen.findByText("Sam Jones")).closest("tr") as HTMLElement;
    const boss = screen.getByText("Boss Person").closest("tr") as HTMLElement;
    expect(within(boss).queryByRole("button", { name: "Delete" })).not.toBeInTheDocument();
    expect(within(boss).queryByRole("button", { name: "Unlock" })).not.toBeInTheDocument();
    await userEvent.click(within(sam).getByRole("button", { name: "Unlock" }));
    await waitFor(() => expect(api.count("POST /admin/users/2/unlock")).toBe(1));
  });

  it("deletes an account only after confirming", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
    const api = mockApi({
      "GET /admin/users": () => json(people),
      "GET /admin/names": () => json([]),
      "DELETE /admin/users/3": () => new Response(null, { status: 204 }),
    });
    render(<Admin me="Boss Person" />);
    const row = (await screen.findByText("Pat Lee")).closest("tr") as HTMLElement;
    await userEvent.click(within(row).getByRole("button", { name: "Delete" }));
    expect(api.count("DELETE /admin/users/3")).toBe(0); // said no
    await userEvent.click(within(row).getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(api.count("DELETE /admin/users/3")).toBe(1));
    expect(confirm).toHaveBeenCalledTimes(2);
  });

  it("shows the server's reason when it refuses", async () => {
    mockApi({
      "GET /admin/users": () => json({ detail: "Only an administrator can do that." }, 403),
      "GET /admin/names": () => json([]),
    });
    render(<Admin me="Joe" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Only an administrator");
  });
});
