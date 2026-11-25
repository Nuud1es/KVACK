import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { NETWORK_MESSAGE } from "../api/client";
import { Approvals } from "./Approvals";
import { json, mockApi } from "../test/mockApi";

const draft = {
  message: { to: ["qa@example.com"], cc: [], subject: "Calibration reminders: 2 due within 30 days", body: "Please arrange these.\n\nSerial | Due\nTP-1 | 2026-12-01" },
  certificate_count: 2, overdue: 0, due_soon: 2,
};
const summary = (status: string) => ({ run_id: "r1", created_by: "alice", query: "Draft reminders", status, route: "action", message: "", created_at: "2026-11-18T10:00:00Z" });
const runView = (status: string, extra: object = {}) => ({
  run_id: "r1", created_by: "alice", query: "Draft reminders", status, route: "action", message: "2 certificates found; 1 email drafted.",
  answer: null, answered: null, citations: [], certificates: [], drafts: [draft], deliveries: [], decided_by: null, ...extra,
});

let status: string;
beforeEach(() => {
  status = "awaiting_approval";
});

function routes(over: Record<string, () => Response> = {}) {
  return mockApi({
    "GET /agent/runs": () => json(status === "awaiting_approval" ? [summary(status)] : []),
    "GET /agent/runs/r1": () => json(runView(status, status === "completed" ? { decided_by: "bob" } : {})),
    "GET /agent/runs/r1/audit": () => json([{ ts: "2026-11-18T10:00:00Z", actor: "alice", event: "run_started", detail: {} }]),
    "POST /agent/runs/r1/approve": () => { status = "completed"; return json(runView("completed", { decided_by: "bob" })); },
    "POST /agent/runs/r1/reject": () => { status = "rejected"; return json(runView("rejected")); },
    ...over,
  });
}

async function open() {
  render(<Approvals me="bob" />);
  await userEvent.click(await screen.findByRole("button", { name: /Draft reminders/ }));
  await screen.findByText(/Calibration reminders: 2 due/);
}

describe("Approvals", () => {
  it("shows the drafted email and sends nothing until the second, explicit confirmation", async () => {
    const api = routes();
    await open();
    expect(screen.getByText(/qa@example.com/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Approve…" }));
    expect(api.count("POST /agent/runs/r1/approve")).toBe(0); // the first click only asks
    const group = screen.getByRole("group", { name: "Confirm approval" });
    expect(group).toHaveTextContent("Send 1 email to qa@example.com as bob?");

    await userEvent.click(within(group).getByRole("button", { name: "Cancel" }));
    expect(api.count("POST /agent/runs/r1/approve")).toBe(0);

    await userEvent.click(screen.getByRole("button", { name: "Approve…" }));
    await userEvent.click(screen.getByRole("button", { name: "Confirm and send" }));
    await waitFor(() => expect(api.count("POST /agent/runs/r1/approve")).toBe(1));
    expect(await screen.findByText(/decided by bob/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Approve…" })).not.toBeInTheDocument();
  });

  it("rejects with a reason and sends nothing", async () => {
    const api = routes();
    await open();
    await userEvent.click(screen.getByRole("button", { name: "Reject…" }));
    await userEvent.type(screen.getByLabelText("Reason"), "wrong list");
    await userEvent.click(screen.getByRole("button", { name: "Reject, send nothing" }));
    await waitFor(() => expect(api.count("POST /agent/runs/r1/reject")).toBe(1));
    expect(api.calls.find((c) => c.key === "POST /agent/runs/r1/reject")?.body).toEqual({ reason: "wrong list" });
    expect(api.count("POST /agent/runs/r1/approve")).toBe(0);
  });

  it("explains a conflict when someone else decided first", async () => {
    routes({ "POST /agent/runs/r1/approve": () => json({ detail: "This run is not waiting for approval" }, 409) });
    await open();
    await userEvent.click(screen.getByRole("button", { name: "Approve…" }));
    await userEvent.click(screen.getByRole("button", { name: "Confirm and send" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Someone else has already decided");
  });

  it("offers a retry when delivery failed", async () => {
    status = "delivery_failed";
    const api = routes({
      "GET /agent/runs": () => json([summary("delivery_failed")]),
      "POST /agent/runs/r1/retry-delivery": () => { status = "completed"; return json(runView("completed")); },
    });
    await open();
    expect(screen.queryByRole("button", { name: "Approve…" })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Retry failed emails" }));
    await waitFor(() => expect(api.count("POST /agent/runs/r1/retry-delivery")).toBe(1));
  });

  it("starts a new request and opens its run", async () => {
    status = "none";
    const api = routes({
      "POST /agent/runs": () => { status = "awaiting_approval"; return json(runView("awaiting_approval")); },
    });
    render(<Approvals me="bob" />);
    await userEvent.type(screen.getByLabelText("Request for the agent"), "Draft reminders");
    await userEvent.click(screen.getByRole("button", { name: "Start" }));
    expect(await screen.findByText(/Calibration reminders: 2 due/)).toBeInTheDocument();
    expect(api.calls.find((c) => c.key === "POST /agent/runs")?.body).toEqual({ query: "Draft reminders" });
  });

  it("recovers when the server drops the connection mid-request, instead of sticking on Working…", async () => {
    let down = true;
    mockApi({
      "GET /agent/runs": () => json([]),
      "POST /agent/runs": () => {
        if (down) throw new TypeError("Failed to fetch"); // what the browser throws when the API restarts
        return json({ run_id: "r1" });
      },
    });
    render(<Approvals me="alice" />);
    await userEvent.type(screen.getByLabelText("Request for the agent"), "Draft reminders");
    await userEvent.click(screen.getByRole("button", { name: "Start" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(NETWORK_MESSAGE);
    // the button is usable again and the box kept what was typed
    expect(screen.getByRole("button", { name: "Start" })).toBeEnabled();
    expect(screen.getByLabelText("Request for the agent")).toHaveValue("Draft reminders");
    down = false;
  });
});
