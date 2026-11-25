import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Chat } from "./Chat";
import { event, json, mockApi, sse } from "../test/mockApi";
import { NETWORK_MESSAGE } from "../api/client";

const citation = {
  n: 1, doc_id: "doc1", filename: "sop.pdf", page: 1, section: "1. Release", clause: null,
  score: 8.7, snippet: "Product must reach 75C core before release.",
};
const answerEvents = [
  event({ event: "sources", citations: [citation] }),
  event({ event: "token", text: "Cooked duck must reach 75C " }),
  event({ event: "token", text: "core [1]." }),
  event({ event: "done", answer: "Cooked duck must reach 75C core [1].", answered: true, cited: [1], prompt_version: "answer_v1" }),
];

async function ask(question = "What core temperature?") {
  await userEvent.type(screen.getByLabelText("Your question"), question);
  await userEvent.click(screen.getByRole("button", { name: "Ask" }));
}

describe("Chat", () => {
  it("shows a clear message when the connection drops while asking", async () => {
    mockApi({ "POST /ask": () => { throw new TypeError("Failed to fetch"); } });
    render(<Chat />);
    await ask();
    expect(await screen.findByRole("alert")).toHaveTextContent(NETWORK_MESSAGE);
  });

  it("says when an answer came from the spec check", async () => {
    mockApi({
      "POST /ask": () =>
        sse([
          event({ event: "sources", citations: [citation] }) +
            event({ event: "token", text: "Batch B2613-01 against spec.pdf: FAIL [1]." }) +
            event({ event: "done", answer: "Batch B2613-01 against spec.pdf: FAIL [1].", answered: true, cited: [1], prompt_version: "answer_v2", via: "coa_check" }),
        ], { "x-request-id": "req0123456789" }),
    });
    render(<Chat />);
    await ask("Does batch B2613-01 meet spec?");
    expect(await screen.findByText(/From the spec check/)).toBeInTheDocument();
  });

  it("says when an answer came from the certificates table", async () => {
    mockApi({
      "POST /ask": () =>
        sse([
          event({ event: "sources", citations: [citation] }) +
            event({ event: "token", text: "TP-20931: next calibration due 2027-11-10 [1]." }) +
            event({ event: "done", answer: "TP-20931: next calibration due 2027-11-10 [1].", answered: true, cited: [1], prompt_version: "answer_v2", via: "certificates" }),
        ], { "x-request-id": "req0123456789" }),
    });
    render(<Chat />);
    await ask("When is TP-20931 due?");
    expect(await screen.findByText(/From the certificates table/)).toBeInTheDocument();
  });

  it("streams an answer, links the citation to the page text, and records a rating", async () => {
    const api = mockApi({
      "POST /ask": () => sse([answerEvents[0]!, answerEvents[1]!, answerEvents[2]! + answerEvents[3]!], { "x-request-id": "req0123456789" }),
      "GET /documents/doc1/pages/1": () =>
        json({ doc_id: "doc1", filename: "sop.pdf", doc_type: "sop", page: 1, method: "text", text: "Intro.\nProduct must reach 75C core before release.\nEnd." }),
      "POST /feedback": () => json({ stored: true, langfuse: false }),
    });
    render(<Chat />);
    await ask();

    expect(await screen.findByText(/Cooked duck must reach 75C core/)).toBeInTheDocument();
    expect(screen.getByText("1 sources")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Source 1: sop.pdf page 1" }));
    const dialog = await screen.findByRole("dialog", { name: "Source page" });
    expect(await within(dialog).findByText("sop.pdf · page 1")).toBeInTheDocument();
    expect(dialog.querySelector("mark")).toHaveTextContent("Product must reach 75C core before release.");
    await userEvent.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: "Good answer" }));
    await waitFor(() => expect(api.count("POST /feedback")).toBe(1));
    expect(api.calls.find((c) => c.key === "POST /feedback")?.body).toEqual({ request_id: "req0123456789", kind: "ask", rating: "up" });
    expect(screen.getByRole("button", { name: "Good answer" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Bad answer" })).toBeDisabled(); // one rating per answer
  });

  it("says so when the documents do not cover the question", async () => {
    mockApi({ "POST /ask": () => sse([event({ event: "no_answer", reason: "no_relevant_documents" })], { "x-request-id": "req0123456789" }) });
    render(<Chat />);
    await ask("Where is the fire assembly point?");
    expect(await screen.findByText(/Nothing relevant was found/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Good answer" })).not.toBeInTheDocument();
  });

  it("shows a server error instead of hanging", async () => {
    mockApi({ "POST /ask": () => json({ detail: "The request failed. Please try again." }, 500) });
    render(<Chat />);
    await ask();
    expect(await screen.findByRole("alert")).toHaveTextContent("The request failed");
    expect(screen.getByLabelText("Your question")).toBeEnabled();
  });

  it("does not send an empty question", async () => {
    const api = mockApi({});
    render(<Chat />);
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
    await userEvent.type(screen.getByLabelText("Your question"), "   ");
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
    expect(api.fetchMock).not.toHaveBeenCalled();
  });

  it("offers every row of a list answer as a download and sends the query back", async () => {
    const query = { intent: "list", filter: { customer: "Tesco" } };
    const api = mockApi({
      "POST /ask": () =>
        sse([
          event({ event: "token", text: "Understood as: list of specs. 104 of the 550 match: 1. ... and 79 more." }) +
            event({
              event: "done", answer: "104 of the 550 match", answered: true, cited: [], prompt_version: "answer_v2",
              via: "fact_spec", export: { query, count: 104 },
            }),
        ], { "x-request-id": "req0123456789" }),
      "POST /export/specs": () =>
        new Response("\ufeffNAV code,Product\r\n4053420,Turkey\r\n", {
          headers: { "content-type": "text/csv", "content-disposition": 'attachment; filename="specs-20261121.csv"' },
        }),
    });
    const created: Blob[] = [];
    URL.createObjectURL = (b: Blob | MediaSource) => (created.push(b as Blob), "blob:test");
    URL.revokeObjectURL = () => {};
    render(<Chat />);
    await ask("which specs use both codes");
    const button = await screen.findByRole("button", { name: "Download all 104 specs (CSV)" });
    await userEvent.click(button);
    await waitFor(() => expect(api.count("POST /export/specs")).toBe(1));
    expect(api.calls.find((c) => c.key === "POST /export/specs")?.body).toEqual({ query });
    await waitFor(() => expect(created).toHaveLength(1));
    expect(await created[0]?.text()).toContain("4053420,Turkey");
  });

  it("shows why a download was refused", async () => {
    mockApi({
      "POST /ask": () =>
        sse([
          event({ event: "token", text: "ok" }) +
            event({ event: "done", answer: "ok", answered: true, cited: [], prompt_version: "answer_v2", via: "fact_spec", export: { query: {}, count: 3 } }),
        ], { "x-request-id": "req0123456789" }),
      "POST /export/specs": () => json({ detail: "No customer called \"Nobody\"." }, 422),
    });
    render(<Chat />);
    await ask("anything");
    await userEvent.click(await screen.findByRole("button", { name: "Download all 3 specs (CSV)" }));
    expect(await screen.findByRole("alert")).toHaveTextContent('No customer called "Nobody"');
  });

  it("has no download button on an answer that is not a list of specs", async () => {
    mockApi({ "POST /ask": () => sse(answerEvents, { "x-request-id": "req0123456789" }) });
    render(<Chat />);
    await ask();
    await screen.findByRole("button", { name: "Source 1: sop.pdf page 1" });
    expect(screen.queryByRole("button", { name: /Download all/ })).not.toBeInTheDocument();
  });

  it("sends the last list's query with a follow-up, and a new topic starts clean", async () => {
    const query = { intent: "list", filter: { product: "whole duck" } };
    const reply = (text: string, withExport: boolean) =>
      sse([
        event({ event: "token", text }) +
          event({ event: "done", answer: text, answered: true, cited: [], prompt_version: "answer_v2", via: "fact_spec", export: withExport ? { query, count: 5 } : null }),
      ]);
    let n = 0;
    const api = mockApi({ "POST /ask": () => reply(`answer ${++n}`, n === 1) });
    render(<Chat />);
    await ask("list whole duck specs");
    await screen.findByText("answer 1");
    await ask("give me all from tesco");
    await screen.findByText("answer 2");
    const bodies = api.calls.filter((c) => c.key === "POST /ask").map((c) => c.body);
    expect(bodies[0]).toEqual({ question: "list whole duck specs" });
    expect(bodies[1]).toEqual({ question: "give me all from tesco", context: query });
    await ask("one more");
    await screen.findByText("answer 3");
    expect(api.calls.filter((c) => c.key === "POST /ask")[2]?.body).toEqual({ question: "one more" });
    await userEvent.click(screen.getByRole("button", { name: "New topic" }));
    expect(screen.queryByText("answer 1")).not.toBeInTheDocument();
  });
});
