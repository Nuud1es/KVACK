import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Login } from "./Login";
import { json, mockApi } from "../test/mockApi";

describe("Login", () => {
  it("rejects an unknown key and does not keep it", async () => {
    mockApi({ "GET /me": () => json({ detail: "Invalid or missing API key" }, 401) });
    const onSignedIn = vi.fn();
    render(<Login onSignedIn={onSignedIn} />);
    await userEvent.click(screen.getByRole("button", { name: "Use an API key" }));
    await userEvent.type(screen.getByLabelText("API key"), "wrong");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("not recognised");
    expect(onSignedIn).not.toHaveBeenCalled();
    expect(sessionStorage.getItem("qacopilot.apiKey")).toBeNull();
  });

  it("signs in with a good key and sends it as x-api-key", async () => {
    const api = mockApi({ "GET /me": () => json({ user: "alice" }) });
    const onSignedIn = vi.fn();
    render(<Login onSignedIn={onSignedIn} />);
    await userEvent.click(screen.getByRole("button", { name: "Use an API key" }));
    await userEvent.type(screen.getByLabelText("API key"), "key-alice");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(onSignedIn).toHaveBeenCalledWith("alice"));
    expect(api.calls[0]?.headers.get("x-api-key")).toBe("key-alice");
    expect(sessionStorage.getItem("qacopilot.apiKey")).toBe("key-alice");
  });

  it("signs in with a name and PIN and keeps the token, not the PIN", async () => {
    const api = mockApi({
      "POST /auth/login": () => json({ token: "tok-1", user: "Alice Smith", expires_in_hours: 12 }),
    });
    const onSignedIn = vi.fn();
    render(<Login onSignedIn={onSignedIn} />);
    await userEvent.type(screen.getByLabelText("Name"), "Alice Smith");
    await userEvent.type(screen.getByLabelText("PIN (4 digits)"), "1234");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(onSignedIn).toHaveBeenCalledWith("Alice Smith"));
    expect(sessionStorage.getItem("qacopilot.apiKey")).toBe("tok-1");
    expect(api.calls[0]?.body).toEqual({ name: "Alice Smith", pin: "1234" });
  });

  it("only accepts four digits and does not enable the button before that", async () => {
    mockApi({});
    render(<Login onSignedIn={vi.fn()} />);
    await userEvent.type(screen.getByLabelText("Name"), "Alice");
    await userEvent.type(screen.getByLabelText("PIN (4 digits)"), "12a3");
    expect(screen.getByLabelText("PIN (4 digits)")).toHaveValue("123");
    expect(screen.getByRole("button", { name: "Sign in" })).toBeDisabled();
  });

  it("registers a new person and shows the server's reason when the name is taken", async () => {
    mockApi({ "POST /auth/register": () => json({ detail: "That name is already taken." }, 409) });
    const onSignedIn = vi.fn();
    render(<Login onSignedIn={onSignedIn} />);
    await userEvent.click(screen.getByRole("button", { name: "Create an account" }));
    await userEvent.type(screen.getByLabelText("Name"), "Alice");
    await userEvent.type(screen.getByLabelText("PIN (4 digits)"), "1234");
    await userEvent.click(screen.getByRole("button", { name: "Create account" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("already taken");
    expect(onSignedIn).not.toHaveBeenCalled();
    expect(screen.getByLabelText("PIN (4 digits)")).toHaveValue("");
  });
});
