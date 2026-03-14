import { describe, it, expect, vi, beforeEach } from "vitest";
import { get, post, del, setApiKey, ApiError } from "@/api/client";

beforeEach(() => {
  vi.restoreAllMocks();
  setApiKey(null);
});

function mockFetch(body: unknown, status = 200): void {
  vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
}

describe("get", () => {
  it("sends a GET request and parses JSON", async () => {
    mockFetch({ items: [] });
    const result = await get<{ items: unknown[] }>("/test");
    expect(result).toEqual({ items: [] });
    expect(fetch).toHaveBeenCalledWith(
      "/api/test",
      expect.objectContaining({
        headers: expect.objectContaining({
          "Content-Type": "application/json",
        }),
      }),
    );
  });

  it("throws ApiError on non-OK response", async () => {
    mockFetch({ detail: "Not found" }, 404);
    try {
      await get("/missing");
      expect.fail("should have thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError);
      expect((e as ApiError).status).toBe(404);
    }
  });
});

describe("post", () => {
  it("sends a POST request with body", async () => {
    mockFetch({ id: "123" }, 201);
    const result = await post("/items", { name: "test" });
    expect(result).toEqual({ id: "123" });
    expect(fetch).toHaveBeenCalledWith(
      "/api/items",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ name: "test" }),
      }),
    );
  });
});

describe("del", () => {
  it("sends a DELETE request", async () => {
    mockFetch({ message: "Deleted" });
    const result = await del("/items/1");
    expect(result).toEqual({ message: "Deleted" });
    expect(fetch).toHaveBeenCalledWith(
      "/api/items/1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });
});

describe("API key", () => {
  it("includes X-Api-Key header when set", async () => {
    setApiKey("my-secret");
    mockFetch({ ok: true });
    await get("/protected");
    expect(fetch).toHaveBeenCalledWith(
      "/api/protected",
      expect.objectContaining({
        headers: expect.objectContaining({ "X-Api-Key": "my-secret" }),
      }),
    );
  });

  it("does not include header when key is null", async () => {
    mockFetch({ ok: true });
    await get("/public");
    const callHeaders = (fetch as ReturnType<typeof vi.fn>).mock.calls[0]?.[1]
      ?.headers as Record<string, string>;
    expect(callHeaders["X-Api-Key"]).toBeUndefined();
  });
});
