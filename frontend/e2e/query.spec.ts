import { test, expect } from "@playwright/test";
import { setupMockApi, COLLECTION_ID } from "./fixtures/mock-api";

test.beforeEach(async ({ page }) => {
  await setupMockApi(page);
});

test("displays query page with input", async ({ page }) => {
  await page.goto(`/collections/${COLLECTION_ID}/query`);
  await expect(
    page.getByPlaceholder(/type your question/i),
  ).toBeVisible();
});

test("sends a query and receives streamed response", async ({ page }) => {
  await page.goto(`/collections/${COLLECTION_ID}/query`);

  const input = page.getByPlaceholder(/type your question/i);
  await input.fill("What is vector search?");

  // Find and click the send button
  await page.getByRole("button", { name: /send/i }).click();

  // The streamed answer should appear
  await expect(page.getByText("Vector search is great.")).toBeVisible({
    timeout: 5000,
  });
});
