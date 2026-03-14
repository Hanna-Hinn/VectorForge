import { test, expect } from "@playwright/test";
import { setupMockApi, COLLECTION_ID } from "./fixtures/mock-api";

test.beforeEach(async ({ page }) => {
  await setupMockApi(page);
});

test("displays analytics summary cards", async ({ page }) => {
  await page.goto(`/collections/${COLLECTION_ID}/analytics`);
  await expect(page.getByText("42", { exact: true })).toBeVisible();
  await expect(page.getByText("15", { exact: true })).toBeVisible();
});

test("displays top queries", async ({ page }) => {
  await page.goto(`/collections/${COLLECTION_ID}/analytics`);
  await expect(
    page.getByText("What is vector search?"),
  ).toBeVisible();
  await expect(page.getByText("How does RAG work?")).toBeVisible();
});
