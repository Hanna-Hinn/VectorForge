import { test, expect } from "@playwright/test";
import { setupMockApi, COLLECTION_ID } from "./fixtures/mock-api";

test.beforeEach(async ({ page }) => {
  await setupMockApi(page);
});

test("displays documents list", async ({ page }) => {
  await page.goto(`/collections/${COLLECTION_ID}/documents`);
  await expect(page.getByText("guide.md")).toBeVisible();
  await expect(page.getByText("readme.txt")).toBeVisible();
});

test("shows document status badges", async ({ page }) => {
  await page.goto(`/collections/${COLLECTION_ID}/documents`);
  // "indexed" badges should appear for both documents
  const badges = page.getByText("indexed");
  await expect(badges.first()).toBeVisible();
});
