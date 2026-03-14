import { test, expect } from "@playwright/test";
import { setupMockApi, COLLECTION_ID } from "./fixtures/mock-api";

test.beforeEach(async ({ page }) => {
  await setupMockApi(page);
});

test("displays collections list", async ({ page }) => {
  await page.goto("/collections");
  await expect(page.getByRole("heading", { name: "test-docs" })).toBeVisible();
  await expect(page.getByText("A test document collection")).toBeVisible();
});

test("navigates to documents page", async ({ page }) => {
  await page.goto("/collections");
  await page.getByText("Documents").first().click();
  await expect(page).toHaveURL(
    `/collections/${COLLECTION_ID}/documents`,
  );
});

test("navigates to query page", async ({ page }) => {
  await page.goto("/collections");
  await page.getByText("Query").first().click();
  await expect(page).toHaveURL(`/collections/${COLLECTION_ID}/query`);
});

test("navigates to analytics page", async ({ page }) => {
  await page.goto("/collections");
  await page.getByText("Analytics").first().click();
  await expect(page).toHaveURL(
    `/collections/${COLLECTION_ID}/analytics`,
  );
});
