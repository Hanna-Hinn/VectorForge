import { test, expect } from "@playwright/test";
import { setupMockApi } from "./fixtures/mock-api";

test.beforeEach(async ({ page }) => {
  await setupMockApi(page);
});

test("displays system health", async ({ page }) => {
  await page.goto("/settings");
  await expect(page.getByText(/healthy/i).first()).toBeVisible();
});

test("displays provider information", async ({ page }) => {
  await page.goto("/settings");
  await expect(page.getByText("voyage")).toBeVisible();
  await expect(page.getByText("openai").first()).toBeVisible();
});
