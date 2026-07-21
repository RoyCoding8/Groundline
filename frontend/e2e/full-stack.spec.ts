import { expect, test } from "@playwright/test";

test("runs an intervention through the real API and inspects its evidence", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Run Viewer" }).click();

  await expect(page.getByText("LEDGER VERIFIED")).toBeVisible();
  await expect(page.getByText("WORLD TRUTH / EXECUTIVE BELIEF")).toBeVisible();
  await expect(page.getByText("AGENT DECISIONS")).toBeVisible();
  await expect(page.locator(".decision-card").first()).toBeVisible();
  await page.getByLabel("Evidence department").selectOption("Engineering");
  await expect(page.locator(".evidence-list li").first()).toBeVisible();

  await page.getByRole("button", { name: "RUN INTERVENTION" }).click();

  await expect(page.getByText("COMPLETED")).toBeVisible({ timeout: 45_000 });
  await expect(page.getByText("12 PAIRED REPLICATES")).toBeVisible();
  await expect(page.getByText("MODEL OUTPUT NEVER SETS WORLD STATE")).toBeVisible();
});
