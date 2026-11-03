import { expect, test } from "@playwright/test";

import { shelveFixtures } from "./helpers";

test("a streamed answer carries footnotes that preview and open the cited sentence", async ({ page }) => {
  await shelveFixtures(page, "Fixture shelf");

  await page.getByLabel("Ask a question about this collection").fill("How many books may members borrow at once?");
  await page.keyboard.press("Enter");

  await expect(page.getByText("Fixture answer · no model call")).toBeVisible();
  const reference = page.getByRole("link", { name: /^Footnote 1:/ });
  await reference.hover();
  await expect(page.getByRole("tooltip")).toContainText("Members may borrow up to six books at once.");
  await expect(page.getByRole("complementary", { name: "Footnotes" })).toContainText("Loan Policy");

  await reference.click();
  await expect(page.locator("mark[data-highlight]")).toHaveText("Members may borrow up to six books at once.");
  await expect(page.getByText("Footnote 1", { exact: true })).toBeVisible();
});

test("a question the documents can't answer is declined, not guessed", async ({ page }) => {
  await shelveFixtures(page, "Second shelf");
  await page.getByLabel("Ask a question about this collection").fill("What is the boiling point of mercury?");
  await page.keyboard.press("Enter");
  await expect(page.getByText("Not in these documents")).toBeVisible();
  await expect(page.locator(".answer__body")).toContainText("The documents don't cover this");
});
