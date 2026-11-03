import { expect, test } from "@playwright/test";

import { shelveFixtures } from "./helpers";

test("without credentials a question returns ranked passages that open at the right line", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Retrieval only", { exact: true })).toBeVisible();
  await shelveFixtures(page, "Retrieval shelf");

  await page.getByLabel("Ask a question about this collection").fill("How many times can a loan be renewed?");
  await page.keyboard.press("Enter");

  const passages = page.getByRole("list", { name: "Ranked passages" });
  await expect(passages).toBeVisible();
  const first = passages.getByRole("listitem").first();
  await expect(first).toContainText("renewed twice");
  await expect(first).toContainText(/BM25 #\d/);

  await first.getByRole("link", { name: "Loan Policy" }).click();
  const highlight = page.locator("mark[data-highlight]");
  await expect(highlight.first()).toBeVisible();
  await expect(page.getByRole("article")).toContainText("A loan can be renewed twice");
  expect((await highlight.allTextContents()).join(" ")).toContain("renewed twice");

  await page.getByRole("link", { name: "Back to the conversation" }).click();
  await expect(page.getByRole("heading", { name: "How many times can a loan be renewed?" })).toBeVisible();
});
