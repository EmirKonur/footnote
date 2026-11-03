import { expect, type Page } from "@playwright/test";
import { fileURLToPath } from "node:url";

const fixtures = fileURLToPath(new URL("../../api/tests/fixtures/", import.meta.url));

/** Create a collection from the library page and upload the fixture documents. */
export async function shelveFixtures(page: Page, name: string): Promise<void> {
  await page.goto("/");
  await page.getByLabel("New collection").fill(name);
  await page.getByRole("button", { name: "Create" }).click();
  await expect(page.getByRole("heading", { name })).toBeVisible();

  await page.getByLabel("Choose files to upload").setInputFiles([`${fixtures}loans.txt`, `${fixtures}field-guide.md`]);
  const documents = page.getByRole("navigation", { name: "Collection" });
  // The first embed in a fresh process loads the ONNX model, so this can take a few seconds.
  await expect(documents.getByRole("link", { name: /Loan Policy/ })).toBeVisible({ timeout: 60_000 });
  await expect(documents.getByRole("link", { name: /Field Guide to the Reading Room/ })).toBeVisible();
}
