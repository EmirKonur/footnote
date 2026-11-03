import { defineConfig, devices } from "@playwright/test";

/**
 * Two complete stacks, each with a fresh data directory:
 *   retrieval-only  API :4412 (FOOTNOTE_RETRIEVAL_ONLY=true)   web :4414
 *   fixtures        API :4413 (FOOTNOTE_FAKE_LLM=fixtures)     web :4415
 * Neither can call the Anthropic API. Both use the real local embedding model, cached
 * in api/data/models (or FOOTNOTE_E2E_MODELS_DIR in CI) so it downloads once.
 */
const modelsDir = process.env.FOOTNOTE_E2E_MODELS_DIR ?? "data/models";
const stacks = [
  { name: "retrieval-only", api: 4412, web: 4414, env: "FOOTNOTE_RETRIEVAL_ONLY=true" },
  { name: "fixtures", api: 4413, web: 4415, env: "FOOTNOTE_FAKE_LLM=fixtures" },
];

export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: { trace: "retain-on-failure", ...devices["Desktop Chrome"] },
  projects: stacks.map((stack) => ({
    name: stack.name,
    testMatch: `${stack.name}.spec.ts`,
    use: { baseURL: `http://localhost:${stack.web}` },
  })),
  webServer: stacks.flatMap((stack) => [
    {
      command:
        `rm -rf e2e/.data/${stack.name} && cd ../api && ` +
        `FOOTNOTE_DATA_DIR=../web/e2e/.data/${stack.name} FOOTNOTE_MODELS_DIR=${modelsDir} ` +
        `FOOTNOTE_PORT=${stack.api} FOOTNOTE_CORS_ORIGINS='["http://localhost:${stack.web}"]' ` +
        `${stack.env} uv run footnote serve`,
      url: `http://127.0.0.1:${stack.api}/api/health`,
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: `FOOTNOTE_API_URL=http://localhost:${stack.api} pnpm exec next start --port ${stack.web}`,
      url: `http://localhost:${stack.web}`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ]),
});
