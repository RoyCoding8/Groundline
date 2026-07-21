import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  use: { baseURL: "http://127.0.0.1:4173", trace: "retain-on-failure" },
  webServer: {
    command: "uv --directory .. run python scripts/serve_e2e.py",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: false,
  },
});
