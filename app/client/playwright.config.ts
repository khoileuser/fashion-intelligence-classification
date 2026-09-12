import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  use: { baseURL: "http://localhost:3100", headless: true },
  webServer: {
    command: "bun run dev --port 3100",
    url: "http://localhost:3100",
    reuseExistingServer: !process.env.CI,
  },
  workers: 1,
});
