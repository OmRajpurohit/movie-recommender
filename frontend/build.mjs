import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const frontendDir = dirname(fileURLToPath(import.meta.url));
const configPath = join(frontendDir, "config.js");
const configuredApiBaseUrl = (process.env.CINEMA_ATLAS_API_BASE_URL || "").trim();

const apiBaseUrl = configuredApiBaseUrl || "https://your-render-backend.onrender.com";
const fileContents = `window.CINEMA_ATLAS_CONFIG = {
    API_BASE_URL: "${apiBaseUrl.replace(/"/g, '\\"')}"
};
`;

mkdirSync(resolve(frontendDir), { recursive: true });
writeFileSync(configPath, fileContents, "utf-8");

const indexHtml = readFileSync(join(frontendDir, "index.html"), "utf-8");
if (!indexHtml.includes("./config.js")) {
    throw new Error("frontend/index.html no longer references ./config.js");
}

console.log(`Prepared frontend config for API base: ${apiBaseUrl}`);
