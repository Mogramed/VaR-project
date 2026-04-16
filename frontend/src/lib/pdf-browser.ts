import { existsSync } from "node:fs";
import { chromium } from "playwright-core";

function candidateExecutables() {
  const fromEnv = process.env.CHROMIUM_PATH;
  return [
    fromEnv,
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files\\Chromium\\Application\\chrome.exe",
  ].filter((value): value is string => Boolean(value));
}

export function resolveChromiumExecutable() {
  return candidateExecutables().find((path) => existsSync(path)) ?? null;
}

export async function launchPdfBrowser() {
  const executablePath = resolveChromiumExecutable();
  if (!executablePath) {
    throw new Error("Chromium executable not found for PDF generation.");
  }

  return chromium.launch({
    headless: true,
    executablePath,
    args: ["--no-sandbox", "--disable-dev-shm-usage", "--font-render-hinting=none"],
  });
}
