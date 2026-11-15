const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const root = path.resolve(__dirname, "..");
const url = process.env.STUDIO_URL || "http://localhost:3000";
const artifacts = process.env.ARTIFACT_DIR || path.join(root, "artifacts");

(async () => {
  await fs.mkdir(artifacts, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--enable-unsafe-swiftshader"],
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.addInitScript(() => {
      localStorage.setItem("studio.selection", "");
      localStorage.setItem("theme", "light");
    });
    await page.goto(url, { waitUntil: "networkidle" });
    await page
      .getByRole("navigation", { name: "Workspace mode" })
      .getByRole("button", { name: "Video", exact: true })
      .click();
    await page
      .getByRole("textbox", { name: "Video prompt", exact: true })
      .fill(
        "A red toy car drives slowly across a gray tabletop. Fixed side view, one continuous shot.",
      );
    await page
      .locator('input[accept=".png,.jpg,.jpeg,.webp"]')
      .setInputFiles(path.join(root, "examples/chair.png"));
    await page.getByAltText("Starting frame").waitFor();
    await page.getByRole("button", { name: "Remove starting image" }).click();
    const createdResponse = page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/videos") &&
        response.request().method() === "POST",
    );
    await page
      .getByRole("button", { name: "Generate video", exact: true })
      .click();
    const created = (await (await createdResponse).json()).jobs[0];
    await page.getByRole("button", { name: "Cancel job", exact: true }).click();
    await page
      .getByRole("button", { name: "Retry", exact: true })
      .waitFor({ timeout: 120000 });
    const cancelled = await (
      await page.request.get(`${url}/api/jobs/${created.id}`)
    ).json();
    assert.equal(cancelled.status, "cancelled");
    const retryResponse = page.waitForResponse((response) =>
      response.url().endsWith(`/api/jobs/${created.id}/retry`),
    );
    await page.getByRole("button", { name: "Retry", exact: true }).click();
    const retried = (await (await retryResponse).json()).jobs[0];
    assert.equal(retried.operation, "video");
    await page.getByLabel("Generated video").waitFor({ timeout: 180000 });
    await page.waitForFunction(
      () => document.querySelector("video")?.readyState >= 1,
    );
    const metadata = await page.locator("video").evaluate(async (video) => {
      await video.play();
      return {
        width: video.videoWidth,
        height: video.videoHeight,
        duration: video.duration,
        paused: video.paused,
      };
    });
    assert.deepEqual(metadata, {
      width: 320,
      height: 192,
      duration: 5.0625,
      paused: false,
    });
    await page.locator("video").evaluate((video) => {
      video.pause();
      video.currentTime = 2;
    });
    await page.waitForFunction(
      () => Math.abs(document.querySelector("video").currentTime - 2) < 0.05,
    );
    assert.equal(await page.locator("canvas").count(), 0);
    await page.locator(".exports summary").click();
    const downloadEvent = page.waitForEvent("download");
    await page.getByRole("link", { name: "MP4", exact: true }).click();
    const download = await downloadEvent;
    assert.equal(
      (await fs.readFile(await download.path())).subarray(4, 8).toString(),
      "ftyp",
    );
    await page.screenshot({ path: `${artifacts}/video-light.png` });
    await page.getByRole("button", { name: "Toggle theme" }).click();
    await page.screenshot({ path: `${artifacts}/video-dark.png` });
    await page.setViewportSize({ width: 390, height: 844 });
    for (const button of await page.locator(".app-header button").all()) {
      const box = await button.boundingBox();
      assert(
        box.x >= 0 && box.x + box.width <= 390,
        "Header control exceeds mobile viewport",
      );
    }
    await page.getByRole("button", { name: "Toggle inspector" }).click();
    await page.screenshot({ path: `${artifacts}/video-mobile.png` });
    await page.getByRole("button", { name: "New video", exact: true }).click();
    assert.equal(
      await page
        .getByRole("textbox", { name: "Video prompt", exact: true })
        .inputValue(),
      "",
    );
    assert.equal(await page.locator("video").count(), 0);
    assert.deepEqual(errors, []);
    const report = {
      cancelled: created.id,
      completed: retried.id,
      metadata,
      errors,
    };
    await fs.writeFile(
      `${artifacts}/video-browser.json`,
      JSON.stringify(report, null, 2),
    );
    console.log(JSON.stringify(report));
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
