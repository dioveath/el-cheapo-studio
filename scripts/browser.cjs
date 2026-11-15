const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const root = path.resolve(__dirname, "..");
const studioUrl = process.env.STUDIO_URL || "http://localhost:3000";
const artifacts = process.env.ARTIFACT_DIR || path.join(root, "artifacts");

(async () => {
  await fs.mkdir(artifacts, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--enable-unsafe-swiftshader"],
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
    deviceScaleFactor: 1,
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  const library = await (
    await page.request.get(`${studioUrl}/api/jobs?limit=200`)
  ).json();
  const source = library.jobs.find(
    (job) =>
      job.status === "completed" &&
      job.operation === "animation" &&
      (!process.env.STUDIO_SOURCE_JOB_ID || job.id === process.env.STUDIO_SOURCE_JOB_ID),
  );
  assert(source, "Complete an animation first.");
  await page.addInitScript((id) => {
    localStorage.setItem("studio.selection", id);
    localStorage.setItem("theme", "light");
  }, source.id);
  await page.goto(studioUrl, { waitUntil: "networkidle" });
  await page
    .getByRole("button", { name: "Play animation", exact: true })
    .waitFor({ timeout: 60000 });
  await page.waitForTimeout(700);
  const stage = await page.locator("canvas").boundingBox();
  const preview = await page.locator(".preview-surface").boundingBox();
  assert(stage.width > 500 && stage.height > 250);
  assert(stage.x >= preview.x && stage.x + stage.width <= preview.x + preview.width);
  await page.screenshot({ path: `${artifacts}/studio-light.png` });
  const rest = await page.locator("canvas").screenshot();
  await page
    .getByRole("button", { name: "Play animation", exact: true })
    .click();
  await page.waitForTimeout(700);
  await page
    .getByRole("button", { name: "Pause animation", exact: true })
    .click();
  const paused = await page.locator("canvas").screenshot();
  assert(!rest.equals(paused), "Playback must deform the character.");
  const time = await page.getByLabel("Animation time").textContent();
  await page.waitForTimeout(300);
  assert.equal(await page.getByLabel("Animation time").textContent(), time);
  assert(
    paused.equals(await page.locator("canvas").screenshot()),
    "Paused scene must stay idle.",
  );
  await page.getByLabel("Playback speed").selectOption("0.5");
  await page.getByLabel("Animation position").fill("1.5");
  await page.waitForTimeout(100);
  assert.equal(await page.getByLabel("Animation time").textContent(), "1.50 s");
  await page.getByRole("button", { name: "Toggle skeleton" }).click();
  await page.screenshot({ path: `${artifacts}/studio-skeleton.png` });
  await page.getByRole("button", { name: "Toggle skeleton" }).click();
  await page.getByRole("button", { name: "Toggle theme" }).click();
  await page.screenshot({ path: `${artifacts}/studio-dark.png` });
  await page.getByRole("button", { name: "Bind pose", exact: true }).click();
  await page.waitForTimeout(400);
  assert.equal(await page.locator(".motion-dock").count(), 0);
  await page.getByRole("button", { name: "Motion", exact: true }).click();
  await page
    .getByRole("button", { name: "Play animation", exact: true })
    .waitFor();
  await page.locator(".exports summary").click();
  const downloadEvent = page.waitForEvent("download");
  await page
    .getByRole("link", { name: "ANIMATION · FBX", exact: true })
    .click();
  const download = await downloadEvent;
  assert(
    (await fs.readFile(await download.path()))
      .subarray(0, 18)
      .toString()
      .startsWith("Kaydara FBX Binary"),
  );
  await page.getByRole("button", { name: "Expand preview" }).click();
  await page.waitForFunction(() => !!document.fullscreenElement);
  await page.getByRole("button", { name: "Expand preview" }).click();
  await page.getByRole("button", { name: "Toggle library" }).click();
  await page.getByLabel("Search assets").fill(source.filename);
  assert.equal(await page.locator(".asset-card").count(), 1);
  await page.getByRole("button", { name: "Close library or queue" }).click();
  await page.getByRole("button", { name: "Create", exact: true }).click();
  const response = page.waitForResponse(
    (response) =>
      response.url().endsWith(`/api/jobs/${source.id}/variants`) &&
      response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "500", exact: true }).click();
  await page
    .getByRole("button", { name: "Save mesh version", exact: true })
    .click();
  const version = (await (await response).json()).jobs[0];
  assert.equal(version.parent_id, source.id);
  await page
    .getByRole("button", { name: "Animate this character" })
    .waitFor({ timeout: 120000 });
  await page.getByRole("button", { name: "Animate this character" }).click();
  await page.getByRole("button", { name: "Wave", exact: true }).click();
  const motionResponse = page.waitForResponse(
    (response) =>
      response.url().endsWith(`/api/jobs/${version.id}/animations`) &&
      response.request().method() === "POST",
  );
  await page
    .getByRole("button", { name: "Generate animation", exact: true })
    .click();
  const animation = (await (await motionResponse).json()).jobs[0];
  assert.equal(animation.operation, "animation");
  await page
    .getByRole("button", { name: "Play animation", exact: true })
    .waitFor({ timeout: 180000 });
  await page.setViewportSize({ width: 390, height: 844 });
  await page
    .getByRole("button", { name: "Toggle inspector", exact: true })
    .click();
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${artifacts}/studio-mobile.png` });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
    false,
  );
  const dock = await page.locator(".motion-dock").boundingBox();
  assert(dock.x >= 0 && dock.x + dock.width <= 390);
  await page.getByRole("button", { name: "Toggle inspector" }).click();
  await page.screenshot({ path: `${artifacts}/studio-mobile-inspector.png` });
  assert.deepEqual(errors, []);
  await fs.writeFile(
    `${artifacts}/studio-browser.json`,
    JSON.stringify(
      {
        source: source.id,
        variant: version.id,
        animation: animation.id,
        status: "passed",
      },
      null,
      2,
    ),
  );
  console.log(
    "PASS: full-screen stage, motion generation, variant, playback/pause/scrub/speed, skeleton, bind comparison, FBX download, fullscreen, library, themes, mobile; no browser errors.",
  );
  await browser.close();
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
