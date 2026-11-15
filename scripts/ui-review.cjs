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
    await page.goto(url, { waitUntil: "networkidle" });
    await page.evaluate(() => {
      localStorage.setItem("studio.selection", "");
      localStorage.setItem("theme", "light");
    });
    await page.reload({ waitUntil: "networkidle" });
    const mode = (name) =>
      page
        .getByRole("navigation", { name: "Workspace mode" })
        .getByRole("button", { name, exact: true });
    await page.screenshot({ path: `${artifacts}/ui-create-empty.png` });
    await mode("Video").click();
    await page.getByRole("button", { name: "Nature", exact: true }).click();
    const draft = await page
      .getByRole("textbox", { name: "Video prompt", exact: true })
      .inputValue();
    assert(draft.includes("flower"));
    await mode("Create").click();
    await mode("Video").click();
    assert.equal(
      await page
        .getByRole("textbox", { name: "Video prompt", exact: true })
        .inputValue(),
      draft,
    );
    await page.getByRole("button", { name: "New video", exact: true }).click();
    await page.screenshot({ path: `${artifacts}/ui-video-empty.png` });
    await page
      .locator('input[accept=".png,.jpg,.jpeg,.webp"]')
      .setInputFiles(path.join(root, "examples/chair.png"));
    await page.getByAltText("Starting frame").waitFor();
    await page.getByRole("button", { name: "Remove starting image" }).click();
    await page.getByText("Advanced settings", { exact: true }).click();
    const seed = await page.getByLabel("Seed", { exact: true }).inputValue();
    await page.getByRole("button", { name: "Randomize seed" }).click();
    assert.notEqual(
      await page.getByLabel("Seed", { exact: true }).inputValue(),
      seed,
    );
    await page.getByText("Advanced settings", { exact: true }).click();
    await page.locator(".recent-card").first().click();
    await page.waitForFunction(
      () => document.querySelector("video")?.readyState >= 1,
    );
    await page.locator("video").evaluate(async (video) => {
      await video.play();
      video.pause();
      video.currentTime = 1;
    });
    await page.screenshot({ path: `${artifacts}/ui-video-light.png` });
    const downloadEvent = page.waitForEvent("download");
    await page.getByRole("link", { name: "Download MP4", exact: true }).click();
    const download = await downloadEvent;
    assert.equal(
      (await fs.readFile(await download.path())).subarray(4, 8).toString(),
      "ftyp",
    );
    await page.getByRole("button", { name: "Toggle library" }).click();
    await page.getByRole("dialog").waitFor();
    assert(
      await page.evaluate(() =>
        document.querySelector("dialog").contains(document.activeElement),
      ),
    );
    await page.getByRole("button", { name: "Videos", exact: true }).click();
    await page.waitForFunction(() =>
      [...document.querySelectorAll(".library-filters button")].some(
        (button) =>
          button.textContent === "Videos" &&
          button.getAttribute("aria-pressed") === "true",
      ),
    );
    await page.getByLabel("Search assets").fill("red toy car");
    assert((await page.locator(".asset-card").count()) > 0);
    await page.screenshot({ path: `${artifacts}/ui-library.png` });
    await page.keyboard.press("Escape");
    assert.equal(await page.getByRole("dialog").count(), 0);
    assert.equal(
      await page.evaluate(() =>
        document.activeElement.getAttribute("aria-label"),
      ),
      "Toggle library",
    );
    await page.getByRole("button", { name: "Toggle theme" }).click();
    await page.screenshot({ path: `${artifacts}/ui-video-dark.png` });
    await page.setViewportSize({ width: 1366, height: 768 });
    const action = await page
      .getByRole("button", { name: "Generate video", exact: true })
      .boundingBox();
    const sidebar = await page.getByLabel("Workspace inspector").boundingBox();
    assert(
      action.y >= sidebar.y &&
        action.y + action.height <= sidebar.y + sidebar.height,
      "Generation must stay visible on a short screen",
    );
    const player = await page.locator("video").boundingBox();
    const frame = await page.locator(".preview-surface").boundingBox();
    assert(
      player.y >= frame.y && player.y + player.height <= frame.y + frame.height,
      "Video player is clipped",
    );
    assert(
      Math.abs(player.width / player.height - 5 / 3) < 0.01,
      "Video aspect ratio changed",
    );
    await page.screenshot({ path: `${artifacts}/ui-short-desktop.png` });
    await page.setViewportSize({ width: 390, height: 844 });
    const preview = await page.locator(".preview-surface").boundingBox();
    const inspector = await page
      .getByLabel("Workspace inspector")
      .boundingBox();
    assert(
      preview.y + preview.height < inspector.y,
      "Mobile settings must not cover the preview",
    );
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      "Mobile page overflows horizontally",
    );
    await page.screenshot({
      path: `${artifacts}/ui-mobile.png`,
      fullPage: true,
    });
    await page.setViewportSize({ width: 320, height: 740 });
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      "320 px layout overflows",
    );
    for (const button of await page.locator(".app-header button").all()) {
      const box = await button.boundingBox();
      assert(
        box.x >= 0 && box.x + box.width <= 320,
        "Header control exceeds 320 px viewport",
      );
    }
    await page.setViewportSize({ width: 1440, height: 1000 });
    const jobs = (
      await (await page.request.get(`${url}/api/jobs?limit=200`)).json()
    ).jobs;
    const animation = jobs.find(
      (job) => job.operation === "animation" && job.status === "completed",
    );
    assert(animation);
    await page.evaluate((id) => {
      localStorage.setItem("studio.selection", id);
      localStorage.setItem("theme", "light");
    }, animation.id);
    await page.reload({ waitUntil: "networkidle" });
    await page
      .getByRole("button", { name: "Play animation", exact: true })
      .waitFor({ timeout: 60000 });
    const canvas = await page.locator("canvas").boundingBox();
    const controls = await page.getByLabel("Workspace inspector").boundingBox();
    assert(
      canvas.width > 500 && canvas.x + canvas.width < controls.x,
      "Canvas must fit beside the settings",
    );
    await page
      .getByRole("button", { name: "Play animation", exact: true })
      .click();
    await page.waitForTimeout(400);
    await page
      .getByRole("button", { name: "Pause animation", exact: true })
      .click();
    await page.getByLabel("Animation position").fill("1.5");
    assert.equal(
      await page.getByLabel("Animation time").textContent(),
      "1.50 s",
    );
    await page.screenshot({ path: `${artifacts}/ui-animation.png` });
    await mode("Create").click();
    await page.getByRole("button", { name: "500", exact: true }).click();
    assert.equal(
      await page
        .getByRole("slider", { name: "Vertex budget", exact: true })
        .inputValue(),
      "500",
    );
    await page.screenshot({ path: `${artifacts}/ui-create.png` });
    assert.deepEqual(errors, []);
    const report = {
      draft_preservation: true,
      image_input: true,
      random_seed: true,
      download: true,
      library_keyboard: true,
      mobile_widths: [390, 320],
      animation_playback: true,
      errors,
    };
    await fs.writeFile(
      `${artifacts}/ui-review.json`,
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
