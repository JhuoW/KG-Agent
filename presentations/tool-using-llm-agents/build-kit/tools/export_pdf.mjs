// PDF export: prints the built deck with Chromium, one 1280x720 page per slide,
// one file per language. Text, math, and diagrams stay vector and selectable.
// For Chinese glyphs it embeds Noto Sans SC / Noto Serif SC (devDependencies), so the
// result does not depend on the fonts installed on the exporting machine.
// Usage: node tools/export_pdf.mjs [--out dist] [--lang en|zh]
import { chromium } from "playwright";
import { existsSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const outDir = resolve(args.includes("--out") ? args[args.indexOf("--out") + 1] : resolve(root, "dist"));
const langs = args.includes("--lang") ? [args[args.indexOf("--lang") + 1]] : ["en", "zh"];
const html = pathToFileURL(resolve(root, "dist/tool-using-llm-agents.html")).href;
const exe = process.env.CHROMIUM_PATH || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const cjkCss = [
  "@fontsource/noto-sans-sc/400.css",
  "@fontsource/noto-sans-sc/600.css",
  "@fontsource/noto-serif-sc/600.css",
].map((p) => pathToFileURL(resolve(root, "node_modules", p)).href);

mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch(existsSync(exe) ? { executablePath: exe } : {});
for (const lang of langs) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 772 }, colorScheme: "light" });
  await page.goto(html);
  for (const url of cjkCss) await page.addStyleTag({ url });
  await page.addStyleTag({ content: `:root {
    --font-body: "Source Sans 3", "Noto Sans SC", sans-serif;
    --font-title: "Source Serif 4", "Noto Serif SC", serif;
    --font-mono: "JetBrains Mono", "Noto Sans SC", monospace; }
    .katex-mathml { display: none !important; } /* invisible accessibility copy; keeps PDF text clean */` });
  const n = await page.evaluate(async (l) => {
    document.documentElement.setAttribute("data-theme", "light");
    window.deckAPI.setLang(l);
    const slides = [...document.querySelectorAll(".slide")];
    // Lay out every slide once so the browser requests every glyph subset it needs.
    for (const s of slides) { s.classList.add("active"); void s.offsetHeight; s.classList.remove("active"); }
    await document.fonts.ready;
    await new Promise((r) => setTimeout(r, 300));
    await document.fonts.ready;
    window.deckAPI.fitAll(true); // recompute content scale with the embedded fonts
    const over = [], fits = [];
    slides.forEach((s, i) => {
      s.classList.add("active");
      const b = s.querySelector(".s-body").getBoundingClientRect();
      const f = s.querySelector(".s-fit").getBoundingClientRect();
      if (f.height > b.height + 1) over.push(i + 1);
      fits.push(parseFloat(s.dataset["fit" + l] || "1"));
      s.classList.remove("active");
    });
    window.deckAPI.go(0, false);
    return { n: slides.length, over, min: Math.min(...fits) };
  }, lang);
  if (n.over.length) { console.error(`${lang}: content overflows on slides ${n.over.join(", ")}`); process.exitCode = 1; }
  const path = resolve(outDir, `tool-using-llm-agents-${lang}.pdf`);
  await page.pdf({ path, preferCSSPageSize: true, printBackground: true });
  console.log(`${lang}: ${n.n} pages, no overflow: ${!n.over.length}, smallest content scale ${n.min} -> ${path}`);
  await page.close();
}
await browser.close();
