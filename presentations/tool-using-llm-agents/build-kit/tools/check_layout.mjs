// Layout check: renders every slide in English and Chinese at 1280x720 after the
// runtime fit (content scale 0.85 to 1.22) and reports content that still overflows,
// math or tables that need a scrollbar, KaTeX errors, and missing web fonts.
// Slides that need a scale below 1 are listed as notes. --shots <dir> saves PNGs.
// Usage: node tools/check_layout.mjs [--shots out/shots] [--only 3,7]
import { chromium } from "playwright";
import { existsSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const shotDir = args.includes("--shots") ? resolve(args[args.indexOf("--shots") + 1]) : null;
const only = args.includes("--only") ? new Set(args[args.indexOf("--only") + 1].split(",").map(Number)) : null;
const langs = args.includes("--lang") ? [args[args.indexOf("--lang") + 1]] : ["en", "zh"];
const file = pathToFileURL(resolve(root, "dist/tool-using-llm-agents.html")).href;
const exe = process.env.CHROMIUM_PATH || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";

const browser = await chromium.launch(existsSync(exe) ? { executablePath: exe } : {});
const page = await browser.newPage({ viewport: { width: 1280, height: 720 + 52 }, deviceScaleFactor: 1 });
await page.goto(file);
await page.evaluate(() => document.fonts.ready);
if (shotDir) mkdirSync(shotDir, { recursive: true });

let problems = 0;
const fitStats = {};
for (const lang of langs) {
  await page.evaluate((l) => window.deckAPI.setLang(l), lang);
  const n = await page.evaluate(() => document.querySelectorAll(".slide").length);
  fitStats[lang] = [];
  for (let i = 0; i < n; i++) {
    if (only && !only.has(i + 1)) continue;
    const r = await page.evaluate((k) => {
      window.deckAPI.go(k, false);
      const s = document.querySelectorAll(".slide")[k];
      const body = s.querySelector(".s-body"), box = s.querySelector(".s-fit");
      const lang = document.documentElement.getAttribute("data-lang");
      const fit = parseFloat(s.dataset["fit" + lang] || "1");
      const out = { id: s.dataset.id, fit, issues: [], notes: [] };
      const bb = body.getBoundingClientRect(), fb = box.getBoundingClientRect();
      if (fb.height > bb.height + 1) out.issues.push(`content overflows by ${Math.round(fb.height - bb.height)}px at fit ${fit}`);
      if (fit < 1) out.notes.push(`shrunk to fit ${fit}`);
      for (const el of s.querySelectorAll(".mblock, .tblwrap, pre.code")) {
        if (el.offsetParent === null) continue;
        if (el.scrollWidth > el.clientWidth + 1)
          out.issues.push(`${el.className.split(" ")[0]} scrolls horizontally by ${el.scrollWidth - el.clientWidth}px`);
      }
      for (const el of s.querySelectorAll(".s-body *")) {
        if (el.offsetParent === null || el.closest(".mblock, .tblwrap, pre, .katex-mathml")) continue;
        const b = el.getBoundingClientRect();
        if (b.width && (b.right > bb.right + 1 || b.left < bb.left - 1)) {
          out.issues.push(`element outside body: <${el.tagName.toLowerCase()} class="${el.className.baseVal ?? el.className}">`);
          break;
        }
      }
      const kerr = s.querySelectorAll(".katex-error").length;
      if (kerr) out.issues.push(`${kerr} KaTeX errors`);
      return out;
    }, i);
    fitStats[lang].push(r.fit);
    if (r.issues.length) {
      problems += r.issues.length;
      console.log(`${lang} #${i + 1} ${r.id}: ${r.issues.join("; ")}`);
    }
    if (r.notes.length) console.log(`note ${lang} #${i + 1} ${r.id}: ${r.notes.join("; ")}`);
    if (shotDir) {
      const clip = await page.evaluate((k) => { const b = document.querySelectorAll(".slide")[k].getBoundingClientRect();
        return { x: b.x, y: b.y, width: b.width, height: b.height }; }, i);
      await page.screenshot({ path: `${shotDir}/${String(i + 1).padStart(2, "0")}-${lang}.png`, clip });
    }
  }
}
for (const [lang, v] of Object.entries(fitStats)) {
  if (!v.length) continue;
  const sorted = [...v].sort((a, b) => a - b);
  console.log(`${lang}: content scale min ${sorted[0]}, median ${sorted[Math.floor(v.length / 2)]}, max ${sorted[v.length - 1]}`);
}
const fonts = await page.evaluate(() => ["600 36px 'Source Serif 4'", "400 21px 'Source Sans 3'", "400 15px 'JetBrains Mono'", "20px KaTeX_Main"]
  .map((f) => [f, document.fonts.check(f)]));
for (const [f, ok] of fonts) if (!ok) { problems++; console.log(`font not available: ${f}`); }
await browser.close();
console.log(problems ? `${problems} layout problems` : "layout check passed: no overflow in any slide or language");
process.exit(problems ? 1 : 0);
