// Batch KaTeX renderer used by build.py.
// stdin : JSON {"macros": {...}, "items": [{"tex": "...", "display": bool}, ...]}
// stdout: JSON ["<span class=katex>...</span>", ...]
// Any KaTeX parse error aborts the build (throwOnError: true).
import katex from "katex";

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const { macros, items } = JSON.parse(Buffer.concat(chunks).toString("utf8"));

const out = items.map(({ tex, display }, i) => {
  try {
    return katex.renderToString(tex, {
      displayMode: display,
      throwOnError: true,
      output: "htmlAndMathml",
      strict: "error",
      trust: false,
      macros: { ...macros },
    });
  } catch (err) {
    process.stderr.write(`KaTeX error in item ${i}: ${tex}\n${err.message}\n`);
    process.exit(1);
  }
});
process.stdout.write(JSON.stringify(out));
