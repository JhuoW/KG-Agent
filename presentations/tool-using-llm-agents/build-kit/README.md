# Build kit: "Tool-Using LLM Agents" slide deck

A bilingual (English / Simplified Chinese) academic slide deck generated from Python content
modules into one self-contained HTML file. KaTeX math is rendered at build time, and the fonts
(Source Serif 4, Source Sans 3, JetBrains Mono, KaTeX) are embedded as data URIs, so the file
works offline. Chinese text uses the viewer's system CJK fonts.

## Requirements
- Python 3.10+ (standard library only; Pillow is not needed)
- Node.js 18+ and npm
- Chromium for the layout check (Playwright; set `CHROMIUM_PATH` if it is not at the default path)

## Build and check
```bash
npm ci                                   # pinned: katex 0.16.11, @fontsource/*, playwright 1.56.1
python3 build.py                         # -> dist/tool-using-llm-agents.html (+ .artifact.html)
python3 tools/lint.py                    # language rules and EN/ZH consistency, exit 1 on error
node tools/check_layout.mjs              # overflow check, both languages; --shots DIR saves PNGs
python3 tools/selfcheck.py               # reproduces the draft's numerical self-check
python3 tools/make_evidence.py           # rewrites evidence/evidence.json and sources.json
```
If `npm ci` cannot download a browser, set `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1` and point
`CHROMIUM_PATH` at an installed Chromium.

## Layout of the kit
| Path | Purpose |
|---|---|
| `build.py` | Generator: loads content, substitutes computed values, renders math, writes HTML |
| `deck/content/s00_intro.py` ... `s09_summary.py` | Content modules, one per section; every prose field has `en` and `zh` |
| `deck/blocks.py` | Block constructors (paragraph, list, math, table, callout, diagram, flow) |
| `deck/numbers.py` | Computes every number shown on a slide; slides reference values as `@@key@@` |
| `deck/diagrams.py` | Inline SVG diagrams with bilingual labels |
| `deck/theme.css`, `deck/runtime.js` | Styles (light and dark tokens) and navigation, language toggle, content fitting |
| `tools/render_math.mjs` | Batch KaTeX renderer (parse errors fail the build) |
| `tools/lint.py` | Banned words, em dashes, question marks, sentence length, synthesis markers, math and number parity between EN and ZH |
| `tools/check_layout.mjs` | Renders every slide in both languages and reports overflow |
| `tools/selfcheck.py` | Lattice recursion versus path enumeration; posterior gradient versus finite differences |
| `tools/make_evidence.py` | Claim-to-source map with status labels |
| `evidence/` | `evidence.json` (48 claims), `sources.json` (13 sources, with read depth), `computed_values.json`, `slide_index.json` |
| `changelog/` | `CHANGELOG.md` (build history) and `source_corrections.md` (changes relative to the v2 draft) |

## Editing
- Text: edit the content module; keep `en` and `zh` in step. The linter fails if math segments or
  numbers differ between the two languages.
- Numbers: change inputs in `deck/numbers.py`; slides update through the placeholders.
- Method name: `METHOD_NAME` in `build.py` (currently the drafts' acronym, never expanded).
- Writing rules enforced by the linter: prose sentences of 12-25 words (hard limit 35), no em dashes,
  no question marks, no "Next:" pointers, no buzzwords, "(my synthesis)" / "（个人归纳）" at most
  once per slide.

## Presenting
Arrow keys, Page Up/Down, Space and Home/End move between slides; `L` switches language; the
toolbar also has a theme switch. `#s12` in the URL opens slide 12. Below 820 px width the deck
switches to a scrolling reading layout. Printing from a browser produces one page per slide in
the current language.
