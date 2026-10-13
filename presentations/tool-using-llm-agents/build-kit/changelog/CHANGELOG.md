# Changelog

## 1.0.0 (2026-10-13)

First complete version: 47 slides, English and Simplified Chinese in one file.

### Content
- Built from the v2 draft (Notion, 2026-10-13); definitions that v2 uses without stating
  (q, m, ρ, B⁻, B⁺, C, ε), the worked-example policy scores, scope statements, and the evaluation
  plan come from the earlier draft (Notion, 2026-10-08) and carry the "From the earlier draft" label.
- FuncBenchGen statistics verified against the full paper text (arXiv 2509.26553v2, Table 3).
- Bibliographic data for Bernstein (1966), Brightwell and Winkler (1991), CTC (2006) and AppWorld (2024)
  verified via Crossref.
- All worked-example numbers computed in `deck/numbers.py` and injected through `@@key@@` placeholders,
  with brute-force path enumeration as a cross-check.
- Independent reproduction of the earlier draft's numerical self-check (`tools/selfcheck.py`).
- Source corrections listed in `source_corrections.md`.

### Build iterations before release
1. First build: lint reported 3 errors (dates written differently in EN and ZH; "Six" spelled out in
   EN but written as a digit in ZH) and 1 warning (a 27-word sentence). Fixed: ISO dates, digits in both
   languages, sentence shortened.
2. Visual review of all 94 renders (47 slides × 2 languages):
   - Chinese slide 25 showed text overlapping a callout although the overflow check passed. Cause: flex
     items in the slide body could shrink below their content height. Fix: body items no longer shrink,
     so any overflow becomes measurable.
   - Many slides left the lower third empty. Fix: runtime content fitting (CSS zoom between 0.85 and
     1.22, chosen per slide and language so content fills the body without vertical or horizontal
     overflow).
   - Lattice diagrams were too small and edge labels collided with nodes. Fix: rectangular nodes,
     labels offset along the edge normal, compact 566-unit canvas, larger type.
   - Per-path versus shared-state diagram redrawn vertically to suit a column.
   - Process flows changed from arrow chains (which left dangling arrows at line ends) to numbered grids.
   - Flight/hotel graph boxes widened so monospace labels stay inside.
   - Notation table symbol column set to no-wrap.
3. Zooming exposed two display formulas that scrolled sideways in narrow columns; the fit now also
   rejects any scale that makes a formula, table, or code block scroll.
4. Checked at 400 px width (reading layout, no horizontal scroll) and in dark theme.
