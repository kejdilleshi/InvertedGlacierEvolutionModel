# Manuscript: past temperature from dated landforms (Argentière, Younger Dryas)

`manuscript.tex` → `manuscript.pdf` (Copernicus template v7.16, The Cryosphere, `manuscript` mode, 210 × 240 mm
pages as Copernicus produces them) and `manuscript_A4.pdf` (same text on A4, for printing: the 240 mm page
leaves ~6 cm blank at the bottom of an A4 sheet).

## Rebuild

```bash
# figures (reads results/, results_archive/, data/; nothing is re-run)
cd <repo root>
IGEM_DEVICE=cpu /home/klleshi/miniconda3/envs/IGEM/bin/python paper/make_figures.py
# Figure 1 (TikZ schematic)
cd paper/figures && /home/klleshi/miniconda3/envs/tex/bin/tectonic -X compile fig01_schematic.tex
# PDFs (BibTeX runs automatically)
cd .. && /home/klleshi/miniconda3/envs/tex/bin/tectonic -X compile manuscript.tex
/home/klleshi/miniconda3/envs/tex/bin/tectonic -X compile manuscript_A4.tex     # print version
```

pdfLaTeX also works (`latexmk -pdf manuscript.tex`).

## Files

| file | content |
|---|---|
| `manuscript.tex`, `references.bib` | text and bibliography (every DOI checked against Crossref/DataCite, 2026-11-09) |
| `make_figures.py` | Figures 2–10 and A1 from the archived runs |
| `figures/fig01_schematic.tex` | Figure 1 |
| `scripts/bench_memory.py`, `data/benchmark_memory.jsonl` | GPU-memory benchmark behind Fig. 10 and Table 5 (RTX 4090, 2026-11-09) |
| `copernicus.cls`, `.bst`, `.cfg`, `pdfscreen*.sty` | Copernicus LaTeX package |
| `fontawesome5.sty`, `fontawesomesymbols-*.tex` | stubs: Tectonic (XeTeX) crashes on the FontAwesome fonts the class loads; harmless with pdfLaTeX |

## Still to fill in

Co-authors and affiliations, author contributions, funding/acknowledgements, the provider of the
2012/2021 Argentière thickness and bed products, and the Zenodo DOI of the code.
