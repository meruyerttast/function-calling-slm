# Final Presentation Slides

Beamer source for the INFO 290 final presentation of "Pushing the Sub-3B
Frontier for Function-Calling SLMs". Targets a 15-20 minute slot per the
issue #10 spec.

## Compile

From this directory:

```bash
make
```

This runs `pdflatex slides.tex` twice (Beamer needs two passes for cross-refs)
and produces `slides.pdf`.

To clean intermediate files:

```bash
make clean
```

## Theme

The deck uses the `metropolis` Beamer theme. It ships with most modern
TeX Live / MacTeX / MiKTeX installs and with Overleaf out of the box. If
`pdflatex` errors with `! LaTeX Error: File 'beamerthememetropolis.sty'
not found`, edit the top of `slides.tex`:

```tex
% Replace
\usetheme{metropolis}

% with
\usetheme{default}
\usecolortheme{seahorse}
```

Both options are kept side by side in the file as comments.

## Frame outline

1. Title (project, team, course)
2. The problem (sub-3B function-calling gap, why on-device matters)
3. Approach (cold-start GRPO, masking, curriculum) with paper citations
4. Data (xLAM 60k -> 4K, four-stage filter)
5. Function masking (before / after example)
6. Variants table V1-V6
7. Results: Non-Live (V1 concrete, V2-V5 placeholders)
8. Results: V4-weighted overall (V1 partial, rest placeholders)
9. Three systems contributions (bfcl-eval patches)
10. Limitations and future work
11. Q&A / thank-you

## Filling in V2-V5 numbers

When BFCL V4 runs for V2-V5 land in `results/v{2,3,4,5}/scores.json`:

1. Open `slides.tex`.
2. Find the two results tables. They are tagged with the comments
   `Results: BFCL V4 Non-Live` and `Results: V4-Weighted Overall`.
3. For each variant row, replace the `\tbd` cells with the matching
   percentage from `scores.json`. Non-Live cell mapping:
   - `Overall` -> `top_level.non_live`
   - `Simple`, `Multiple`, `Parallel`, `Parallel-Mult.`, `Irrelevance`
     -> read from the raw bfcl CSV at
     `<bfcl_score_dir>/<variant>/data_non_live.csv`, columns
     "Simple Acc", "Multiple Acc", "Parallel Acc", "Parallel Multiple Acc",
     "Irrelevance Acc". (Sub-buckets are not stored in `scores.json`.)
4. For the V4-Weighted Overall table:
   - Use the `top_level.{non_live,live,hallucination,multi_turn,agentic}`
     fields and the `overall` field for the right-most column.
5. Remove the `\todo{...}` line below the table once the numbers are in
   and replace with a one-paragraph reading of the results.
6. Recompile with `make`.

## Placeholders

`\todo{...}` and `\tbd` are defined at the top of `slides.tex` and render
in red so they are obvious during a dry run. Search for `TODO` / `\tbd`
before final delivery.

## Notes

- No animations or pauses. Every frame is static.
- Citations are inline footnotes pointing at arXiv URLs (no bibtex).
- `aspectratio=169` for widescreen projection.
