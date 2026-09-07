"""Bengali OCR benchmark.

Renders a sample of PDF pages once, runs every available OCR engine over the
identical images, and reports which engine you should build ingestion on.

Usage (from the repo root):

    # 1. see what is set up, without spending anything
    python -m tools.ocr_bench.run "doc.pdf" --pages 2,3,5 --dry-run

    # 2. benchmark, no ground truth needed
    python -m tools.ocr_bench.run "doc.pdf" --pages 2,3,5

    # 3. optional: get real CER/WER by transcribing a few pages
    python -m tools.ocr_bench.run "doc.pdf" --pages 2,3,5 --make-gold-template
    #    ...type the correct Bengali into the gold_*.txt files, then:
    python -m tools.ocr_bench.run "doc.pdf" --pages 2,3,5 --gold

Renders and OCR results are cached, so re-running is free. Use --force to redo.
Keep --pages small: a 354-page book does not need full OCR to pick an engine.
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

import pymupdf

from . import metrics
from .engines import OCR_ENGINES, TextLayer

# Bengali on a Windows console dies under cp1252 without this.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_pages(spec, page_count):
    """'2,3,5' or '4-9' or 'all' -> sorted 0-based indices."""
    if spec.strip().lower() == "all":
        return list(range(page_count))
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.update(range(lo - 1, hi))
        else:
            out.add(int(part) - 1)
    bad = [p + 1 for p in out if not 0 <= p < page_count]
    if bad:
        raise SystemExit(f"pages out of range for a {page_count}-page PDF: {bad}")
    return sorted(out)


def render(pdf_path, pages, dpi, out_dir, force=False):
    """Rasterise once; every engine then sees byte-identical input."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    paths = {}
    try:
        for i in pages:
            png = out_dir / f"page_{i + 1:04d}.png"
            if force or not png.exists():
                doc[i].get_pixmap(dpi=dpi).save(png)
            paths[i] = png
    finally:
        doc.close()
    return paths


def load_gold(gold_dir, pages):
    """Read hand-transcribed reference text, skipping untouched templates."""
    gold = {}
    for i in pages:
        f = gold_dir / f"gold_page_{i + 1:04d}.txt"
        if not f.exists():
            continue
        text = io.open(f, encoding="utf-8").read()
        stripped = "\n".join(
            ln for ln in text.splitlines() if not ln.startswith("#")
        ).strip()
        if stripped:
            gold[i] = stripped
    return gold


def write_gold_templates(gold_dir, pages, render_paths):
    gold_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for i in pages:
        f = gold_dir / f"gold_page_{i + 1:04d}.txt"
        if f.exists():
            continue
        io.open(f, "w", encoding="utf-8").write(
            f"# Ground truth for page {i + 1}.\n"
            f"# Open {render_paths[i].name} beside this file and type exactly what\n"
            f"# you see, including punctuation. Lines starting with # are ignored.\n"
            f"# Transcribing 3-5 representative pages is enough for real CER/WER.\n\n"
        )
        made.append(f)
    return made


def collect(pdf_path, pages, render_paths, engine_names, out_root, force):
    """Run each available engine over each page, caching text to disk."""
    results = {}
    timings = {}
    skipped = {}

    engines = []
    for name in engine_names:
        inst = OCR_ENGINES[name]()
        ok, why = inst.available()
        if ok:
            engines.append(inst)
        else:
            skipped[name] = why

    for i in pages:
        results[i] = {}
        # the PDF's own text layer, as the baseline being replaced
        text_layer = TextLayer(pdf_path, i)
        if text_layer.available()[0]:
            results[i]["textlayer"] = text_layer.ocr(None)

        for eng in engines:
            cache = out_root / eng.name / f"page_{i + 1:04d}.txt"
            cache.parent.mkdir(parents=True, exist_ok=True)
            if cache.exists() and not force:
                results[i][eng.name] = io.open(cache, encoding="utf-8").read()
                continue
            try:
                text, secs = eng.run(str(render_paths[i]))
            except Exception as exc:
                print(f"  ! {eng.name} failed on page {i + 1}: {exc}")
                skipped.setdefault(eng.name, f"runtime error: {exc}")
                continue
            io.open(cache, "w", encoding="utf-8").write(text)
            results[i][eng.name] = text
            timings.setdefault(eng.name, []).append(secs)
            print(f"  {eng.name} page {i + 1}: {len(text)} chars in {secs:.1f}s")

    return results, timings, skipped


def build_report(pdf_path, pages, dpi, results, timings, skipped, gold):
    engine_names = sorted({e for page in results.values() for e in page})
    lines = []
    add = lines.append

    add(f"# OCR benchmark - {Path(pdf_path).name}\n")
    add(f"- pages sampled: {[p + 1 for p in pages]}")
    add(f"- render dpi: {dpi}")
    truth = f"yes, {len(gold)} page(s)" if gold else "no (consensus mode)"
    add(f"- ground truth: {truth}\n")

    if skipped:
        add("## Engines skipped\n")
        for name, why in sorted(skipped.items()):
            add(f"- `{name}` - {why}")
        add("")

    agreements = {}
    for i in pages:
        for name, score in metrics.mean_agreement(results[i]).items():
            agreements.setdefault(name, []).append(score)

    add("## Per-engine summary\n")
    head = ("| engine | chars/pg | tokens/pg | bengali | illegal-initial | "
            "triples | agreement | s/pg |")
    if gold:
        head = head + " CER | WER |"
    add(head)
    add("|" + "---|" * (head.count("|") - 1))

    rows = []
    for name in engine_names:
        sigs = [metrics.quality_signals(results[i][name])
                for i in pages if name in results[i]]
        if not sigs:
            continue
        count = len(sigs)

        def avg(key, _sigs=sigs, _n=count):
            return sum(s[key] for s in _sigs) / _n

        agree = agreements.get(name)
        agree_mean = sum(agree) / len(agree) if agree else None
        agree_s = f"{agree_mean:.3f}" if agree_mean is not None else "-"
        secs = timings.get(name)
        secs_s = f"{sum(secs) / len(secs):.1f}" if secs else "-"

        cer_v = wer_v = None
        if gold:
            cs = [metrics.cer(gold[i], results[i][name])
                  for i in pages if i in gold and name in results[i]]
            ws = [metrics.wer(gold[i], results[i][name])
                  for i in pages if i in gold and name in results[i]]
            cs = [c for c in cs if c is not None]
            ws = [w for w in ws if w is not None]
            cer_v = sum(cs) / len(cs) if cs else None
            wer_v = sum(ws) / len(ws) if ws else None

        row = (f"| `{name}` | {avg('chars'):.0f} | {avg('tokens'):.0f} | "
               f"{avg('bengali_ratio'):.1%} | {avg('initial_dependent_rate'):.2%} | "
               f"{avg('triple_letter_runs'):.1f} | {agree_s} | {secs_s} |")
        if gold:
            row += f" {cer_v:.3f} |" if cer_v is not None else " - |"
            row += f" {wer_v:.3f} |" if wer_v is not None else " - |"
        add(row)
        rows.append({
            "name": name,
            "cer": cer_v,
            "illegal": avg("initial_dependent_rate"),
            "agreement": agree_mean,
            "chars": avg("chars"),
        })

    add("")
    inflated = [r["name"] for r in rows if r["cer"] is not None and r["cer"] > 1.0]
    if inflated:
        add(f"> **CER above 1.0 for {', '.join('`' + n + '`' for n in inflated)}.**")
        add("> That almost always means the gold file covers only part of the page,")
        add("> so the engine's extra text counts as insertions. Transcribe the whole")
        add("> page or drop it from the gold set - a partial reference makes CER")
        add("> meaningless rather than merely pessimistic.")
        add("")
    add("**illegal-initial** is the column to read first. A Bengali word cannot")
    add("begin with a dependent vowel sign, so any nonzero value is proof that")
    add("engine's output is corrupt - no ground truth required. `textlayer` is the")
    add("PDF's own text, included to show what you are replacing.")
    add("")

    add("## Ranking\n")
    candidates = [r for r in rows if r["name"] != "textlayer"]
    if not candidates:
        add("_No OCR engine ran. Set up at least one and re-run._")
    elif gold and any(r["cer"] is not None for r in candidates):
        ranked = sorted((r for r in candidates if r["cer"] is not None),
                        key=lambda r: r["cer"])
        for pos, r in enumerate(ranked, 1):
            add(f"{pos}. `{r['name']}` - CER {r['cer']:.3f}")
        add("")
        add("Ranked on CER against your transcription. This is the real answer.")
    else:
        ranked = sorted(candidates,
                        key=lambda r: (r["illegal"], -(r["agreement"] or 0), -r["chars"]))
        for pos, r in enumerate(ranked, 1):
            agree_s = "-" if r["agreement"] is None else f"{r['agreement']:.3f}"
            add(f"{pos}. `{r['name']}` - illegal-initial {r['illegal']:.2%}, "
                f"agreement {agree_s}, {r['chars']:.0f} chars/pg")
        add("")
        add("Consensus ranking, not ground truth: engines can agree and both be")
        add("wrong on the same rare conjuncts. Transcribe the lowest-agreement page")
        add("with --make-gold-template to confirm before committing.")
    add("")

    add("## Where engines disagree\n")
    add("Lowest-agreement pages are the ones worth transcribing by hand.\n")
    any_pairs = False
    for i in pages:
        pairs = {k: v for k, v in metrics.consensus_matrix(results[i]).items()
                 if "textlayer" not in k}
        if not pairs:
            continue
        any_pairs = True
        worst = min(pairs.values())
        detail = ", ".join(f"{a}/{b} {v:.3f}" for (a, b), v in sorted(pairs.items()))
        add(f"- page {i + 1}: worst pair {worst:.3f} - {detail}")
    if not any_pairs:
        add("_Needs two or more OCR engines to compare._")
    add("")
    return "\n".join(lines)


def write_side_by_side(out_root, pages, results, head_lines=12):
    """Dump each engine's first lines per page so you can eyeball the difference."""
    for i in pages:
        parts = [f"===== page {i + 1} ====="]
        for name in sorted(results[i]):
            head = "\n".join(results[i][name].splitlines()[:head_lines])
            parts.append(f"--- {name} ---\n{head}")
        io.open(out_root / f"page_{i + 1:04d}.compare.txt", "w",
                encoding="utf-8").write("\n\n".join(parts))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--pages", default="1-3",
                    help="1-based, e.g. '2,3,5' or '4-9' or 'all' (default: 1-3)")
    ap.add_argument("--dpi", type=int, default=300,
                    help="render resolution; 300 is the usual OCR sweet spot")
    ap.add_argument("--engines", default="all",
                    help=f"comma list of {','.join(OCR_ENGINES)} (default: all)")
    ap.add_argument("--out", default="tools/ocr_bench/_out")
    ap.add_argument("--gold", action="store_true",
                    help="score against hand-transcribed gold_page_*.txt files")
    ap.add_argument("--make-gold-template", action="store_true",
                    help="render pages and create empty files to transcribe into")
    ap.add_argument("--force", action="store_true", help="ignore caches and re-run")
    ap.add_argument("--dry-run", action="store_true",
                    help="show engine availability and page count, spend nothing")
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        raise SystemExit(f"no such file: {args.pdf}")

    doc = pymupdf.open(args.pdf)
    page_count = doc.page_count
    doc.close()
    pages = parse_pages(args.pages, page_count)

    names = list(OCR_ENGINES) if args.engines == "all" else [
        n.strip() for n in args.engines.split(",") if n.strip()]
    unknown = [n for n in names if n not in OCR_ENGINES]
    if unknown:
        raise SystemExit(f"unknown engine(s): {unknown}. known: {list(OCR_ENGINES)}")

    out_root = Path(args.out) / Path(args.pdf).stem
    render_dir = out_root / "render"

    print(f"{Path(args.pdf).name}: {page_count} pages, benchmarking "
          f"{len(pages)} of them at {args.dpi} dpi")

    print("\nengine availability:")
    ready = []
    for n in names:
        ok, why = OCR_ENGINES[n]().available()
        print(f"  {'OK     ' if ok else 'skipped'} {n:16s} {why}")
        if ok:
            ready.append(n)

    if args.dry_run:
        print(f"\ndry run: would OCR {len(pages)} page(s) with {ready or 'nothing'}")
        return

    print(f"\nrendering {len(pages)} page(s)...")
    render_paths = render(args.pdf, pages, args.dpi, render_dir, args.force)

    if args.make_gold_template:
        made = write_gold_templates(out_root / "gold", pages, render_paths)
        print(f"\nrendered images: {render_dir}")
        print(f"gold templates : {out_root / 'gold'}  ({len(made)} new)")
        print("\nTranscribe those files against the PNGs, then re-run with --gold.")
        return

    if not ready:
        raise SystemExit(
            "\nNo OCR engine is set up. See tools/ocr_bench/README.md - "
            "google_vision is the quickest to get running."
        )

    print(f"\nrunning {ready}...")
    results, timings, skipped = collect(
        args.pdf, pages, render_paths, ready, out_root, args.force)

    gold = load_gold(out_root / "gold", pages) if args.gold else {}
    if args.gold and not gold:
        print("\n! --gold given but no transcribed gold files found; "
              "falling back to consensus mode")

    report = build_report(args.pdf, pages, args.dpi, results, timings, skipped, gold)
    write_side_by_side(out_root, pages, results)

    report_path = out_root / "report.md"
    io.open(report_path, "w", encoding="utf-8").write(report)
    io.open(out_root / "raw.json", "w", encoding="utf-8").write(
        json.dumps({str(k + 1): v for k, v in results.items()},
                   ensure_ascii=False, indent=2))

    print("\n" + report)
    print(f"report      : {report_path}")
    print(f"side-by-side: {out_root}/page_*.compare.txt")


if __name__ == "__main__":
    main()
