# summarizeKoppenBasinComposition.py
#
# Convert a QGIS Zonal Histogram result (KG_1 ... KG_30) into one clean
# static-climate composition record per HydroBASINS L4 basin.
#
# Fractions are calculated among VALID Köppen pixels only.
# KG_NODATA is retained separately as a coverage/QC field.
#
# No dominant/mixed threshold is imposed here. Keep KG_dom_frac and entropy
# continuous for the scientific comparison.

from pathlib import Path
import csv
import math

INPUT_CSV = Path("data/interim/koppen_geiger_1991_2020_1km_6933.csv")

OUTPUT_SUMMARY = Path("data/processed/koppen_basin_static_composition_summary.csv")
OUTPUT_LONG = Path("data/processed/koppen_basin_class_fractions_long.csv")
OUTPUT_STATS = Path("results/tables/koppen_basin_static_composition_stats.csv")

QC_VALID_COVERAGE_THRESHOLD = 0.90


def fnum(v):
    if v in (None, ""):
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def main():
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    kg_cols = [f"KG_{i}" for i in range(1, 31)]
    missing = [c for c in ["HYBAS_ID"] + kg_cols if c not in fieldnames]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    meta_cols = [
        c for c in
        ["HYBAS_ID", "MAIN_BAS", "PFAF_ID", "SUB_AREA", "UP_AREA", "basin_km2"]
        if c in fieldnames
    ]

    summary_rows = []
    long_rows = []

    for r in rows:
        counts = {i: fnum(r.get(f"KG_{i}", 0)) for i in range(1, 31)}
        valid_total = sum(counts.values())
        nodata = fnum(r.get("KG_NODATA", 0))
        all_total = valid_total + nodata

        valid_coverage = (
            valid_total / all_total
            if all_total > 0
            else math.nan
        )

        positive = [(i, c) for i, c in counts.items() if c > 0]
        positive.sort(key=lambda x: (-x[1], x[0]))

        fractions = {
            i: (c / valid_total if valid_total > 0 else math.nan)
            for i, c in counts.items()
        }

        dom_code = positive[0][0] if positive else ""
        dom_frac = fractions[dom_code] if positive else math.nan

        second_code = positive[1][0] if len(positive) > 1 else ""
        second_frac = fractions[second_code] if len(positive) > 1 else 0.0

        pvals = (
            [c / valid_total for _, c in positive]
            if valid_total > 0
            else []
        )

        # Shannon entropy: 0 for one class, larger for more/evenly mixed classes.
        H = -sum(p * math.log(p) for p in pvals if p > 0)

        # Normalize against the full 30-class coding scheme.
        H_norm30 = H / math.log(30) if pvals else math.nan

        # Effective number of equally abundant classes.
        effective_classes = math.exp(H) if pvals else math.nan

        out = {c: r.get(c, "") for c in meta_cols}
        out.update({
            "KG_valid_pixel_count": int(valid_total),
            "KG_nodata_pixel_count": int(nodata),
            "KG_valid_coverage": valid_coverage,
            "KG_dom_code": dom_code,
            "KG_dom_frac": dom_frac,
            "KG_second_code": second_code,
            "KG_second_frac": second_frac,
            "KG_top2_frac": (
                dom_frac + second_frac if positive else math.nan
            ),
            "KG_n_classes_any": len(positive),
            "KG_n_classes_ge_1pct": sum(p >= 0.01 for p in pvals),
            "KG_n_classes_ge_5pct": sum(p >= 0.05 for p in pvals),
            "KG_shannon_entropy": H if pvals else math.nan,
            "KG_entropy_norm30": H_norm30,
            "KG_effective_classes": effective_classes,
            "KG_qc_pass_90pct_valid": bool(
                valid_total > 0
                and valid_coverage >= QC_VALID_COVERAGE_THRESHOLD
            ),
        })
        summary_rows.append(out)

        for code, count in positive:
            long_rows.append({
                "HYBAS_ID": r["HYBAS_ID"],
                "KG_code": code,
                "KG_pixel_count": int(count),
                "KG_fraction": fractions[code],
            })

    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_LONG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_STATS.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_SUMMARY.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    with OUTPUT_LONG.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["HYBAS_ID", "KG_code", "KG_pixel_count", "KG_fraction"],
        )
        w.writeheader()
        w.writerows(long_rows)

    valid = [r for r in summary_rows if r["KG_valid_pixel_count"] > 0]
    doms = sorted(float(r["KG_dom_frac"]) for r in valid)

    def percentile(vals, p):
        if not vals:
            return math.nan
        if len(vals) == 1:
            return vals[0]
        k = (len(vals) - 1) * p
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - k) + vals[hi] * (k - lo)

    stats = [
        ("total_basins", len(summary_rows)),
        ("basins_with_valid_KG", len(valid)),
        ("basins_without_valid_KG", len(summary_rows) - len(valid)),
        (
            "basins_qc_pass_valid_coverage_ge_0.90",
            sum(bool(r["KG_qc_pass_90pct_valid"]) for r in summary_rows),
        ),
        ("median_dominant_fraction", percentile(doms, 0.50)),
        ("p25_dominant_fraction", percentile(doms, 0.25)),
        ("p75_dominant_fraction", percentile(doms, 0.75)),
        ("dominant_fraction_ge_0.90_count", sum(v >= 0.90 for v in doms)),
        ("dominant_fraction_ge_0.80_count", sum(v >= 0.80 for v in doms)),
        ("dominant_fraction_ge_0.70_count", sum(v >= 0.70 for v in doms)),
        ("dominant_fraction_lt_0.60_count", sum(v < 0.60 for v in doms)),
    ]

    with OUTPUT_STATS.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(stats)

    print("============================================================")
    print("KÖPPEN–GEIGER BASIN COMPOSITION SUMMARY")
    print("============================================================")
    print(f"Basins: {len(summary_rows):,}")
    print(f"With valid KG pixels: {len(valid):,}")
    print(
        f"QC pass (>=90% valid coverage): "
        f"{sum(bool(r['KG_qc_pass_90pct_valid']) for r in summary_rows):,}"
    )
    print(f"Median dominant-class fraction: {percentile(doms, 0.50):.3f}")
    print(f"Dominant >= 0.90: {sum(v >= 0.90 for v in doms):,}")
    print(f"Dominant >= 0.80: {sum(v >= 0.80 for v in doms):,}")
    print(f"Dominant <  0.60: {sum(v < 0.60 for v in doms):,}")
    print("")
    print(f"Saved: {OUTPUT_SUMMARY}")
    print(f"Saved: {OUTPUT_LONG}")
    print(f"Saved: {OUTPUT_STATS}")


if __name__ == "__main__":
    main()
