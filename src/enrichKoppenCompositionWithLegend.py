# enrichKoppenCompositionWithLegend.py
#
# Adds the official Beck et al. (2023) Köppen–Geiger class abbreviation/name
# to the already prepared basin-composition tables.
#
# INPUTS
#   data/processed/koppen_basin_static_composition_summary.csv
#   data/processed/koppen_basin_class_fractions_long.csv
#   data/interim/koppen_geiger_legend_codes.csv
#
# OUTPUTS
#   data/processed/koppen_basin_static_composition_summary_labeled.csv
#   data/processed/koppen_basin_class_fractions_long_labeled.csv
#
# Existing composition/QC fields are preserved exactly.
# No homogeneity threshold is introduced.

from pathlib import Path
import numpy as np
import pandas as pd

SUMMARY_IN = Path("data/processed/koppen_basin_static_composition_summary.csv")
LONG_IN = Path("data/processed/koppen_basin_class_fractions_long.csv")
LEGEND_IN = Path("data/interim/koppen_geiger_legend_codes.csv")

SUMMARY_OUT = Path("data/processed/koppen_basin_static_composition_summary_labeled.csv")
LONG_OUT = Path("data/processed/koppen_basin_class_fractions_long_labeled.csv")
QC_OUT = Path("results/tables/koppen_legend_enrichment_qc.csv")


def load_legend():
    legend = pd.read_csv(LEGEND_IN)

    required = {
        "KG_code", "KG_abbr", "KG_name",
        "KG_rgb_r", "KG_rgb_g", "KG_rgb_b",
    }
    missing = required - set(legend.columns)
    if missing:
        raise ValueError(f"Legend missing columns: {sorted(missing)}")

    legend["KG_code"] = pd.to_numeric(legend["KG_code"], errors="raise").astype(int)

    if legend["KG_code"].duplicated().any():
        raise ValueError("Duplicate KG_code values in legend.")

    expected = set(range(1, 31))
    found = set(legend["KG_code"])
    if found != expected:
        raise ValueError(
            f"Legend must contain exactly codes 1..30. "
            f"Missing={sorted(expected-found)}, extra={sorted(found-expected)}"
        )

    return legend.sort_values("KG_code").reset_index(drop=True)


def normalize_nullable_code(series):
    x = pd.to_numeric(series, errors="coerce")
    return x.astype("Int64")


def enrich_summary(summary, legend):
    required = {
        "HYBAS_ID",
        "KG_dom_code", "KG_dom_frac",
        "KG_second_code", "KG_second_frac",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Summary table missing required columns: {sorted(missing)}")

    out = summary.copy()
    out["HYBAS_ID"] = pd.to_numeric(out["HYBAS_ID"], errors="raise").astype("int64")
    out["KG_dom_code"] = normalize_nullable_code(out["KG_dom_code"])
    out["KG_second_code"] = normalize_nullable_code(out["KG_second_code"])

    dom = legend.rename(columns={
        "KG_code": "KG_dom_code",
        "KG_abbr": "KG_dom_abbr",
        "KG_name": "KG_dom_name",
        "KG_rgb_r": "KG_dom_rgb_r",
        "KG_rgb_g": "KG_dom_rgb_g",
        "KG_rgb_b": "KG_dom_rgb_b",
    })
    sec = legend.rename(columns={
        "KG_code": "KG_second_code",
        "KG_abbr": "KG_second_abbr",
        "KG_name": "KG_second_name",
        "KG_rgb_r": "KG_second_rgb_r",
        "KG_rgb_g": "KG_second_rgb_g",
        "KG_rgb_b": "KG_second_rgb_b",
    })

    out = out.merge(dom, on="KG_dom_code", how="left", validate="many_to_one")
    out = out.merge(sec, on="KG_second_code", how="left", validate="many_to_one")

    # Convenient display strings while retaining all numeric fields.
    out["KG_dom_label"] = np.where(
        out["KG_dom_abbr"].notna(),
        out["KG_dom_abbr"].astype(str) + " — " + out["KG_dom_name"].astype(str),
        pd.NA,
    )
    out["KG_second_label"] = np.where(
        out["KG_second_abbr"].notna(),
        out["KG_second_abbr"].astype(str) + " — " + out["KG_second_name"].astype(str),
        pd.NA,
    )

    return out


def enrich_long(long_df, legend):
    required = {"HYBAS_ID", "KG_code", "KG_pixel_count", "KG_fraction"}
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"Long table missing required columns: {sorted(missing)}")

    out = long_df.copy()
    out["HYBAS_ID"] = pd.to_numeric(out["HYBAS_ID"], errors="raise").astype("int64")
    out["KG_code"] = pd.to_numeric(out["KG_code"], errors="raise").astype(int)

    out = out.merge(
        legend,
        on="KG_code",
        how="left",
        validate="many_to_one",
    )

    out["KG_label"] = (
        out["KG_abbr"].astype(str)
        + " — "
        + out["KG_name"].astype(str)
    )

    return out


def qc(summary, long_df, legend):
    rows = []

    rows.append({
        "check": "legend_exactly_30_codes",
        "value": len(legend),
        "pass": len(legend) == 30,
    })

    bad_dom = summary[
        summary["KG_dom_code"].notna() & summary["KG_dom_abbr"].isna()
    ]
    rows.append({
        "check": "summary_dominant_codes_all_mapped",
        "value": len(bad_dom),
        "pass": len(bad_dom) == 0,
    })

    bad_sec = summary[
        summary["KG_second_code"].notna() & summary["KG_second_abbr"].isna()
    ]
    rows.append({
        "check": "summary_second_codes_all_mapped",
        "value": len(bad_sec),
        "pass": len(bad_sec) == 0,
    })

    bad_long = long_df[long_df["KG_abbr"].isna()]
    rows.append({
        "check": "long_table_codes_all_mapped",
        "value": len(bad_long),
        "pass": len(bad_long) == 0,
    })

    # Fractions should still sum to 1 over valid classes for every basin
    # (allowing tiny floating-point error).
    sums = long_df.groupby("HYBAS_ID")["KG_fraction"].sum()
    max_err = float((sums - 1.0).abs().max()) if len(sums) else np.nan
    rows.append({
        "check": "max_abs_long_fraction_sum_minus_1",
        "value": max_err,
        "pass": bool(np.isfinite(max_err) and max_err <= 1e-6),
    })

    return pd.DataFrame(rows)


def main():
    print("--- Enrich Köppen basin composition with official class names ---")

    for f in [SUMMARY_IN, LONG_IN, LEGEND_IN]:
        if not f.exists():
            raise FileNotFoundError(f"Missing input: {f}")

    legend = load_legend()
    summary = pd.read_csv(SUMMARY_IN)
    long_df = pd.read_csv(LONG_IN)

    summary_labeled = enrich_summary(summary, legend)
    long_labeled = enrich_long(long_df, legend)
    qc_df = qc(summary_labeled, long_labeled, legend)

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    LONG_OUT.parent.mkdir(parents=True, exist_ok=True)
    QC_OUT.parent.mkdir(parents=True, exist_ok=True)

    summary_labeled.to_csv(SUMMARY_OUT, index=False)
    long_labeled.to_csv(LONG_OUT, index=False)
    qc_df.to_csv(QC_OUT, index=False)

    print(f"✅ Saved: {SUMMARY_OUT}")
    print(f"✅ Saved: {LONG_OUT}")
    print(f"✅ Saved: {QC_OUT}")

    print("\nLegend spot-check:")
    print(
        legend.loc[
            legend["KG_code"].isin([1, 2, 3, 26, 27, 29, 30]),
            ["KG_code", "KG_abbr", "KG_name"],
        ].to_string(index=False)
    )

    print("\nQC:")
    print(qc_df.to_string(index=False))

    if not qc_df["pass"].all():
        raise RuntimeError("One or more legend-enrichment QC checks failed.")

    print("\n--- Done ---")


if __name__ == "__main__":
    main()
