"""
Update battery barcodes in extracted cycle CSV files in place.

Expected directory structure:
project_folder/
├── update_barcodes.py
├── battery_barcodes.csv          # or battery_barcodes(1).csv
└── battery_extracted/
    ├── cycles_10/
    ├── cycles_30/
    └── cycles_100/

Mapping CSV columns required:
    source_file, cell_idx, barcode

The mapping is matched using the batch date extracted from source_file and
the cell_idx in each extracted CSV. Since the mapping file has no cycle-level
column, the same actual barcode is applied to every cycle for that cell.

By default, this script creates a .bak copy of each CSV before overwriting it.
Set CREATE_BACKUPS = False if you do not want backup files.
"""

from pathlib import Path
import re
import shutil
import sys

import pandas as pd


# ---------- CONFIGURATION ----------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "battery_extracted"

# The script checks these mapping filenames in order.
MAPPING_CANDIDATES = [
    BASE_DIR / "battery_barcodes.csv",
    BASE_DIR / "battery_barcodes(1).csv",
]

CYCLE_FOLDERS = ["cycles_10", "cycles_30", "cycles_100"]
CREATE_BACKUPS = True
# -----------------------------------


DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


def find_mapping_file() -> Path:
    for path in MAPPING_CANDIDATES:
        if path.is_file():
            return path

    candidates = ", ".join(str(p) for p in MAPPING_CANDIDATES)
    raise FileNotFoundError(
        "Could not find the barcode mapping CSV. Place it beside this script "
        f"with one of these names: {candidates}"
    )


def extract_batch_date(value: object) -> str | None:
    match = DATE_PATTERN.search(str(value))
    return match.group(1) if match else None


def main() -> int:
    mapping_path = find_mapping_file()
    print(f"Mapping file: {mapping_path}")
    print(f"Extracted data folder: {OUTPUT_DIR}")

    mapping_df = pd.read_csv(mapping_path, dtype={"barcode": str})

    required_mapping_columns = {"source_file", "cell_idx", "barcode"}
    missing = required_mapping_columns - set(mapping_df.columns)
    if missing:
        raise ValueError(
            f"Mapping CSV is missing required columns: {sorted(missing)}"
        )

    mapping_df["batch_date_key"] = mapping_df["source_file"].map(
        extract_batch_date
    )
    mapping_df["cell_idx"] = pd.to_numeric(
        mapping_df["cell_idx"], errors="raise"
    ).astype(int)
    mapping_df["barcode"] = mapping_df["barcode"].astype("string").str.strip()

    if mapping_df["batch_date_key"].isna().any():
        bad_rows = mapping_df[mapping_df["batch_date_key"].isna()]
        raise ValueError(
            "Could not extract a YYYY-MM-DD batch date from some source_file "
            f"values. Example rows:\n{bad_rows.head().to_string(index=False)}"
        )

    if mapping_df["barcode"].isna().any() or mapping_df["barcode"].eq("").any():
        raise ValueError("The mapping CSV contains missing or empty barcodes.")

    duplicate_mask = mapping_df.duplicated(
        ["batch_date_key", "cell_idx"], keep=False
    )
    if duplicate_mask.any():
        duplicates = mapping_df.loc[
            duplicate_mask, ["source_file", "batch_date_key", "cell_idx", "barcode"]
        ]
        raise ValueError(
            "Duplicate batch-date/cell-index mappings found. Resolve these "
            f"before updating files:\n{duplicates.to_string(index=False)}"
        )

    barcode_map = {
        (row.batch_date_key, int(row.cell_idx)): str(row.barcode)
        for row in mapping_df.itertuples(index=False)
    }
    print(f"Loaded {len(barcode_map)} barcode mappings.")

    csv_files = []
    for folder_name in CYCLE_FOLDERS:
        folder = OUTPUT_DIR / folder_name
        if not folder.is_dir():
            print(f"WARNING: Folder not found; skipping: {folder}")
            continue
        csv_files.extend(sorted(folder.glob("*.csv")))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in the expected folders under {OUTPUT_DIR}"
        )

    # Preflight every CSV before changing any file.
    plans = []
    errors = []

    for csv_path in csv_files:
        batch_date = extract_batch_date(csv_path.stem)
        if not batch_date:
            errors.append(f"{csv_path}: filename does not contain YYYY-MM-DD.")
            continue

        try:
            df = pd.read_csv(csv_path, dtype={"barcode": str})
        except Exception as exc:
            errors.append(f"{csv_path}: could not read CSV ({exc}).")
            continue

        required_csv_columns = {"cell_idx", "barcode", "cycle_index"}
        missing_csv_columns = required_csv_columns - set(df.columns)
        if missing_csv_columns:
            errors.append(
                f"{csv_path}: missing columns {sorted(missing_csv_columns)}."
            )
            continue

        try:
            df["cell_idx"] = pd.to_numeric(
                df["cell_idx"], errors="raise"
            ).astype(int)
        except Exception as exc:
            errors.append(f"{csv_path}: invalid cell_idx values ({exc}).")
            continue

        mapped = df["cell_idx"].map(
            lambda idx: barcode_map.get((batch_date, int(idx)))
        )
        matched_mask = mapped.notna()

        plans.append(
            {
                "path": csv_path,
                "date": batch_date,
                "df": df,
                "mapped": mapped,
                "matched_mask": matched_mask,
                "matched_rows": int(matched_mask.sum()),
                "unmatched_rows": int((~matched_mask).sum()),
            }
        )

    if errors:
        print("\nPreflight failed. No CSV files were modified:")
        for error in errors:
            print(f"  - {error}")
        return 1

    # Do not overwrite files if no rows in any file can be mapped.
    if sum(plan["matched_rows"] for plan in plans) == 0:
        raise ValueError(
            "No rows matched the mapping. Check the batch dates, cell_idx "
            "values, and the location/content of the mapping CSV."
        )

    total_updated = 0
    total_unmatched = 0

    for plan in plans:
        csv_path = plan["path"]
        df = plan["df"]
        mapped = plan["mapped"]
        matched_mask = plan["matched_mask"]

        if plan["matched_rows"] == 0:
            print(
                f"NO MATCHES: {csv_path} "
                f"(batch date {plan['date']}); file left unchanged."
            )
            total_unmatched += plan["unmatched_rows"]
            continue

        if CREATE_BACKUPS:
            backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
            shutil.copy2(csv_path, backup_path)

        # Update only rows with a mapping; all other columns are preserved.
        df.loc[matched_mask, "barcode"] = mapped.loc[matched_mask]

        # Overwrite the original CSV in place.
        df.to_csv(csv_path, index=False)

        # Verify every mapped row after saving.
        check_df = pd.read_csv(csv_path, dtype={"barcode": str})
        expected = check_df["cell_idx"].map(
            lambda idx: barcode_map.get((plan["date"], int(idx)))
        )
        check_mask = expected.notna()
        verified = (
            check_df.loc[check_mask, "barcode"].astype(str).reset_index(drop=True)
            == expected.loc[check_mask].astype(str).reset_index(drop=True)
        ).all()

        if not verified:
            raise RuntimeError(
                f"Verification failed after updating {csv_path}. "
                "Restore the .bak file if necessary."
            )

        total_updated += plan["matched_rows"]
        total_unmatched += plan["unmatched_rows"]

        print(
            f"UPDATED: {csv_path} | rows={len(df)} | "
            f"updated={plan['matched_rows']} | "
            f"unmatched={plan['unmatched_rows']} | verified={verified}"
        )

    print("\n========== SUMMARY ==========")
    print(f"CSV files found: {len(csv_files)}")
    print(f"CSV files checked: {len(plans)}")
    print(f"Rows updated: {total_updated}")
    print(f"Rows without a mapping: {total_unmatched}")
    print(f"Backups created: {CREATE_BACKUPS}")
    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
