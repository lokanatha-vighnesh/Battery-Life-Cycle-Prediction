"""
Merge the three batch-date CSV files separately for each cycle limit.

Outputs:
    battery_extracted/merged_cycles_10.csv
    battery_extracted/merged_cycles_30.csv
    battery_extracted/merged_cycles_100.csv

Original CSV files remain unchanged.
"""
from pathlib import Path
import sys
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "battery_extracted"
CYCLE_LIMITS = [10, 30, 100]


def main():
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    for limit in CYCLE_LIMITS:
        folder = INPUT_DIR / f"cycles_{limit}"
        if not folder.is_dir():
            print(f"WARNING: Folder not found, skipping: {folder}")
            continue

        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            print(f"WARNING: No CSV files found in {folder}; skipping.")
            continue

        frames = []
        expected_columns = None

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            if expected_columns is None:
                expected_columns = list(df.columns)
            elif set(df.columns) != set(expected_columns):
                raise ValueError(
                    f"Column mismatch in {csv_file.name}. "
                    f"Expected {expected_columns}; found {list(df.columns)}"
                )

            df = df[expected_columns]
            frames.append(df)
            print(f"Read {csv_file.name}: {len(df):,} rows")

        merged = pd.concat(frames, ignore_index=True)
        output_file = INPUT_DIR / f"merged_cycles_{limit}.csv"
        merged.to_csv(output_file, index=False)

        print(
            f"\nCreated: {output_file}\n"
            f"Source files: {len(csv_files)}\n"
            f"Total rows: {len(merged):,}\n"
            f"Columns: {len(merged.columns)}\n"
        )

    print("Merge process complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
