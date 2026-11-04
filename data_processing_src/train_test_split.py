#```python
from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SPLIT_FILE = Path(r"C:\Users\lokan\Document\DL\Weekly_Bundles\Release_Bundle_RevD2\Release_Bundle_RevD2\I3\I3_cell_split.csv")
MERGED_DIR = BASE_DIR / "merged_cycles"
OUTPUT_DIR = BASE_DIR / "split files"

DATASETS = {
    10: MERGED_DIR / "merged_cycles_10.csv",
    30: MERGED_DIR / "merged_cycles_30.csv",
    100: MERGED_DIR / "merged_cycles_100.csv",
}

SPLITS = ["train", "test", "val"]

# ============================================================
# 1. LOAD CELL SPLIT FILE
# ============================================================

if not SPLIT_FILE.exists():
    raise FileNotFoundError(
        f"Cell split file not found: {SPLIT_FILE}"
    )

split_df = pd.read_csv(SPLIT_FILE, dtype=str)

required_split_columns = {"cell", "part"}

if not required_split_columns.issubset(split_df.columns):
    raise ValueError(
        "I3_cell_split.csv must contain 'cell' and 'part' columns."
    )

# Normalize cell identifiers and split names.
split_df["cell"] = (
    split_df["cell"]
    .fillna("")
    .str.strip()
    .str.upper()
)

split_df["part"] = (
    split_df["part"]
    .fillna("")
    .str.strip()
    .str.lower()
)

# Remove rows without usable identifiers or split assignments.
split_df = split_df[
    (split_df["cell"] != "")
    & (split_df["part"] != "")
].copy()

# Reject invalid split names.
invalid_parts = sorted(
    set(split_df["part"]) - set(SPLITS)
)

if invalid_parts:
    raise ValueError(
        f"Invalid split labels found: {invalid_parts}. "
        f"Expected only {SPLITS}."
    )

# Ensure each cell belongs to exactly one split.
split_counts = split_df.groupby("cell")["part"].nunique()

conflicting_cells = split_counts[split_counts > 1].index.tolist()

if conflicting_cells:
    raise ValueError(
        "Some cells belong to multiple splits. Resolve these "
        f"conflicts in I3_cell_split.csv first: {conflicting_cells}"
    )

# Duplicate rows with the same assignment are harmless.
split_df = split_df.drop_duplicates(subset="cell")

# Build barcode-to-split mapping.
cell_to_split = split_df.set_index("cell")["part"].to_dict()

print("=" * 65)
print("CELL SPLIT SUMMARY")
print("=" * 65)

for split_name in SPLITS:
    count = (split_df["part"] == split_name).sum()
    print(f"{split_name.upper():<10}: {count} cells")

print(f"{'TOTAL':<10}: {len(split_df)} cells")

# ============================================================
# 2. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nOutput directory: {OUTPUT_DIR.resolve()}")

# ============================================================
# 3. PROCESS EACH CYCLE DATASET
# ============================================================

overall_unmatched = set()

for cycle_count, input_file in DATASETS.items():

    print("\n" + "=" * 65)
    print(f"PROCESSING {cycle_count}-CYCLE DATASET")
    print("=" * 65)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_file}"
        )

    # Read identifiers as strings to preserve barcodes.
    df = pd.read_csv(input_file, dtype=str)

    if "barcode" not in df.columns:
        raise ValueError(
            f"'barcode' column not found in {input_file}"
        )

    original_rows = len(df)

    # Normalize barcodes for matching.
    normalized_barcode = (
        df["barcode"]
        .fillna("")
        .str.strip()
        .str.upper()
    )

    # Assign each row to the split belonging to its cell.
    df["_normalized_barcode"] = normalized_barcode
    df["_split"] = normalized_barcode.map(cell_to_split)

    # Identify non-empty barcodes missing from the split file.
    unmatched_mask = (
        df["_normalized_barcode"].ne("")
        & df["_split"].isna()
    )

    unmatched_barcodes = set(
        df.loc[unmatched_mask, "_normalized_barcode"].unique()
    )

    overall_unmatched.update(unmatched_barcodes)

    if unmatched_barcodes:
        print(
            f"WARNING: {len(unmatched_barcodes)} distinct "
            "barcodes are absent from the cell split file."
        )
        print("Examples:", sorted(unmatched_barcodes)[:10])

    # Report empty barcodes separately.
    empty_barcode_rows = df["_normalized_barcode"].eq("").sum()

    if empty_barcode_rows:
        print(
            f"WARNING: {empty_barcode_rows} rows have empty barcodes."
        )

    # Report cell counts and create each split.
    assigned_rows = 0

    for split_name in SPLITS:

        split_data = df.loc[
            df["_split"] == split_name
        ].copy()

        # Remove helper columns from the output.
        split_data.drop(
            columns=["_normalized_barcode", "_split"],
            inplace=True,
        )

        output_file = (
            OUTPUT_DIR
            / f"cycles_{cycle_count}_{split_name}.csv"
        )

        split_data.to_csv(output_file, index=False)

        assigned_rows += len(split_data)

        unique_cells = (
            split_data["barcode"]
            .fillna("")
            .str.strip()
            .str.upper()
            .replace("", pd.NA)
            .nunique()
        )

        print(
            f"{output_file.name:<28} "
            f"Rows: {len(split_data):>6} | "
            f"Cells: {unique_cells:>4}"
        )

    # Verify that every matched row was assigned exactly once.
    if assigned_rows != df["_split"].notna().sum():
        raise RuntimeError(
            f"Row assignment verification failed for {input_file}"
        )

    print(f"Input rows:              {original_rows}")
    print(f"Rows assigned to splits: {assigned_rows}")
    print(
        f"Rows not assigned:       "
        f"{original_rows - assigned_rows}"
    )

# ============================================================
# 4. FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("SPLITTING COMPLETED")
print("=" * 65)

print(f"Output folder: {OUTPUT_DIR.resolve()}")
print("Expected output files: 9")

if overall_unmatched:
    print(
        f"\nWARNING: {len(overall_unmatched)} distinct barcodes "
        "could not be matched to I3_cell_split.csv."
    )
    print(
        "These rows were excluded from train/test/val outputs. "
        "Review the barcode mismatches before training."
    )
else:
    print("All non-empty barcodes matched the cell split file.")

print("\nThe original merged CSV files were not modified.")