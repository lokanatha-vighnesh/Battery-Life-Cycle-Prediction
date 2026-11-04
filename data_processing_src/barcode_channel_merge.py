#```python
import shutil
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("merged_cycles")

MAPPING_FILE = Path("battery_channel_barcode_mapping.csv")

MERGED_FILES = [
    DATA_DIR / "merged_cycles_10.csv",
    DATA_DIR / "merged_cycles_30.csv",
    DATA_DIR / "merged_cycles_100.csv",
]

# --------------------------------------------------
# 1. Load the channel ID mapping
# --------------------------------------------------

if not MAPPING_FILE.exists():
    raise FileNotFoundError(
        f"Mapping file not found: {MAPPING_FILE.resolve()}"
    )

mapping_df = pd.read_csv(MAPPING_FILE, dtype=str)

required_mapping_columns = {"barcode", "channel_id"}

if not required_mapping_columns.issubset(mapping_df.columns):
    raise ValueError(
        f"Mapping file must contain: {required_mapping_columns}"
    )

# Normalize barcodes and channel IDs
mapping_df["barcode"] = (
    mapping_df["barcode"]
    .fillna("")
    .str.strip()
    .str.upper()
)

mapping_df["channel_id"] = (
    mapping_df["channel_id"]
    .fillna("")
    .str.strip()
)

# Remove rows without a usable barcode or channel ID
mapping_df = mapping_df[
    (mapping_df["barcode"] != "")
    & (mapping_df["channel_id"] != "")
].copy()

# --------------------------------------------------
# 2. Detect barcode ambiguity
# --------------------------------------------------

# Count distinct channel IDs for each barcode
channel_counts = mapping_df.groupby("barcode")["channel_id"].nunique()

ambiguous_barcodes = channel_counts[channel_counts > 1].index

if len(ambiguous_barcodes) > 0:
    print("\nWARNING: Some barcodes map to multiple channel IDs.")
    print("These barcodes will not be updated automatically.")

    ambiguous_rows = mapping_df[
        mapping_df["barcode"].isin(ambiguous_barcodes)
    ].sort_values("barcode")

    print(
        ambiguous_rows[
            ["barcode", "channel_id"]
        ].drop_duplicates().to_string(index=False)
    )

# Keep only barcodes with exactly one distinct channel ID
unambiguous_mapping = mapping_df[
    ~mapping_df["barcode"].isin(ambiguous_barcodes)
].copy()

# One row per barcode
unambiguous_mapping = (
    unambiguous_mapping[["barcode", "channel_id"]]
    .drop_duplicates()
)

barcode_to_channel = dict(
    zip(
        unambiguous_mapping["barcode"],
        unambiguous_mapping["channel_id"],
    )
)

print(f"\nUnique barcodes in mapping: {mapping_df['barcode'].nunique()}")
print(f"Unambiguous barcode mappings: {len(barcode_to_channel)}")
print(f"Ambiguous barcodes: {len(ambiguous_barcodes)}")

# --------------------------------------------------
# 3. Update all merged CSV files
# --------------------------------------------------

for file_path in MERGED_FILES:

    if not file_path.exists():
        print(f"\nFILE NOT FOUND: {file_path}")
        continue

    print(f"\nProcessing: {file_path}")

    df = pd.read_csv(file_path, dtype=str)

    if "barcode" not in df.columns:
        print("ERROR: barcode column not found.")
        continue

    if "channel_id" not in df.columns:
        print("ERROR: channel_id column not found.")
        continue

    # Create backup only if one does not already exist
    backup_path = file_path.with_suffix(
        file_path.suffix + ".bak"
    )

    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"Backup created: {backup_path}")
    else:
        print(f"Existing backup preserved: {backup_path}")

    # Normalize barcodes for matching
    normalized_barcodes = (
        df["barcode"]
        .fillna("")
        .str.strip()
        .str.upper()
    )

    # Map channel IDs using barcode alone
    mapped_channels = normalized_barcodes.map(barcode_to_channel)

    matched = mapped_channels.notna()
    unmatched = ~matched & (normalized_barcodes != "")

    # Update only rows with an unambiguous mapping
    df.loc[matched, "channel_id"] = mapped_channels[matched]

    # Save the updated CSV
    df.to_csv(file_path, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Rows with mapped channel IDs: {matched.sum()}")
    print(f"Rows with unmatched barcodes: {unmatched.sum()}")
    print(f"Rows with ambiguous barcodes: "
          f"{normalized_barcodes.isin(ambiguous_barcodes).sum()}")
    print(f"Updated file: {file_path}")

print("\nBarcode-based mapping completed.")
