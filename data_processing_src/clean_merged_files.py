from pathlib import Path
import pandas as pd
import shutil

# Folder containing the merged CSV files
DATA_DIR = Path("merged_cycles")

FILES = [
    DATA_DIR / "merged_cycles_10.csv",
    DATA_DIR / "merged_cycles_30.csv",
    DATA_DIR / "merged_cycles_100.csv",
]

# Remove these barcodes only from the 2017-05-12 batch
REMOVE_BARCODES = {
    "el150800460486",
    "el150800460514",
    "el150800464977",
    "el150800460623",
    "el150800464865",
}

# Remove these channels only from their specified batch dates
REMOVE_CHANNELS_BY_DATE = {
    "2017-05-12": {13, 19, 21, 22, 31},
    "2018-04-12": {33, 41},
}

CREATE_BACKUPS = True


def main():
    # Validate all files before modifying any of them
    dataframes = {}

    for file_path in FILES:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path, dtype=str)

        required_columns = {"barcode", "batch_date", "channel_id"}
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{file_path.name} is missing columns: {sorted(missing)}"
            )

        dataframes[file_path] = df

    total_removed = 0

    for file_path, df in dataframes.items():
        original_rows = len(df)

        # Normalize fields for reliable matching
        dates = df["batch_date"].str.strip()
        barcodes = df["barcode"].str.strip().str.lower()

        channels = pd.to_numeric(
            df["channel_id"].str.strip(),
            errors="coerce"
        )

        # Rule 1: Remove specified barcodes only from 2017-05-12
        remove_by_barcode = (
            dates.eq("2017-06-30")
            & barcodes.isin(REMOVE_BARCODES)
        )

        # Rule 2: Remove specified channels from their respective batches
        remove_by_channel = pd.Series(False, index=df.index)

        for batch_date, channel_ids in REMOVE_CHANNELS_BY_DATE.items():
            remove_by_channel |= (
                dates.eq(batch_date)
                & channels.isin(channel_ids)
            )

        # Combine both exclusion rules
        remove_mask = remove_by_barcode | remove_by_channel

        removed_rows = int(remove_mask.sum())
        cleaned_df = df.loc[~remove_mask].copy()

        # Back up the original file before overwriting it
        if CREATE_BACKUPS:
            backup_path = file_path.with_suffix(
                file_path.suffix + ".bak"
            )

            if not backup_path.exists():
                shutil.copy2(file_path, backup_path)

        # Save cleaned data in place
        cleaned_df.to_csv(file_path, index=False)

        total_removed += removed_rows

        print(f"\nFile: {file_path.name}")
        print(f"Original rows: {original_rows:,}")
        print(f"Removed rows: {removed_rows:,}")
        print(f"Remaining rows: {len(cleaned_df):,}")

        # Verify barcode exclusions
        remaining_barcodes = (
            cleaned_df["batch_date"].str.strip().eq("2017-06-30")
            & cleaned_df["barcode"].str.strip().str.lower().isin(
                REMOVE_BARCODES
            )
        ).sum()

        print(
            "Excluded barcodes remaining in 2017-06-30:",
            int(remaining_barcodes)
        )

        # Verify channel exclusions
        remaining_dates = cleaned_df["batch_date"].str.strip()
        remaining_channels = pd.to_numeric(
            cleaned_df["channel_id"].str.strip(),
            errors="coerce"
        )

        for batch_date, channel_ids in REMOVE_CHANNELS_BY_DATE.items():
            remaining = (
                remaining_dates.eq(batch_date)
                & remaining_channels.isin(channel_ids)
            ).sum()

            print(
                f"Excluded channels remaining in {batch_date}:",
                int(remaining)
            )

            if remaining:
                raise RuntimeError(
                    f"Channel exclusion verification failed for "
                    f"{file_path.name}, batch {batch_date}"
                )

        if remaining_barcodes:
            raise RuntimeError(
                f"Barcode exclusion verification failed for {file_path.name}"
            )

    print("\n========== CLEANING COMPLETE ==========")
    print(f"Total rows removed: {total_removed:,}")
    print("All three merged files have been processed.")


if __name__ == "__main__":
    main()