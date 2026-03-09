"""
preprocess.py
-------------
Memory-efficient preprocessing pipeline for an ML Intrusion Detection System.

Improvements over v1
--------------------
1. STREAMING WRITES — chunks are written to Parquet immediately via
   pyarrow.ParquetWriter; the full dataset is never held in RAM at once.
2. SCHEMA LOCK-IN — the PyArrow schema is inferred from the first valid chunk
   and enforced on every subsequent chunk to prevent schema-mismatch errors.
3. DUAL LABELS — both label columns are preserved side-by-side:
     * label_binary  (int8)  — 0 = BENIGN, 1 = ATTACK  (binary classification)
     * attack_type   (str)   — original label string    (multi-class classification)
4. DTYPE OPTIMISATION — float64 → float32 and int64 → int32; roughly halves
   RAM and on-disk size for large numeric datasets.
5. ROBUST SCHEMA HANDLING — unexpected extra columns are dropped; columns
   missing from a particular file are inserted as NaN and subsequently
   removed by dropna, so a single bad file cannot abort the whole pipeline.
"""

import os
import glob
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Suppress pandas 4.x FutureWarnings (select_dtypes object/str deprecation)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ---------------------------------------------------------------------------
# Constants  (tune CHUNK_SIZE to the available RAM of the target machine)
# ---------------------------------------------------------------------------
RAW_CSV_DIR  = "raw_csv"
OUTPUT_DIR   = "processed"
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "cleaned.parquet")
CHUNK_SIZE   = 100_000        # rows per chunk
DROP_COLS    = {"flow_id", "source_ip", "destination_ip", "timestamp"}
BENIGN_LABEL = "BENIGN"


# ---------------------------------------------------------------------------
# Column / name helpers
# ---------------------------------------------------------------------------

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, lower-case, and replace runs of spaces → underscores."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def downcast_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory footprint by downcasting wide numeric types:
      float64 → float32  (sufficient precision for network-flow features)
      int64   → int32    (flow counters rarely exceed 2^31)

    Applied *after* cleaning so that NaN-free columns cast without issues.
    """
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include="int64").columns:
        df[col] = df[col].astype(np.int32)
    return df


def coerce_numeric_columns(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce object/string columns to numeric where possible.
    'attack_type' is intentionally excluded — it holds human-readable strings.
    Coercion errors produce NaN, which are removed by the subsequent dropna.
    """
    for col in chunk.select_dtypes(include=["object", "str"]).columns:
        if col == "attack_type":   # preserve raw label string
            continue
        try:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        except Exception:
            pass
    return chunk


def align_to_schema(
    chunk: pd.DataFrame,
    reference_cols: list[str],
    reference_dtypes: dict[str, np.dtype],
) -> pd.DataFrame:
    """
    Align a chunk to the reference column set established by the first chunk.

    Handles two kinds of schema inconsistency that are common in multi-day
    CICIDS exports:
      • Extra columns  — silently dropped (unexpected additions)
      • Missing columns — inserted as NaN; the subsequent dropna removes
                          affected rows, so no invalid data reaches the output

    Finally, columns are reordered and cast to the reference dtypes so that
    PyArrow's ParquetWriter receives a schema-compatible table every time.
    """
    # Drop columns not present in the reference schema
    extra = set(chunk.columns) - set(reference_cols)
    if extra:
        log.debug("    Dropping %d unexpected column(s): %s", len(extra), sorted(extra))
        chunk.drop(columns=list(extra), inplace=True)

    # Insert columns present in reference but absent in this chunk
    missing = set(reference_cols) - set(chunk.columns)
    if missing:
        log.debug("    Inserting %d missing column(s) as NaN: %s", len(missing), sorted(missing))
        for col in missing:
            chunk[col] = np.nan

    # Reorder to exactly match the reference and coerce dtypes
    chunk = chunk[reference_cols].copy()
    for col, dtype in reference_dtypes.items():
        try:
            chunk[col] = chunk[col].astype(dtype)
        except (ValueError, TypeError):
            # Leave as-is; rows that can't be cast will be caught by dropna
            pass

    return chunk


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def encode_labels(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Split the raw 'label' column into two derived columns and drop the original:

      label_binary (int8)
        0 = BENIGN  — used as the target for binary classifiers
        1 = ATTACK

      attack_type (object/str)
        Original label string retained for multi-class classification
        e.g. 'DoS Hulk', 'Bot', 'FTP-Patator', …

    Encoding is done before numeric coercion so that the original strings
    are preserved intact.
    """
    raw = chunk["label"].astype(str).str.strip()

    # Binary target — use int8 (smallest signed integer) to save space
    chunk["label_binary"] = (raw.str.upper() != BENIGN_LABEL).astype(np.int8)

    # Multi-class target — plain string, kept as-is
    chunk["attack_type"] = raw

    chunk.drop(columns=["label"], inplace=True)
    return chunk


# ---------------------------------------------------------------------------
# Per-chunk cleaning
# ---------------------------------------------------------------------------

def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline applied to a single chunk in order:

      1. Encode labels  → produces label_binary + attack_type, drops 'label'
      2. Replace ±inf   → NaN  (common in computed CICIDS features)
      3. Coerce object columns to numeric (non-numeric strings become NaN)
      4. Drop rows with any NaN value
      5. Drop duplicate rows within the chunk
      6. Downcast numerics (float64→float32, int64→int32)

    Raises KeyError if the 'label' column is absent so the caller can log a
    meaningful warning and skip the offending chunk.
    """
    if "label" not in chunk.columns:
        raise KeyError(
            "'label' column not found. "
            f"Available columns: {chunk.columns.tolist()}"
        )

    # 1 – label encoding (before numeric coercion to preserve string labels)
    chunk = encode_labels(chunk)

    # 2 – replace infinities with NaN on numeric columns only
    numeric_cols = chunk.select_dtypes(include=np.number).columns
    chunk[numeric_cols] = chunk[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # 3 – coerce remaining object/str columns to numeric
    chunk = coerce_numeric_columns(chunk)

    # 4 – remove rows that contain any NaN (covers original NaN, former infs,
    #     and any values that failed numeric coercion)
    chunk.dropna(inplace=True)

    # 5 – within-chunk deduplication (see cross-file dedup note in run_pipeline)
    chunk.drop_duplicates(inplace=True)

    # 6 – downcast to smaller numeric types
    chunk = downcast_numerics(chunk)

    return chunk


# ---------------------------------------------------------------------------
# Streaming pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """
    Main streaming pipeline.

    Architecture
    ~~~~~~~~~~~~
    For each CSV file, for each chunk:
      1. Standardise column names.
      2. Drop ML-irrelevant columns (flow_id, IPs, timestamp).
      3. Clean the chunk (label encoding, NaN/inf removal, dedup, downcast).
      4. On the first valid chunk: lock in the PyArrow schema and open the
         ParquetWriter.  On subsequent chunks: align to that schema.
      5. Write the cleaned PyArrow table directly to the Parquet file.
      6. Update lightweight counters for final statistics.

    Memory footprint
    ~~~~~~~~~~~~~~~~
    At any given moment only one chunk (~100k rows) resides in memory.
    Row-count statistics are tracked with plain integers rather than by
    re-reading the output file.

    Cross-file deduplication note
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    True global deduplication requires holding all rows in memory, which
    defeats the purpose of streaming.  Within-chunk dedup is applied instead.
    If strict global dedup is required as a post-processing step, the output
    Parquet can be re-read in a separate pass.
    """
    csv_files = sorted(glob.glob(os.path.join(RAW_CSV_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{RAW_CSV_DIR}/'")

    log.info("Found %d CSV file(s) to process.", len(csv_files))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Schema state — populated lazily from the first valid clean chunk.
    # ------------------------------------------------------------------
    reference_cols:   Optional[list[str]]           = None
    reference_dtypes: Optional[dict[str, np.dtype]] = None
    parquet_schema:   Optional[pa.Schema]           = None
    writer:           Optional[pq.ParquetWriter]    = None

    # Lightweight running counters (no DataFrames stored between chunks)
    total_raw_rows   = 0
    total_clean_rows = 0
    total_benign     = 0
    total_attack     = 0

    try:
        for file_path in csv_files:
            log.info("  Reading: %s", file_path)
            file_clean_rows = 0

            # Open the CSV reader; skip the whole file if it's unreadable
            try:
                reader = pd.read_csv(
                    file_path,
                    chunksize=CHUNK_SIZE,
                    low_memory=False,
                )
            except Exception as exc:
                log.error("Cannot open '%s': %s — skipping file.", file_path, exc)
                continue

            for chunk_idx, chunk in enumerate(reader, start=1):
                total_raw_rows += len(chunk)

                # ---- Per-chunk preprocessing -------------------------
                try:
                    chunk = standardise_columns(chunk)

                    # Drop columns not useful for ML
                    to_drop = DROP_COLS.intersection(chunk.columns)
                    if to_drop:
                        chunk.drop(columns=list(to_drop), inplace=True)

                    chunk = clean_chunk(chunk)

                except KeyError as exc:
                    # Missing 'label' column — log and skip this chunk only
                    log.warning(
                        "    Skipping chunk %d of '%s': %s",
                        chunk_idx, file_path, exc,
                    )
                    continue
                except Exception as exc:
                    log.error(
                        "    Unexpected error in chunk %d of '%s': %s — skipping.",
                        chunk_idx, file_path, exc,
                    )
                    continue

                if chunk.empty:
                    log.debug(
                        "    Chunk %d is empty after cleaning — skipping.", chunk_idx
                    )
                    continue

                # ---- Schema management & Parquet writing --------------
                if reference_cols is None:
                    # ── First valid chunk: lock in the reference schema ──
                    reference_cols   = list(chunk.columns)
                    reference_dtypes = {col: chunk[col].dtype for col in reference_cols}

                    # Build the canonical PyArrow schema without pandas index
                    # metadata so the writer produces a clean, portable file.
                    first_table    = pa.Table.from_pandas(chunk, preserve_index=False)
                    parquet_schema = first_table.schema

                    writer = pq.ParquetWriter(
                        OUTPUT_FILE, parquet_schema, compression="snappy"
                    )
                    log.info(
                        "    Schema locked from first chunk: %d columns "
                        "(%d features + label_binary + attack_type).",
                        len(reference_cols),
                        len(reference_cols) - 2,  # subtract the two label cols
                    )
                    # Stream the first chunk immediately
                    writer.write_table(first_table)

                else:
                    # ── Subsequent chunks: align to the locked schema ──
                    chunk = align_to_schema(chunk, reference_cols, reference_dtypes)

                    # align_to_schema may have inserted NaN for missing cols;
                    # remove those rows before writing.
                    chunk.dropna(inplace=True)

                    if chunk.empty:
                        continue

                    table = pa.Table.from_pandas(
                        chunk, schema=parquet_schema, preserve_index=False
                    )
                    writer.write_table(table)

                # ---- Update running counters (no DataFrames kept) ----
                n_clean  = len(chunk)
                n_benign = int((chunk["label_binary"] == 0).sum())
                n_attack = int((chunk["label_binary"] == 1).sum())

                file_clean_rows  += n_clean
                total_clean_rows += n_clean
                total_benign     += n_benign
                total_attack     += n_attack

                log.debug(
                    "    Chunk %d: %d clean rows (%d benign, %d attack).",
                    chunk_idx, n_clean, n_benign, n_attack,
                )

            log.info("    → %d clean rows from this file.", file_clean_rows)

    finally:
        # Always close the writer (even on exception) so the Parquet footer
        # is flushed and the file remains readable up to the last row group.
        if writer is not None:
            writer.close()
            log.info("ParquetWriter closed and footer flushed.")

    if total_clean_rows == 0:
        raise RuntimeError("No data remained after cleaning. Check your raw CSV files.")

    # ---------------------------------------------------------------------------
    # Final statistics — derived from counters, no file re-read required
    # ---------------------------------------------------------------------------
    pct_attack = total_attack / total_clean_rows * 100 if total_clean_rows else 0.0

    log.info("Raw rows read (all files)  : %d", total_raw_rows)
    log.info("=" * 50)
    log.info("DATASET STATISTICS")
    log.info("  Total clean rows : %d", total_clean_rows)
    log.info("  Benign flows     : %d  (%.2f%%)", total_benign, 100.0 - pct_attack)
    log.info("  Attack flows     : %d  (%.2f%%)", total_attack, pct_attack)
    if reference_cols is not None:
        log.info(
            "  Features         : %d  (excl. label_binary + attack_type)",
            len(reference_cols) - 2,
        )
    log.info("=" * 50)

    size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    log.info("Output → %s  (%.2f MB)", OUTPUT_FILE, size_mb)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=== IDS Preprocessing Pipeline START ===")
    run_pipeline()
    log.info("=== IDS Preprocessing Pipeline COMPLETE ===")
