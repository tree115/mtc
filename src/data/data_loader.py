"""
DATA LOADER – FINAL SPLIT-AWARE VERSION

✔ Compatible 100% with dataset.py (split-aware)
✔ Supports fast per-split loading
✔ Keeps backward compatibility with load_all_data
✔ Designed for MALLORN Kaggle structure
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from config import RAW_DATA_DIR, TRAIN_LOG, TEST_LOG, NUM_SPLITS


class DataLoader:
    """
    Load lightcurve data and metadata for MALLORN dataset
    """

    def __init__(self, base_path=None):
        self.base_path = base_path or RAW_DATA_DIR

    # ======================================================
    # METADATA
    # ======================================================
    def load_metadata(self, mode="train"):
        """
        Load train_log.csv or test_log.csv
        """
        file_path = TRAIN_LOG if mode == "train" else TEST_LOG

        if not file_path.exists():
            raise FileNotFoundError(f"❌ File not found: {file_path}")

        df = pd.read_csv(file_path)
        print(f"✅ Loaded {mode} metadata: {len(df)} objects")

        return df

    # ======================================================
    # SPLIT HELPERS (CRITICAL)
    # ======================================================
    def get_splits(self, mode="train"):
        """
        Return list of available splits, e.g.:
        ["split_01", "split_02", ..., "split_20"]
        """
        splits = []
        for i in range(1, NUM_SPLITS + 1):
            sp = f"split_{i:02d}"
            split_dir = self.base_path / sp
            if split_dir.exists():
                splits.append(sp)

        if not splits:
            raise RuntimeError("❌ No split directories found")

        return splits

    def load_split_data(self, split, mode="train", sample_frac=1.0):
        """
        Load lightcurves + metadata for ONE split only
        This is the FAST PATH used by dataset.py
        """
        split_path = self.base_path / split / f"{mode}_full_lightcurves.csv"

        if not split_path.exists():
            print(f"⚠️ Missing file: {split_path}")
            return pd.DataFrame()

        # --------------------------------------------------
        # Load lightcurves
        # --------------------------------------------------
        try:
            lc = pd.read_csv(split_path)
        except Exception as e:
            print(f"❌ Failed to load {split_path}: {e}")
            return pd.DataFrame()

        # Standardize column names
        lc = lc.rename(columns={
            "Time (MJD)": "mjd",
            "Flux": "flux",
            "Flux_err": "flux_err",
        })

        if "object_id" not in lc.columns:
            raise ValueError(f"❌ object_id missing in {split_path}")

        # --------------------------------------------------
        # Load metadata and filter by split
        # --------------------------------------------------
        meta = self.load_metadata(mode)
        meta = meta[meta["split"] == split]

        if sample_frac < 1.0:
            meta = meta.sample(frac=sample_frac, random_state=42)

        object_ids = set(meta["object_id"].values)

        # Filter lightcurves
        lc = lc[lc["object_id"].isin(object_ids)]

        if lc.empty:
            return pd.DataFrame()

        # Merge metadata
        data = lc.merge(meta, on="object_id", how="left")

        return data

    # ======================================================
    # BACKWARD COMPATIBILITY (SLOW, NOT USED ANYMORE)
    # ======================================================
    def load_lightcurves(self, object_ids, mode="train"):
        """
        Load lightcurves for a list of object_ids
        (legacy method – slower)
        """
        all_lightcurves = []

        metadata = self.load_metadata(mode)
        object_to_split = metadata.set_index("object_id")["split"].to_dict()

        split_to_objects = {}
        for oid in object_ids:
            if oid in object_to_split:
                split_to_objects.setdefault(object_to_split[oid], []).append(oid)

        for split, obj_list in tqdm(split_to_objects.items(),
                                    desc=f"Loading {mode} lightcurves"):
            split_path = self.base_path / split / f"{mode}_full_lightcurves.csv"

            if not split_path.exists():
                continue

            lc = pd.read_csv(split_path)
            lc = lc.rename(columns={
                "Time (MJD)": "mjd",
                "Flux": "flux",
                "Flux_err": "flux_err",
            })

            lc = lc[lc["object_id"].isin(obj_list)]
            if not lc.empty:
                all_lightcurves.append(lc)

        if not all_lightcurves:
            return pd.DataFrame()

        return pd.concat(all_lightcurves, ignore_index=True)

    def load_all_data(self, mode="train", sample_frac=1.0):
        """
        Load ALL splits together (legacy – slow)
        """
        print(f"📥 Loading ALL data ({mode}) – SLOW PATH")

        meta = self.load_metadata(mode)

        if sample_frac < 1.0:
            meta = meta.sample(frac=sample_frac, random_state=42)

        lc = self.load_lightcurves(meta["object_id"].tolist(), mode)

        if lc.empty:
            raise RuntimeError("❌ No lightcurves loaded")

        data = lc.merge(meta, on="object_id", how="left")
        return data
