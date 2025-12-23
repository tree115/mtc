"""
TDE DATASET BUILDER – FINAL SPLIT-AWARE VERSION WITH PROGRESS BAR

- Logs chỉ hiển thị 1 lần cho thông tin quan trọng
- Thanh tiến trình % hiển thị clean
- Không thay đổi logic feature_engineer
- Merge metadata an toàn, tránh cột _y rỗng
- Convert abs_mag_bin thành numeric để tránh lỗi Non-numeric columns
"""

import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

from config import TRAIN_FEATURES, TEST_FEATURES
from data_loader import DataLoader
from preprocessor import Preprocessor
from feature_engineer import UltimateFeatureEngineer


class TdeDataset:
    def __init__(self, sample_frac=1.0, random_state=42):
        self.sample_frac = sample_frac
        self.random_state = random_state

        self.loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.feature_engineer = UltimateFeatureEngineer()

    # ======================================================
    # CORE: BUILD FEATURES BY SPLIT (WITH PROGRESS BAR)
    # ======================================================
    def _build_features_by_split(self, mode="train"):
        print("=" * 80)
        print(f"⚙️  FEATURE ENGINEERING BY SPLIT ({mode.upper()})")
        print("=" * 80)

        all_features = []
        splits = self.loader.get_splits(mode)

        for sp in splits:
            print(f"\n🔹 Processing {sp}")

            # 1. Load split
            data = self.loader.load_split_data(
                split=sp,
                mode=mode,
                sample_frac=self.sample_frac,
            )

            if data is None or len(data) == 0:
                print("   ⚠️ Empty split – skipped")
                continue

            # 2. Preprocess
            data = self.preprocessor.preprocess_pipeline(data)

            # 3. Feature engineering per object with tqdm
            objects = data['object_id'].unique()
            features_list = []

            print(f"   ⏳ Extracting features for {len(objects)} objects...")
            for obj_id in tqdm(objects, desc=f"Processing {sp}", ncols=100, leave=True):
                obj_data = data[data['object_id'] == obj_id]
                feat = self.feature_engineer.extract_all_features(obj_data)
                if not feat.empty:
                    feat["object_id"] = obj_id
                    features_list.append(feat)

            if not features_list:
                print("   ⚠️ No features extracted – skipped")
                continue

            feat = pd.concat(features_list, ignore_index=True)
            feat["split"] = sp
            all_features.append(feat)

            print(f"   ✅ Objects processed: {len(feat)}")

        if not all_features:
            raise RuntimeError("❌ No features extracted from any split")

        features = pd.concat(all_features, ignore_index=True)
        print(f"\n✅ TOTAL OBJECTS: {features['object_id'].nunique()}")

        return features

    # ======================================================
    # TRAIN DATASET
    # ======================================================
    def create_train_dataset(self, save=True):
        print("=" * 80)
        print("🏗️  BUILD TRAIN DATASET (SPLIT-AWARE FINAL)")
        print("=" * 80)

        features = self._build_features_by_split(mode="train")

        metadata = self.loader.load_metadata("train")
        if self.sample_frac < 1.0:
            metadata = metadata.sample(
                frac=self.sample_frac,
                random_state=self.random_state
            )

        meta_cols = ["object_id", "target", "split", "Z", "EBV"]
        meta_cols = [c for c in meta_cols if c in metadata.columns]
        metadata = metadata[meta_cols].drop_duplicates("object_id")

        dataset = features.merge(
            metadata,
            on=["object_id", "split"],
            how="left"
        )

        # Xử lý các cột rỗng, đặc biệt là Z, EBV, split
        dataset['Z'] = pd.to_numeric(dataset.get('Z', pd.Series(0.0)), errors='coerce').fillna(0.0)
        dataset['EBV'] = pd.to_numeric(dataset.get('EBV', pd.Series(0.0)), errors='coerce').fillna(0.0)
        if 'split' in dataset.columns:
            dataset['split'] = dataset['split'].fillna('missing')

        dataset = self._handle_missing(dataset)
        self._print_stats(dataset, is_test=False)

        if save:
            TRAIN_FEATURES.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_csv(TRAIN_FEATURES, index=False)
            print(f"💾 Saved train dataset → {TRAIN_FEATURES}")

        return dataset

    # ======================================================
    # TEST DATASET
    # ======================================================
    def create_test_dataset(self, save=True):
        print("=" * 80)
        print("🧪 BUILD TEST DATASET (SPLIT-AWARE FINAL)")
        print("=" * 80)

        features = self._build_features_by_split(mode="test")

        metadata = self.loader.load_metadata("test")
        if self.sample_frac < 1.0:
            metadata = metadata.sample(
                frac=self.sample_frac,
                random_state=self.random_state
            )

        meta_cols = ["object_id", "split", "Z", "EBV"]
        meta_cols = [c for c in meta_cols if c in metadata.columns]
        metadata = metadata[meta_cols].drop_duplicates("object_id")

        dataset = features.merge(
            metadata,
            on=["object_id", "split"],
            how="left"
        )

        # Xử lý các cột rỗng
        dataset['Z'] = pd.to_numeric(dataset.get('Z', pd.Series(0.0)), errors='coerce').fillna(0.0)
        dataset['EBV'] = pd.to_numeric(dataset.get('EBV', pd.Series(0.0)), errors='coerce').fillna(0.0)
        if 'split' in dataset.columns:
            dataset['split'] = dataset['split'].fillna('missing')

        dataset = self._handle_missing(dataset)
        dataset = self._align_with_train(dataset)
        self._print_stats(dataset, is_test=True)

        if save:
            TEST_FEATURES.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_csv(TEST_FEATURES, index=False)
            print(f"💾 Saved test dataset → {TEST_FEATURES}")

        return dataset

    # ======================================================
    # HELPERS
    # ======================================================
    def _handle_missing(self, df):
        df = df.copy()

        # Numeric → median
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # Object → 'missing' và convert abs_mag_bin
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if col == "abs_mag_bin":
                df[col] = pd.factorize(df[col])[0]  # convert string → numeric
            else:
                df[col] = df[col].fillna("missing")

        return df

    def _align_with_train(self, test_df):
        if not TRAIN_FEATURES.exists():
            raise RuntimeError(
                "❌ train_features.csv not found. "
                "Run create_train_dataset() first."
            )

        train_cols = pd.read_csv(TRAIN_FEATURES, nrows=0).columns.tolist()
        feature_cols = [c for c in train_cols if c != "target"]

        for col in feature_cols:
            if col not in test_df.columns:
                test_df[col] = 0.0

        test_df = test_df[feature_cols]
        return test_df

    def _print_stats(self, df, is_test=False):
        print("\n📊 DATASET STATS")
        print(f"Shape   : {df.shape}")
        print(f"Objects : {df['object_id'].nunique()}")

        if not is_test and "target" in df.columns:
            vc = df["target"].value_counts()
            tde = vc.get(1, 0)
            non = vc.get(0, 0)
            print(f"TDE     : {tde}")
            print(f"Non-TDE : {non}")
            print(f"TDE ratio: {tde / len(df):.4f}")


# ======================================================
# IMPORT HELPERS
# ======================================================
def create_train_dataset(sample_frac=1.0, save=True):
    builder = TdeDataset(sample_frac=sample_frac)
    return builder.create_train_dataset(save=save)


def create_test_dataset(sample_frac=1.0, save=True):
    builder = TdeDataset(sample_frac=sample_frac)
    return builder.create_test_dataset(save=save)
