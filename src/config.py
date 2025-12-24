"""
Cấu hình đường dẫn và hằng số cho dự án TDE Mallorn
"""

from pathlib import Path
import os

# =====================================================
# 1️⃣ PROJECT ROOT (CODE)
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# =====================================================
# 2️⃣ DATA ROOT (CÓ THỂ NẰM NGOÀI PROJECT)
# 👉 CHỈ CẦN SỬA DÒNG NÀY
# =====================================================
DATA_ROOT = Path(
    os.environ.get(
        "TDE_DATA_ROOT",          # cho phép set bằng biến môi trường
        "/content/data"           # 👈 DÁN ĐƯỜNG DẪN DATA Ở ĐÂY
    )
)

# =====================================================
# 3️⃣ DATA DIRECTORIES
# =====================================================
RAW_DATA_DIR = DATA_ROOT 
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
CACHE_DIR = DATA_ROOT / "cache"

# =====================================================
# 4️⃣ OUTPUT (LUÔN Ở PROJECT)
# =====================================================
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
LOGS_DIR = OUTPUTS_DIR / "logs"

# =====================================================
# 5️⃣ CREATE DIRS
# =====================================================
for d in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CACHE_DIR,
    MODELS_DIR,
    SUBMISSIONS_DIR,
    LOGS_DIR
]:
    d.mkdir(parents=True, exist_ok=True)

# =====================================================
# 6️⃣ FILE PATHS
# =====================================================
# Raw data files
TRAIN_LOG = RAW_DATA_DIR / "train_log.csv"
TEST_LOG = RAW_DATA_DIR / "test_log.csv"

# Processed feature files
TRAIN_FEATURES = PROCESSED_DATA_DIR / "all_train_features.csv"
TEST_FEATURES = PROCESSED_DATA_DIR / "all_test_features.csv"

# Model and submission files
SUBMISSION_FILE = SUBMISSIONS_DIR / "submission.csv"
MODEL_FILE = MODELS_DIR / "best_model.pkl"

# =====================================================
# 7️⃣ PHYSICAL CONSTANTS
# =====================================================
EFF_WAVELENGTHS = {
    'u': 3641,
    'g': 4704,
    'r': 6155,
    'i': 7504,
    'z': 8695,
    'y': 10056
}

R_V = 3.1
FILTERS = ['u', 'g', 'r', 'i', 'z', 'y']
NUM_SPLITS = 20

# =====================================================
# 8️⃣ MODEL CONFIG
# =====================================================
SEED = 42
CV_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50

# =====================================================
# 9️⃣ FEATURE CONFIG
# =====================================================
# Các features để loại bỏ khỏi training
DROP_COLS = [
    "object_id",
    "split",
    "Z",
    "EBV",
    "SpecType",
    "Z_err",
    "English Translation",
    "target"  # target chỉ loại bỏ khi training
]

# =====================================================
# 🔟 LOG
# =====================================================
print("✅ TDE Mallorn config loaded")
print(f"📁 PROJECT_ROOT : {PROJECT_ROOT}")
print(f"📁 DATA_ROOT    : {DATA_ROOT}")
print(f"📁 OUTPUTS_DIR  : {OUTPUTS_DIR}")