#  Mallorn Astronomical Classification Challenge – Bài Tập Lớn Machine Learning

##  Giới thiệu
Đây là dự án bài tập lớn môn **Machine Learning** thực hiện tham gia cuộc thi [Mallorn Astronomical Classification Challenge](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/data) trên Kaggle.  
Mục tiêu của dự án là phát triển mô hình học máy để phân loại các sự kiện thiên văn, đặc biệt là **Tidal Disruption Events (TDEs)**, từ dữ liệu **lightcurve** thu thập bởi LSST.

Nhóm triển khai đầy đủ các bước:
- Tiền xử lý dữ liệu và xử lý missing values.
- Trích xuất đặc trưng quan trọng từ lightcurve.
- Huấn luyện mô hình LightGBM, thử các chiến lược feature selection và ensemble.
- Tối ưu ngưỡng phân loại (threshold) dựa trên F1-score và Precision–Recall Curve.
- Đánh giá hiệu năng bằng các metrics: F1-score, Precision, Recall, ROC AUC, AP.
- Tạo báo cáo OOF và trực quan hóa các kết quả (feature importance, confusion matrix, ROC/PR curves).
---

## 👥 Nhóm thực hiện

| Họ và tên | Mã sinh viên |
|-----------|--------------|
| Vũ Thị Kim Chi | 23021489 |
| Nguyễn Đoàn Hoài Thương | 23021733 |
| Nguyễn Thị Thanh Tuyền | 23021717 |
---


# MTC - Pipeline Phân loại TDE

Repo này chứa pipeline máy học để phân loại **Tidal Disruption Events (TDEs)** từ dữ liệu lightcurve thiên văn, sử dụng kỹ thuật **feature engineering** nâng cao và mô hình **LightGBM**.

---

## Cấu trúc thư mục
```
TDE-Mallorn-Detection/
├── README.md                         
├── data/                             # Thư mục dữ liệu
│   ├── raw/                          # Dữ liệu gốc từ Kaggle (giữ nguyên)
│   │   ├── split_01/                 # Các split chính thức của MALLORN
│   │   ├── split_02/
│   │   └── ...
│   └── processed/                    # Dữ liệu sau khi preprocessing & trích xuất đặc trưng
│
├── src/                              # Source code chính
│   ├── __init__.py                   # Đánh dấu src là một Python package
│   ├── config.py                     # Cấu hình chung & đường dẫn
│   │
│   ├── data/                         # Xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── data_loader.py            # Load dữ liệu raw/processed
│   │   ├── dataset.py                # Dataset dùng cho train/inference
│   │   └── preprocessor.py           # Preprocessing (extinction, cleaning, vật lý)
│   │
│   ├── features/                     # Trích xuất đặc trưng
│   │   ├── __init__.py
│   │   ├── feature_engineer.py       # Trích xuất đặc trưng light curve
│   │   ├── gp_features.py            # Đặc trưng từ Gaussian Process
│   │   └── utils.py                  # Hàm hỗ trợ feature engineering
│   │
│   ├── models/                       # Huấn luyện & dự đoán
│   │   ├── __init__.py
│   │   ├── model.py                  # Định nghĩa model (LightGBM)
│   │   ├── train.py                  # Train model cơ bản
│   │   ├── train_final.py            # Train cuối với Optuna
│   │   ├── predict.py                # Dự đoán & tạo file submission
│   │   └── optuna_tune.py            # Tối ưu hyperparameter
│   │
│   └── visualization/                # Vẽ hình & trực quan hóa
│       ├── __init__.py
│       └── shap_analysis.py          # Vẽ feature importance
│
├── notebooks/                        # Notebook phân tích
│   ├── run.ipynb                     # Chạy toàn bộ pipeline
│   ├── eda.ipynb                     # Khám phá và phân tích dữ liệu
│
├── outputs/                          # Kết quả sinh ra (không commit)
│   ├── models/                       # Model đã train
│   ├── submissions/                  # File nộp Kaggle
│   ├── logs/                         # Log huấn luyện
│   ├── feature_importance/           # Độ quan trọng đặc trưng
│   └── optuna_studies/               # Kết quả Optuna

```
---

## Tính năng

- Pipeline LightGBM đầy đủ với **cross-validation**, **feature selection**, và **ensemble**.
- Tối ưu cho **dữ liệu mất cân bằng**, tự động điều chỉnh ngưỡng dựa trên **F1-score** và **Precision-Recall curve**.
- Tự động tính toán và trực quan hóa **feature importance**.
- Hỗ trợ **OOF prediction** để đánh giá tổng thể.
- Hỗ trợ nhiều **chiến lược trích xuất đặc trưng**, bao gồm Gaussian Process (GP).

---

