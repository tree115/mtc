# MTC - Pipeline Phân loại TDE

Repo này chứa pipeline máy học để phân loại **Tidal Disruption Events (TDEs)** từ dữ liệu lightcurve thiên văn, sử dụng kỹ thuật **feature engineering** nâng cao và mô hình **LightGBM**.

---

## 🗂 Cấu trúc thư mục

- `config.py` : File cấu hình cho đường dẫn, features và thư mục lưu model.
- `data_loader.py` : Load dữ liệu lightcurve và log thô.
- `dataset.py` : Wrapper dataset và các hàm tiền xử lý dữ liệu.
- `feature_engineer.py` : Các hàm trích xuất đặc trưng tùy chỉnh.
- `gp_features.py` : Trích xuất đặc trưng từ Gaussian Process cho lightcurve.
- `model.py` : Huấn luyện model, cross-validation, ensemble với LightGBM.
- `preprocessor.py` : Tiền xử lý dữ liệu, xử lý NaN, mã hóa categorical.
- `train.py` : Script huấn luyện với nhiều chiến lược và tối ưu ngưỡng phân loại.
- `predict.py` : Script dự đoán cho dữ liệu mới.

---

## ⚡ Tính năng

- Pipeline LightGBM đầy đủ với **cross-validation**, **feature selection**, và **ensemble**.
- Tối ưu cho **dữ liệu mất cân bằng**, tự động điều chỉnh ngưỡng dựa trên **F1-score** và **Precision-Recall curve**.
- Tự động tính toán và trực quan hóa **feature importance**.
- Hỗ trợ **OOF prediction** để đánh giá tổng thể.
- Hỗ trợ nhiều **chiến lược trích xuất đặc trưng**, bao gồm Gaussian Process (GP).

---

## 📦 Yêu cầu

- Python 3.9+
- pandas, numpy
- scikit-learn
- lightgbm
- matplotlib, seaborn
- joblib

Cài đặt dependencies:

```bash
pip install -r requirements.txt
