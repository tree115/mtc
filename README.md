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

## Tính năng

- Pipeline LightGBM đầy đủ với **cross-validation**, **feature selection**, và **ensemble**.
- Tối ưu cho **dữ liệu mất cân bằng**, tự động điều chỉnh ngưỡng dựa trên **F1-score** và **Precision-Recall curve**.
- Tự động tính toán và trực quan hóa **feature importance**.
- Hỗ trợ **OOF prediction** để đánh giá tổng thể.
- Hỗ trợ nhiều **chiến lược trích xuất đặc trưng**, bao gồm Gaussian Process (GP).

---

