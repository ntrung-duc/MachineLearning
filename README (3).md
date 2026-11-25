# Hotel Booking Cancellation Prediction

## 1. Giới thiệu

Đồ án Học máy cơ bản dự đoán khả năng hủy đặt phòng khách sạn bằng Gaussian Naive Bayes, SVM (LinearSVC) và Random Forest.

Mục tiêu: phân tích dữ liệu, xử lý đặc trưng, huấn luyện, tối ưu và so sánh ba mô hình. Chỉ triển khai backend, không xây dựng website/app. Thời gian dự kiến: 2 ngày.

**Dataset:** [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) — 119.390 bản ghi, 32 cột. Target: `is_canceled` (0: không hủy, 1: hủy).

## 2. Phân công công việc

### Đức — Preprocessing, Naive Bayes & SVM

- [ ] EDA, kiểm tra missing values, duplicates và outliers.
- [ ] Làm sạch dữ liệu, chia train/test, xây dựng pipeline encoding, imputation và scaling phù hợp.
- [ ] Huấn luyện, thử nghiệm đặc trưng mới và tuning GaussianNB, LinearSVC.
- [ ] Đánh giá, phân tích lỗi và lưu pipeline hai mô hình.

### Giang — Feature Engineering, Random Forest & Evaluation

- [ ] Rà soát data leakage, tạo và lựa chọn đặc trưng.
- [ ] Huấn luyện, thử nghiệm đặc trưng mới và tuning Random Forest.
- [ ] Phân tích Feature Importance và lỗi dự đoán, lưu pipeline.
- [ ] Xây dựng hàm đánh giá chung, tổng hợp metrics và biểu đồ so sánh.
- [ ] Tích hợp backend trong `train_all.py`.

**Cả hai:** Thống nhất giao diện module, kiểm tra chéo code, hoàn thiện báo cáo, slide và chuẩn bị vấn đáp.

## 3. Quy trình thực nghiệm

```text
Load dataset → Kiểm tra dữ liệu → Train/Test Split
→ EDA trên train → Preprocessing Pipeline
→ Baseline → Feature Engineering / Selection
→ Cross-validation & Tuning → Final Test → Comparison
```

Cấu hình chung: train/test 80/20, `random_state=42`, `stratify=y`, Stratified 3-Fold CV (shuffle, seed 42), tối ưu F1-score lớp 1.

- Dùng cùng tập train/test và bộ đặc trưng gốc; pipeline có thể khác nhau theo thuật toán.
- Loại `reservation_status`, `reservation_status_date` và các trường không khả dụng tại thời điểm đặt phòng.
- Fit imputation, encoding, scaling và feature selection trong từng fold train bằng Pipeline; không dùng test để tuning.
- SVM chuẩn hóa biến số; GaussianNB cần đầu vào dense; Random Forest không cần scaling.
- Thử đặc trưng mới: `total_guests`, `total_nights`, `estimated_cost`, `has_children`, `previous_cancel_rate`, `lead_time_group`, `is_family`. Xử lý giá trị thiếu và mẫu số bằng 0; chỉ dùng thông tin khả dụng lúc đặt phòng.
- So sánh baseline, bổ sung đặc trưng, lựa chọn đặc trưng (nếu đủ thời gian) và tuning bằng CV. Chốt cấu hình trước khi đánh giá test.

| Mô hình | Tham số tuning | Phương pháp |
|---|---|---|
| GaussianNB | `var_smoothing` | GridSearchCV |
| LinearSVC | `C` | GridSearchCV |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` | RandomizedSearchCV (10 cấu hình) |

## 4. Cấu trúc project

```text
Hotel_Booking_ML/
├── dataset/hotel_bookings.csv
├── notebooks/
│   ├── 01_Preprocessing_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Naive_Bayes_SVM.ipynb
│   ├── 04_Random_Forest.ipynb
│   └── 05_Model_Comparison.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                 # Đức
│   ├── preprocessing.py          # Đức
│   ├── naive_bayes.py             # Đức
│   ├── svm_model.py               # Đức
│   ├── feature_engineering.py     # Giang
│   ├── random_forest.py           # Giang
│   ├── evaluation.py              # Giang
│   └── train_all.py               # Giang
├── models/
├── results/
│   ├── cv_metrics.csv
│   ├── metrics.csv
│   └── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

Notebook 01, 03 do Đức phụ trách; notebook 02, 04, 05 do Giang phụ trách. Notebook gọi module trong `src/`, không lặp logic backend. Lưu pipeline đã fit bằng Joblib.

## 5. Tiến độ (2 ngày)

| Thời gian | Đức | Giang |
|---|---|---|
| Ngày 1 sáng | EDA, preprocessing, split | Feature Engineering, evaluation |
| Ngày 1 chiều | NB và SVM baseline | RF baseline |
| Ngày 2 sáng | Thử đặc trưng, tuning NB/SVM | Thử đặc trưng, tuning RF |
| Ngày 2 chiều | Đánh giá NB/SVM | Đánh giá RF, so sánh, tích hợp |
| Cuối ngày 2 | Kiểm tra toàn bộ code, báo cáo | Kiểm tra toàn bộ code, báo cáo |

## 6. Kết quả đánh giá

| Mô hình | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Naive Bayes | — | — | — | — | — |
| SVM | — | — | — | — | — |
| Random Forest | — | — | — | — | — |

Lưu kết quả CV và test riêng; bổ sung confusion matrix, ROC curve, Feature Importance, thời gian chạy và biểu đồ so sánh. ROC-AUC dùng `decision_function()` cho LinearSVC, `predict_proba()[:, 1]` cho NB/RF. Không điền số liệu trước khi thực nghiệm.

## 7. Làm việc trên GitHub

- `main`: phiên bản ổn định; `dev-duc`: Đức; `dev-giang`: Giang.
- Commit theo chức năng, push thường xuyên, đồng bộ main và tạo Pull Request để kiểm tra chéo.
- Tích hợp sớm module chung; không tự ý sửa file người khác phụ trách hoặc chia lại dữ liệu.
- Dataset và model lớn lưu Google Drive chung hoặc tải từ nguồn gốc; dùng `.gitignore` cho file lớn và cache.

## 8. Hướng dẫn chạy

Python 3.11 (đề xuất), Google Colab hoặc local. Thư viện: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, jupyter.

Tải dataset vào `dataset/`, chạy tại thư mục gốc:

```bash
python -m pip install -r requirements.txt
python -m src.train_all
```

**Tiêu chí hoàn thành:** Ba mô hình chạy được, có baseline và tuning, thực nghiệm đặc trưng, kết quả đánh giá, pipeline đã lưu, báo cáo và code backend không chứa comment theo yêu cầu giảng viên.

**Trạng thái:** Kế hoạch triển khai; cấu trúc, lệnh chạy và kết quả sẽ được kiểm thử, cập nhật sau khi hoàn thành code.
