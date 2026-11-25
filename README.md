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
- Không tự động xóa bản ghi giống nhau nếu chưa xác định là lỗi trùng lặp. Rà soát `assigned_room_type`, `booking_changes`, `days_in_waiting_list` và các trường phát sinh sau thời điểm dự đoán.
- Chỉ sử dụng thông tin khả dụng tại thời điểm đặt phòng; ghi lại lý do giữ hoặc loại từng đặc trưng nghi ngờ.
- So sánh baseline, bổ sung đặc trưng, lựa chọn đặc trưng (nếu đủ thời gian) và tuning bằng CV. Chốt cấu hình trước khi đánh giá test.

### 3.1. Feature Engineering (Giang)

| Đặc trưng | Công thức / cách tạo |
|---|---|
| `total_guests` | adults + children + babies |
| `total_nights` | stays_in_weekend_nights + stays_in_week_nights |
| `estimated_cost` | adr × total_nights |
| `has_children` | children + babies > 0 |
| `previous_cancel_rate` | previous_cancellations / (previous_cancellations + previous_bookings_not_canceled) |
| `lead_time_group` | Nhóm lead_time: 0–7, 8–30, 31–90, trên 90 ngày |
| `is_family` | adults > 0 và children + babies > 0 |

Xử lý giá trị thiếu trước khi tính hoặc giữ NaN để pipeline xử lý; tỷ lệ hủy khi mẫu số bằng 0 quy ước bằng 0. Kiểm tra tính khả dụng của các biến nguồn. Feature Selection có thể thử SelectPercentile trong pipeline; phân tích Feature Importance của RF và giải thích đặc trưng giữ/loại.

### 3.2. Hyperparameter Tuning

| Mô hình | Tham số và giá trị thử | Phương pháp |
|---|---|---|
| GaussianNB | var_smoothing: 1e-7, 1e-8, 1e-9, 1e-10 | GridSearchCV |
| LinearSVC | C: 0.01, 0.1, 1, 10 | GridSearchCV |
| Random Forest | n_estimators: 50, 100, 200; max_depth: 10, 20, None; min_samples_split: 2, 5, 10; min_samples_leaf: 1, 2, 4 | RandomizedSearchCV (10 cấu hình, seed 42) |

Các tham số trong pipeline dùng tiền tố `model__`. Kiểm tra hội tụ LinearSVC; ưu tiên baseline nếu thiếu thời gian.

### 3.3. Danh sách thực nghiệm

| TN | Nội dung |
|---|---|
| TN1 | Baseline của cả ba mô hình trên đặc trưng gốc |
| TN2 | Thêm đặc trưng mới, giữ nguyên tham số |
| TN3 | Thử lựa chọn đặc trưng (nếu đủ thời gian) |
| TN4 | Tuning cấu hình được chọn từ TN1–TN3 |
| TN5 | Fit lại trên toàn bộ train và đánh giá cấu hình đã chốt trên test |

TN1–TN4 dùng cùng các fold CV; ghi F1 trung bình và độ lệch chuẩn. Không dùng test để chọn đặc trưng, tham số hay mô hình. Báo cáo cả kết quả cải tiến không thành công.

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

### 4.1. Giao diện module dùng chung

| Module | Hàm dự kiến | Đầu ra |
|---|---|---|
| preprocessing.py | `load_and_split(path)` | X_train, X_test, y_train, y_test (chưa fit preprocessing) |
| preprocessing.py | `build_preprocessor(model_name, X_train)` | ColumnTransformer chưa fit, dùng schema đầu vào tương ứng |
| feature_engineering.py | `add_features(X)` | DataFrame có đặc trưng mới, giữ nguyên index |
| naive_bayes.py, svm_model.py, random_forest.py | `build_model(X_train, use_features=False, use_selection=False)` | Pipeline chưa fit |
| evaluation.py | `evaluate_model(model, X, y)` | Dict metrics và confusion matrix |
| train_all.py | `main()` | Chạy thực nghiệm, lưu pipeline và kết quả |

`X` là DataFrame, `y` là Series nhãn 0/1. Pipeline gồm bước tạo đặc trưng (nếu bật), preprocessing, lựa chọn đặc trưng (nếu bật), model. Hàm tạo đặc trưng chỉ thực hiện phép biến đổi cố định theo từng dòng; mọi biến đổi học từ dữ liệu phải fit trong fold train. `config.py` lưu seed, target, test size, số fold, scoring và danh sách cột loại bỏ. Hai thành viên thống nhất chữ ký hàm trước khi code.

## 5. Tiến độ (2 ngày)

| Thời gian | Đức | Giang |
|---|---|---|
| Ngày 1 sáng | EDA, preprocessing, split | Feature Engineering, evaluation |
| Ngày 1 chiều | NB và SVM baseline | RF baseline |
| Ngày 2 sáng | Thử đặc trưng, tuning NB/SVM | Thử đặc trưng, tuning RF |
| Ngày 2 chiều | Đánh giá NB/SVM | Đánh giá RF, so sánh, tích hợp |
| Cuối ngày 2 | Kiểm tra toàn bộ code, báo cáo | Kiểm tra toàn bộ code, báo cáo |

## 6. Kết quả đánh giá

### Cross-validation (F1 trung bình ± độ lệch chuẩn)

| Mô hình | Baseline | Feature Engineering | Feature Selection | Tuned |
|---|---|---|---|---|
| Naive Bayes | — | — | — | — |
| SVM | — | — | — | — |
| Random Forest | — | — | — | — |

### Final Test (cấu hình đã chốt)

| Mô hình | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Naive Bayes | — | — | — | — | — |
| SVM | — | — | — | — | — |
| Random Forest | — | — | — | — | — |

Lưu kết quả CV vào `results/cv_metrics.csv` và test vào `results/metrics.csv`; bổ sung confusion matrix, ROC curve, Feature Importance, thời gian chạy và biểu đồ so sánh. ROC-AUC dùng `decision_function()` cho LinearSVC, `predict_proba()[:, 1]` cho NB/RF. Không điền số liệu trước khi thực nghiệm.

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

Thống nhất phiên bản thư viện thực tế trong `requirements.txt`. Kiểm tra tải lại pipeline bằng Joblib và chạy toàn bộ backend trên máy mang đi thi.

**Tiêu chí hoàn thành:** Ba mô hình chạy được, có baseline và tuning, thực nghiệm đặc trưng, kết quả đánh giá, pipeline đã lưu, báo cáo và code backend không chứa comment theo yêu cầu giảng viên.

**Trạng thái:** Kế hoạch triển khai; cấu trúc, lệnh chạy và kết quả sẽ được kiểm thử, cập nhật sau khi hoàn thành code.
