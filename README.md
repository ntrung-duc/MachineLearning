# Hotel Booking Cancellation Prediction

## 1. Giới thiệu dự án

Dự án Machine Learning dự đoán khả năng khách hàng hủy đặt phòng khách sạn (Hotel Booking Cancellation Prediction).

Mục tiêu là xây dựng, huấn luyện và đánh giá 3 mô hình Machine Learning:

- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest

Sau đó so sánh hiệu quả của các mô hình để lựa chọn mô hình phù hợp nhất cho bài toán dự đoán hủy phòng.

**Phạm vi dự án:** Chỉ tập trung vào code Machine Learning, tiền xử lý dữ liệu, huấn luyện, tuning và đánh giá mô hình. Không xây dựng website, backend hoặc ứng dụng demo.

---

## 2. Dataset

Dataset sử dụng: Hotel Booking Demand Dataset.

- File dữ liệu: `hotel_bookings.csv`
- Target: `is_canceled`
- `0`: Không hủy đặt phòng
- `1`: Hủy đặt phòng

Dataset chứa các thông tin liên quan đến đặt phòng khách sạn như loại khách sạn, thời gian lưu trú, số lượng khách, loại khách, giá phòng và các thông tin đặt phòng khác.

---

## 3. Phân công công việc

### Đức

Phụ trách tiền xử lý dữ liệu, khám phá dữ liệu và 2 mô hình Machine Learning.

**Nhiệm vụ:**

- Khám phá và phân tích dữ liệu (EDA).
- Kiểm tra dữ liệu thiếu, dữ liệu trùng lặp và các giá trị bất thường.
- Xử lý missing values.
- Encoding các biến categorical.
- Chuẩn bị dữ liệu đầu vào cho các mô hình.
- Chia dữ liệu train/test.
- Xây dựng mô hình Naive Bayes.
- Xây dựng mô hình SVM.
- Train, đánh giá và tuning nhẹ 2 mô hình.
- Kiểm tra và đảm bảo code chạy được.

**Các file phụ trách:**

```text
src/
├── preprocessing.py
├── naive_bayes.py
└── svm_model.py
```

**Notebook phụ trách:**

```text
notebooks/
├── 01_Preprocessing_EDA.ipynb
├── 02_Naive_Bayes.ipynb
└── 03_SVM.ipynb
```

---

### Giang

Phụ trách mô hình Random Forest, đánh giá và so sánh hiệu quả các mô hình.

**Nhiệm vụ:**

- Xây dựng mô hình Random Forest.
- Train và tuning nhẹ Random Forest.
- Xây dựng các hàm đánh giá mô hình.
- Tổng hợp kết quả của Naive Bayes, SVM và Random Forest.
- So sánh hiệu quả giữa 3 mô hình.
- Vẽ confusion matrix cho các mô hình.
- Vẽ biểu đồ so sánh các metrics.
- Xác định mô hình hoạt động tốt nhất dựa trên kết quả thực nghiệm.

**Các file phụ trách:**

```text
src/
├── random_forest.py
├── evaluation.py
└── model_comparison.py
```

**Notebook phụ trách:**

```text
notebooks/
├── 04_Random_Forest.ipynb
└── 05_Model_Comparison.ipynb
```

---

## 4. Phân công theo thời gian (2 ngày)

### Ngày 1 — Preprocessing và xây dựng mô hình

| Thời gian | Đức | Giang |
|---|---|---|
| 8:00–9:00 | Tải dataset, khám phá dữ liệu | Tìm hiểu dataset, nghiên cứu Random Forest |
| 9:00–11:00 | EDA, xử lý missing values, encoding | Chuẩn bị cấu trúc code Random Forest |
| 11:00–12:00 | Hoàn thành preprocessing, train/test split | Thống nhất cách sử dụng dữ liệu đầu vào |
| 13:00–15:00 | Code và train Naive Bayes | Code Random Forest |
| 15:00–17:00 | Code và train SVM | Train, đánh giá Random Forest |
| 17:00–18:00 | Kiểm tra code, đẩy GitHub | Kiểm tra code, đẩy GitHub |

### Ngày 2 — Evaluation, tuning và hoàn thiện

| Thời gian | Đức | Giang |
|---|---|---|
| 8:00–10:00 | Kiểm tra, chạy lại NB và SVM | Kiểm tra, chạy lại RF |
| 10:00–12:00 | Hỗ trợ kiểm tra dữ liệu, kết quả | Code evaluation metrics, confusion matrix |
| 13:00–14:30 | Tuning nhẹ NB và SVM | Tuning nhẹ RF, tổng hợp metrics |
| 14:30–16:00 | Kiểm tra kết quả sau tuning | Code model comparison, vẽ biểu đồ |
| 16:00–17:00 | Test tích hợp toàn bộ code | Test tích hợp toàn bộ code |
| 17:00–18:00 | Hoàn thiện README, backup code | Hoàn thiện bảng so sánh và kết quả |

---

## 5. Quy trình Machine Learning

```text
Load Dataset
     |
     v
Exploratory Data Analysis (EDA)
     |
     v
Data Preprocessing
- Handle Missing Values
- Encode Categorical Features
- Prepare Features and Target
     |
     v
Train/Test Split (80/20)
     |
     v
Train 3 Models
- Naive Bayes
- SVM
- Random Forest
     |
     v
Evaluate Baseline Models
     |
     v
Hyperparameter Tuning
     |
     v
Evaluate Tuned Models
     |
     v
Compare Models
     |
     v
Select Best Model
```

---

## 6. Các metrics đánh giá

Các mô hình sử dụng chung tập test và các metrics để so sánh công bằng.

- **Accuracy:** Tỷ lệ dự đoán đúng trên tổng số mẫu.
- **Precision:** Tỷ lệ dự đoán hủy phòng đúng trong số các mẫu dự đoán hủy.
- **Recall:** Khả năng phát hiện các booking thực sự bị hủy.
- **F1-score:** Trung bình điều hòa giữa Precision và Recall.
- **ROC-AUC:** Đánh giá khả năng phân biệt hai lớp của mô hình.
- **Confusion Matrix:** Thể hiện số lượng dự đoán đúng và sai của từng lớp.

---

## 7. Hyperparameter Tuning

Sau khi train các mô hình với tham số mặc định (baseline), nhóm sẽ tuning nhẹ để tìm cấu hình tốt hơn.

### Naive Bayes

Thử điều chỉnh:

- `var_smoothing`

### SVM

Thử điều chỉnh:

- `C`
- `kernel`
- `gamma`

### Random Forest

Thử điều chỉnh:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

Tuning được thực hiện trong phạm vi phù hợp với thời gian 2 ngày.

---

## 8. Cấu trúc project

```text
Hotel_Booking_ML/
│
├── dataset/
│   └── hotel_bookings.csv
│
├── notebooks/
│   ├── 01_Preprocessing_EDA.ipynb
│   ├── 02_Naive_Bayes.ipynb
│   ├── 03_SVM.ipynb
│   ├── 04_Random_Forest.ipynb
│   └── 05_Model_Comparison.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── naive_bayes.py
│   ├── svm_model.py
│   ├── random_forest.py
│   ├── evaluation.py
│   └── model_comparison.py
│
├── results/
│   ├── metrics.csv
│   └── figures/
│       ├── confusion_matrix_nb.png
│       ├── confusion_matrix_svm.png
│       ├── confusion_matrix_rf.png
│       └── model_comparison.png
│
├── requirements.txt
└── README.md
```

---

## 9. Quy định làm việc chung

- Cả hai sử dụng chung dataset và cách chia train/test.
- Không tự ý thay đổi preprocessing làm ảnh hưởng kết quả so sánh nếu chưa thống nhất.
- Mỗi người phụ trách code của mình và commit lên GitHub thường xuyên.
- Khi hoàn thành một phần, kiểm tra code chạy thành công trước khi push.
- Ngày 2 phải tích hợp và chạy lại toàn bộ mô hình.
- Ưu tiên code chạy được, kết quả đánh giá đầy đủ trước khi tuning nâng cao.

---

## 10. Checklist hoàn thành

### Đức

- [ ] EDA
- [ ] Preprocessing
- [ ] Train/test split
- [ ] Naive Bayes baseline
- [ ] Naive Bayes tuning
- [ ] SVM baseline
- [ ] SVM tuning
- [ ] Push code lên GitHub

### Giang

- [ ] Random Forest baseline
- [ ] Random Forest tuning
- [ ] Evaluation metrics
- [ ] Confusion matrix
- [ ] Model comparison
- [ ] Comparison chart
- [ ] Tổng hợp metrics của cả 3 mô hình
- [ ] Push code lên GitHub

### Cả hai

- [ ] Test code từ đầu đến cuối
- [ ] Đảm bảo không có data leakage
- [ ] Thống nhất kết quả
- [ ] Hoàn thiện project trong 2 ngày
