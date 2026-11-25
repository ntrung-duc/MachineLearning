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

Phụ trách tiền xử lý dữ liệu, khám phá dữ liệu và 2 mô hình Machine Learning: Naive Bayes và SVM.

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
└── 02_Naive_Bayes_SVM.ipynb
```

---

### Giang

Phụ trách mô hình Random Forest, đánh giá và so sánh hiệu quả các mô hình.

**Nhiệm vụ:**

- Xây dựng mô hình Random Forest.
- Train và tuning nhẹ Random Forest.
- Xây dựng các hàm đánh giá mô hình dùng chung.
- Tổng hợp kết quả của Naive Bayes, SVM và Random Forest.
- So sánh hiệu quả giữa 3 mô hình.
- Vẽ confusion matrix cho các mô hình.
- Vẽ biểu đồ so sánh các metrics.
- Xác định mô hình hoạt động tốt nhất dựa trên kết quả thực nghiệm.

**Các file phụ trách:**

```text
src/
├── random_forest.py
└── evaluation.py
```

**Notebook phụ trách:**

```text
notebooks/
├── 03_Random_Forest.ipynb
└── 04_Model_Comparison.ipynb
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
| 10:00–12:00 | Hỗ trợ kiểm tra dữ liệu, kết quả | Hoàn thiện evaluation metrics, confusion matrix |
| 13:00–14:30 | Tuning nhẹ NB và SVM | Tuning nhẹ RF, tổng hợp metrics |
| 14:30–16:00 | Kiểm tra kết quả sau tuning | Hoàn thiện model comparison, vẽ biểu đồ |
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
- Random State cố định
- Stratified Split
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

**Quy ước chung:**

- Các mô hình sử dụng chung train/test split để đảm bảo so sánh công bằng.
- Sử dụng `random_state` cố định để có thể tái lập kết quả.
- Sử dụng `stratify=y` khi chia dữ liệu để giữ tỷ lệ các lớp.
- Preprocessing phải được fit trên tập train và áp dụng lên test nhằm tránh data leakage.
- SVM cần scaling dữ liệu phù hợp.
- Các bước preprocessing cần được thống nhất trước khi tích hợp mô hình.

---

## 6. Các metrics đánh giá

Các mô hình sử dụng chung tập test và các metrics để so sánh công bằng.

- **Accuracy:** Tỷ lệ dự đoán đúng trên tổng số mẫu.
- **Precision:** Tỷ lệ dự đoán hủy phòng đúng trong số các mẫu dự đoán hủy.
- **Recall:** Khả năng phát hiện các booking thực sự bị hủy.
- **F1-score:** Trung bình điều hòa giữa Precision và Recall.
- **ROC-AUC:** Đánh giá khả năng phân biệt hai lớp của mô hình.
- **Confusion Matrix:** Thể hiện số lượng dự đoán đúng và sai của từng lớp.

Các metrics được sử dụng để đánh giá cả mô hình baseline và mô hình sau tuning.

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
│   ├── 02_Naive_Bayes_SVM.ipynb
│   ├── 03_Random_Forest.ipynb
│   └── 04_Model_Comparison.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── naive_bayes.py
│   ├── svm_model.py
│   ├── random_forest.py
│   └── evaluation.py
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

### Mô tả các thư mục

| Thư mục/File | Chức năng |
|---|---|
| `dataset/` | Lưu dataset đầu vào |
| `notebooks/` | Thực hiện EDA, training, tuning và model comparison |
| `src/` | Chứa các module Python triển khai preprocessing, model và evaluation |
| `results/` | Lưu metrics và kết quả thực nghiệm |
| `results/figures/` | Lưu confusion matrix và biểu đồ so sánh |
| `requirements.txt` | Danh sách thư viện cần cài đặt |
| `README.md` | Tài liệu mô tả project |

### Mô tả notebooks

| Notebook | Nội dung |
|---|---|
| `01_Preprocessing_EDA.ipynb` | Khám phá dữ liệu, xử lý missing values, encoding và chuẩn bị dữ liệu |
| `02_Naive_Bayes_SVM.ipynb` | Train, evaluate và tuning Naive Bayes, SVM |
| `03_Random_Forest.ipynb` | Train, evaluate và tuning Random Forest |
| `04_Model_Comparison.ipynb` | Tổng hợp metrics, confusion matrix, biểu đồ và so sánh các mô hình |

### Mô tả source code

| File | Chức năng |
|---|---|
| `preprocessing.py` | Xử lý missing values, encoding, scaling và train/test split |
| `naive_bayes.py` | Xây dựng, train và tuning Naive Bayes |
| `svm_model.py` | Xây dựng, train và tuning SVM |
| `random_forest.py` | Xây dựng, train và tuning Random Forest |
| `evaluation.py` | Các hàm tính metrics và hỗ trợ đánh giá, confusion matrix |

---

## 9. Quy định làm việc chung

- Cả hai sử dụng chung dataset và cách chia train/test.
- Không tự ý thay đổi preprocessing làm ảnh hưởng kết quả so sánh nếu chưa thống nhất.
- Thống nhất tên biến và định dạng dữ liệu đầu vào/đầu ra giữa các file.
- Mỗi người phụ trách code của mình và commit lên GitHub thường xuyên.
- Khi hoàn thành một phần, kiểm tra code chạy thành công trước khi push.
- Các file source code được tổ chức trong `src/`, notebook dùng để chạy thử, thực nghiệm và trình bày kết quả.
- Phần model comparison được thực hiện trong `04_Model_Comparison.ipynb`, không tạo thêm `model_comparison.py`.
- Ngày 2 phải tích hợp và chạy lại toàn bộ mô hình.
- Ưu tiên code chạy được, kết quả đánh giá đầy đủ trước khi tuning nâng cao.

### Quy ước dữ liệu dùng chung

Module preprocessing cần chuẩn bị dữ liệu thống nhất để các mô hình sử dụng:

```python
X_train, X_test, y_train, y_test
```

Các mô hình phải sử dụng cùng tập train/test để đảm bảo kết quả so sánh có ý nghĩa.

---

## 10. Checklist hoàn thành

### Đức

- [ ] EDA
- [ ] Kiểm tra missing values, duplicates và outliers
- [ ] Preprocessing
- [ ] Encoding categorical variables
- [ ] Train/test split
- [ ] Naive Bayes baseline
- [ ] Naive Bayes tuning
- [ ] SVM baseline
- [ ] SVM tuning
- [ ] Kiểm tra code chạy thành công
- [ ] Push code lên GitHub

### Giang

- [ ] Random Forest baseline
- [ ] Random Forest tuning
- [ ] Evaluation metrics
- [ ] Confusion matrix
- [ ] Model comparison
- [ ] Comparison chart
- [ ] Tổng hợp metrics của cả 3 mô hình
- [ ] Lưu kết quả vào `results/metrics.csv`
- [ ] Push code lên GitHub

### Cả hai

- [ ] Thống nhất preprocessing và train/test split
- [ ] Test code từ đầu đến cuối
- [ ] Đảm bảo không có data leakage
- [ ] Kiểm tra baseline và tuned models
- [ ] Thống nhất kết quả
- [ ] Hoàn thiện project trong 2 ngày

---

## 11. Kết quả dự kiến

Sau khi hoàn thành dự án, nhóm sẽ có:

- Code preprocessing dữ liệu.
- 3 mô hình Machine Learning: Naive Bayes, SVM và Random Forest.
- Kết quả baseline và tuned model.
- Các metrics đánh giá mô hình.
- Confusion matrix cho từng mô hình.
- Biểu đồ so sánh hiệu quả các mô hình.
- Bảng tổng hợp kết quả trong `results/metrics.csv`.
- Kết luận và lựa chọn mô hình phù hợp nhất cho bài toán Hotel Booking Cancellation Prediction.

**Lưu ý:** Mô hình tốt nhất sẽ được xác định dựa trên kết quả thực nghiệm, không lựa chọn trước khi training và evaluation.
