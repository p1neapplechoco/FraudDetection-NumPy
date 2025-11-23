# Credit Card Fraud Detection

Dự án mô hình học máy để phát hiện giao dịch gian lận trong thẻ tín dụng, được implement hoàn toàn bằng NumPy.

---

## Mục Lục

- [Giới Thiệu](#giới-thiệu)
- [Dataset](#dataset)
- [Phương Pháp](#phương-pháp)
- [Cài Đặt](#cài-đặt)
- [Sử Dụng](#sử-dụng)
- [Kết Quả](#kết-quả)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Thách Thức & Giải Pháp](#thách-thức--giải-pháp)
- [Hướng Phát Triển](#hướng-phát-triển)
- [Contributors](#contributors)
- [License](#license)
- [Liên Hệ](#liên-hệ)

---

## Giới Thiệu

### Mô Tả Bài Toán

Phát hiện gian lận thẻ tín dụng là một bài toán phân loại nhị phân với dữ liệu **cực kỳ mất cân bằng**, trong đó:
- Giao dịch hợp lệ chiếm đại đa số (~99.8%)
- Giao dịch gian lận chỉ chiếm thiểu số (~0.2%)

Đây là bài toán **phát hiện sự kiện khó xảy ra** điển hình trong thực tế.

### Động Lực & Ứng Dụng Thực Tế

**Tầm quan trọng:**
- Thiệt hại tài chính: Gian lận thẻ tín dụng gây thiệt hại hàng tỷ USD mỗi năm
- Bảo vệ khách hàng: Phát hiện sớm giúp ngăn chặn tổn thất và bảo vệ uy tín ngân hàng
- Tự động hóa: Giảm chi phí xác minh thủ công

**Ứng dụng:**
- Hệ thống cảnh báo gian lận real-time
- Đánh giá rủi ro giao dịch
- Phân tích hành vi giao dịch bất thường

### Mục Tiêu Cụ Thể

1. **Implement từ đầu**: Xây dựng các thuật toán học máy chỉ sử dụng NumPy
2. **Xử lý imbalanced data**: Áp dụng các kỹ thuật resampling (under-sampling, over-sampling)
3. **So sánh mô hình**: Đánh giá hiệu quả của Logistic Regression và Random Forest

---

## Dataset

### Nguồn Dữ Liệu

**Credit Card Fraud Detection Dataset** từ Kaggle
- Link: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Dữ liệu thực từ giao dịch thẻ tín dụng của khách hàng châu Âu (tháng 9/2013)

### Mô Tả Features

Dataset gồm **31 features**:

| Feature | Mô Tả | Type |
|---------|-------|------|
| `Time` | Thời gian từ giao dịch đầu tiên (giây) | Numeric |
| `V1-V28` | Features được biến đổi bằng PCA | Numeric |
| `Amount` | Số tiền giao dịch | Numeric |
| `Class` | Target variable (0: Normal, 1: Fraud) | Binary |

**Lưu ý về PCA transformation:**
- Các features V1-V28 đã được PCA
- Không có thông tin về ý nghĩa cụ thể của từng feature
- Chỉ `Time` và `Amount` là features gốc

### Kích Thước & Đặc Điểm

- **Tổng số giao dịch**: 284,807 transactions
- **Số lượng fraud**: 492 (0.172%)
- **Số lượng normal**: 284,315 (99.828%)
- **Imbalance ratio**: ~1:578

**Đặc điểm quan trọng:**
- Tỷ lệ mất cân bằng cực cao
- Dữ liệu đầy đủ
- V1-V28 đã được chuẩn hóa

--

## Research Questions & Insights

### Q1: Giao dịch gian lận có xu hướng có số tiền lớn hơn giao dịch thường không?

**Giả thuyết ban đầu:** Fraudsters thường thực hiện các giao dịch lớn để tối đa hóa lợi nhuận.

**Trả lời:** Giao dịch gian lận có giá trị **thấp hơn** giao dịch thường!

**Insights:**
- Trung vị của giao dịch gian lận thấp hơn đáng kể so với giao dịch bình thường.
- Những giao dịch gian lận sử dụng chiến lược thực hiện nhiều giao dịch nhỏ thay vì một giao dịch lớn
- Đặc trưng `Amount` quan trọng nhưng **không theo hướng trực quan**.

**Kết luận:**
- Không thể phát hiện gian lận chỉ bằng cách lọc giao dịch lớn
- Cần xem xét patterns phức tạp hơn (tần suất, thời gian, kết hợp nhiều đặc trưng)

### Q2: Giao dịch gian lận có xu hướng xảy ra vào thời điểm cụ thể nào trong ngày?

**Giả thuyết:** Fraud có thể tập trung vào giờ off-peak khi giám sát kém hơn.

**Trả lời:**  Gian lận tập trung vào các thời điểm cụ thể trong ngày.

**Quan sát từ phân tích:**

**1. Giờ cao điểm cho các giao dịch gian lận:**
- **Giờ đêm khuya (2-4 AM)**: tỉ lệ gian lận cao nhất
  
- **Giờ trưa (11 AM - 1 PM)**: tỉ lệ gian lận cũng cao
  - Trộn lẫn vào các giao dịch bình thường
  - Khó phát hiện hơn với lượng giao dịch khủng

**2. Khoảng thời gian an toàn:**
- Sáng sớm (5-8 AM) và chiều tối (6-9 PM): tỉ lệ gian lận thấp
- Có thể do tăng cường monitoring trong giờ hành chính

### Q3: Các PCA components nào có correlation mạnh nhất với fraud?

**Motivation:** V1-V28 là PCA transformed features - cần hiểu which components matter most.

**Trả lời:** V14, V17, V12 có tương quan mạnh nhất.

**1. Tương quan âm:**
- **V14, V17, V12**: tương quan mạnh nhất (~-0.30 đến -0.33)
- **V10, V16, V3**: tương quan trung bình (~-0.20 đến -0.25)

**2. Tương quan dương:**
- **V2, V4, V11**: tương quan dương nhưng yếu (<0.15)

**3. Tương quan gần 0:**
- **Time, Amount**: tương quan rất thấp với Class
  - Không có mối quan hệ tuyến tính rõ ràng
  - Quan hệ phi tuyến có thể tồn tại

**4. Không có đặc trưng nào có |tương quan| > 0.35:**
- Fraud detection cần **sự kết hợp giữa nhiều đặc trưng**
- Mô hình tuyến tính (Logistic Regression) sẽ perform tệ

---

## Phương Pháp

### 1. Quy Trình Xử Lý Dữ Liệu

#### 1.1 Data Cleaning & Preprocessing
```
Raw Data > Handle Outliers > Feature Scaling > Train/Test Split
```

**Các bước chi tiết:**

1. **Xử lí các giá trị ngoại lai**
   - Sử dụng phương pháp IQR
   - Công thức: `outliers = values < Q1 - 1.5*IQR hoặc values > Q3 + 1.5*IQR`

2. **Feature Scaling**
   - **StandardScaler**: `z = (x - μ) / σ`
   - **RobustScaler**: `z = (x - median) / IQR` (tốt với outliers)
   - **MinMaxScaler**: `z = (x - min) / (max - min)`

3. **Chia tập dữ liệu**
   - Stratified split để giữ tỷ lệ class
   - Ratio: 80% training / 20% testing

#### 1.2 Handling Imbalanced Data

**Under-Sampling:**
```python
majority_indices = random.choice(majority_indices, size=len(minority))
balanced_data = concatenate([minority, majority_indices])
```

**Over-Sampling:**
```python
minority_repeated = tile(minority, (ratio, 1))
balanced_data = concatenate([majority, minority_repeated])
```

### 2. Thuật Toán Sử Dụng

#### 2.1 Logistic Regression

**Mô tả**: Linear classifier sử dụng sigmoid function để dự đoán xác suất.

**Công thức toán học:**

1. **Linear Model**:
   ```
   z = X · w + b
   ```
   Trong đó:
   - X: feature matrix (n_samples × n_features)
   - w: weight vector (n_features × 1)
   - b: bias term (scalar)

2. **Sigmoid Function**:
   ```
   σ(z) = 1 / (1 + e^(-z))
   ```
   Output: xác suất trong khoảng [0, 1]

3. **Binary Cross-Entropy Loss**:
   ```
   L = -1/n Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
   ```

4. **Gradient Descent**:
   ```
   ∂L/∂w = 1/n · X^T · (ŷ - y)
   ∂L/∂b = 1/n · Σ(ŷ - y)
   
   w := w - α · ∂L/∂w
   b := b - α · ∂L/∂b
   ```
   Trong đó α là learning rate

**Implementation NumPy:**

```python
z = np.einsum('ij,j->i', X, weights) + bias
predictions = 1 / (1 + np.exp(-z))

error = predictions - y
dw = np.einsum('ji,j->i', X, error) / n_samples
db = np.mean(error)

weights -= learning_rate * dw
bias -= learning_rate * db
```

**Ưu điểm einsum:**
- Nhanh hơn `np.dot()` cho một số operations
- Memory efficient hơn
- Dễ đọc với notation rõ ràng

#### 2.2 Random Forest

**Mô tả**: Ensemble method kết hợp nhiều Decision Trees.

**Thuật toán Decision Tree:**

1. **Gini Impurity** (độ đo split quality):
   ```
   Gini = 1 - Σ(p_i)²
   ```
   Trong đó p_i là tỷ lệ của class i tại node

2. **Information Gain**:
   ```
   Gain = Gini_parent - Σ(n_i/n · Gini_i)
   ```

3. **Splitting Process**:
   - Với mỗi feature, thử tất cả threshold có thể
   - Chọn split có Gini impurity nhỏ nhất
   - Recursively split cho đến stopping criteria

**Random Forest Ensemble:**

1. **Bootstrap Aggregating (Bagging)**:
   ```
   Cho mỗi tree t trong T trees:
       1. Sample n_samples với replacement từ training data
       2. Train decision tree trên bootstrap sample
       3. Chỉ xem xét √n_features random features tại mỗi split
   ```

2. **Prediction**:
   ```
   ŷ = mode([tree_1(x), tree_2(x), ..., tree_T(x)])
   ```
   (Majority voting)

3. **Probability Estimation**:
   ```
   P(class=1|x) = (Σ I(tree_t(x) = 1)) / T
   ```

**Implementation NumPy:**

```python
def gini(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.einsum('i,i->', probs, probs)

def find_best_split(X, y, feature_idx):
    sorted_idx = np.argsort(X[:, feature_idx])
    thresholds = np.unique(X[sorted_idx, feature_idx])
    
    left_masks = X[:, feature_idx][:, None] <= thresholds
    ...
```

**Tối ưu hóa:**
- Pre-allocate arrays với `np.empty()`
- Dùng boolean indexing thay vì loops
- Cache calculations để tránh redundant computation

### 3. Evaluation Metrics

Với dữ liệu mất cân bằng, chỉ dùng accuracy là không đủ, cần nhiều thang đo:

#### 3.1 Confusion Matrix

```
                Predicted
              |  0  |  1  |
    ----------|-----|-----|
Actual   0    | TN  | FP  |
         1    | FN  | TP  |
```

#### 3.2 Metrics

1. **Precision** (Độ chính xác của dự đoán positive):
   ```
   Precision = TP / (TP + FP)
   ```
2. **Recall/Sensitivity** (Tỷ lệ phát hiện được fraud):
   ```
   Recall = TP / (TP + FN)
   ```

3. **F1-Score** (Harmonic mean của Precision và Recall):
   ```
   F1 = 2 · (Precision · Recall) / (Precision + Recall)
   ```

4. **PR-AUC** (Area Under Precision-Recall Curve):
- Phù hợp với dữ liệu mất cân bằng.
---

## Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.8 trở lên
- NumPy 1.20+
- Matplotlib 3.3+
- Seaborn 0.11+

### Các Bước Cài Đặt

1. **Clone repository**:
   ```bash
   git clone https://github.com/p1neapplechoco/FraudDetection-NumPy.git
   cd FraudDetection-NumPy
   ```

2. **Tạo virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # hoặc
   .venv\Scripts\activate     # Windows
   ```

3. **Cài đặt dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Cài đặt package**:
   ```bash
   python setup.py install
   ```

### Download Dataset

- Tải dữ liệu từ link: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
```bash
# Tạo thư mục data nếu chưa có
mkdir -p data/raw
```
- Di chuyển dữ liệu vào thư mục vừa tạo ở trên

---

## Sử Dụng

### 1. Data Exploration

```bash
jupyter-notebook notebooks/01_data_exploration.ipynb
```

**Nội dung:**
- Phân tích phân phối dữ liệu
- Khám phá dữ liệu

### 2. Data Preprocessing

```bash
jupyter-notebook notebooks/02_preprocessing.ipynb
```

**Các bước thực hiện:**
- Xử lí các giá trị ngoại lai
- Thực hiện scaling ở các cột

### 3. Handling Imbalance

```bash
jupyter-notebook notebooks/03_handling_imbalance.ipynb
```

**Áp dụng:**
- Under-sampling technique
- Over-sampling technique
- So sánh hiệu quả của các phương pháp
- Lưu processed data cho modeling

### 4. Model Training & Evaluation

```bash
jupyter-notebook notebooks/04_modeling.ipynb
```
---

## Kết Quả

### Performance Metrics

#### Logistic Regression (trained on under-sampled data)

| Metric | Under-sampled Test | Original Imbalanced Test |
|--------|-------------------|--------------------------|
| Accuracy | 0.9392 | 0.9481 |
| Precision | 0.9724 | 0.0326 |
| Recall | 0.9097 | 0.9430 |
| F1-Score | 0.9400 | 0.0630 |
| PR-AUC | 0.9647 | 0.4879 |

#### Random Forest - Baseline (n_trees=10)

| Metric | Under-sampled Test | Original Imbalanced Test |
|--------|-------------------|--------------------------|
| Accuracy | 0.9392 | 0.9622 |
| Precision | 0.9724 | 0.0445 |
| Recall | 0.9097 | 0.9494 |
| F1-Score | 0.9400 | 0.0849 |
| PR-AUC | 0.9647 | 0.4970 |

#### Random Forest - Optimized (n_trees=50, max_depth=15)

| Metric | Original Imbalanced Test |
|--------|--------------------------|
| Accuracy | **0.9802** |
| Precision | **0.0823** |
| Recall | **0.9557** |
| F1-Score | **0.1516** |
| PR-AUC | **0.5191** |

### Phân Tích & Nhận Xét

1. **Ảnh hưởng của phương pháp Resampling**:
   - Under-sampling **cải thiện đáng kể recall** (khả năng phát hiện fraud)
   - Trade-off: Precision giảm (nhiều false positives hơn)
   - Phù hợp với fraud detection vì **miss fraud > false alarm**

2. **So sánh mô hình**:
   - Random Forest tốt hơn Logistic Regression
   - Các đặc trưng phi tuyến có ảnh hưởng lớn

---

## Cấu Trúc Dự Án

```
P4DS-LAB02/
│
├── data/                          # Dữ liệu
│   ├── raw/                       # Dữ liệu gốc
│   │   └── creditcard.csv
│   ├── processed/                 # Dữ liệu sau preprocessing
│   │   └── creditcard_processed.csv
│   └── final/                     # Dữ liệu train/test final
│       ├── original/              # Imbalanced data
│       ├── under_sampled/         # Under-sampled data
│       └── over_sampled/          # Over-sampled data
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # EDA và phân tích dữ liệu
│   ├── 02_preprocessing.ipynb     # Tiền xử lý dữ liệu
│   ├── 03_handling_imbalance.ipynb # Xử lý imbalanced data
│   └── 04_modeling.ipynb          # Training và evaluation
│
├── src/                           # Source code
│   ├── __init__.py
│   │
│   ├── data_preparation/          # Module xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── preprocessing/         # Preprocessing tools
│   │   │   ├── __init__.py
│   │   │   ├── cleaner.py        # Data cleaning
│   │   │   ├── outlier_handler.py # Outlier detection & handling
│   │   │   └── scaler.py         # Feature scaling methods
│   │   ├── resampling/           # Resampling techniques
│   │   │   ├── __init__.py
│   │   │   └── resampler.py      # Under/Over sampling
│   │   └── splitting/            # Data splitting
│   │       ├── __init__.py
│   │       └── data_splitter.py  # Train/test split
│   │
│   ├── models/                    # ML models (NumPy only)
│   │   ├── __init__.py
│   │   ├── logistic_regression.py # Logistic Regression
│   │   └── random_forest.py       # Random Forest
│   │
│   └──evaluator/                 # Model evaluation
│   │   ├── __init__.py
│   │   └── evaluator.py          # Metrics calculation
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── README.md                      # Tài liệu này
└── LICENSE                        # MIT License
```
---

## Thách Thức & Giải Pháp

### Thách Thức 1: NumPy

**Vấn đề:**
- Không có Scikit-learn, TensorFlow hay các framework ML
- Phải implement tất cả từ scratch
- Hiệu năng chậm với pure Python loops

**Giải pháp:**

1. **Vectorization với NumPy**:
   ```python
   # Chậm - Python loop
   for i in range(n):
       result[i] = sum(X[i] * w)
   
   # Nhanh - NumPy vectorization
   result = np.dot(X, w)
   ```

2. **Einstein Summation (einsum)**:
   ```python
   # Matrix multiplication
   np.einsum('ij,j->i', X, w)  # Faster than np.dot for some cases
   
   # Batch dot products
   np.einsum('ij,ij->i', A, B)
   ```

3. **Broadcasting**:
   ```python
   # Subtract mean from each column
   X_centered = X - np.mean(X, axis=0)  # Automatic broadcasting
   ```

4. **Pre-allocation**:
   ```python
   # Slow - append in loop
   result = []
   for x in X:
       result.append(predict(x))
   
   # Fast - pre-allocate
   result = np.empty(len(X))
   for i, x in enumerate(X):
       result[i] = predict(x)
   ```

### Thách Thức 2: Mất cân bằng lớp

**Vấn đề:**
- Mô hình bị thiên về lớp chiếm đa số
- Accuracy cao nhưng recall thấp
- Không phát hiện được fraud

**Giải pháp:**

1. **Phương pháp Resampling**:
   - Under-sampling: Giảm majority xuống --> Chọn under-sampling vì tránh overfitting
   - Over-sampling: Tăng minority lên

2. **Thang đo hợp lí**:
   - Không dùng accuracy
   - Tập trung vào Recall và PR-AUC


### Thách Thức 3: Numerical Stability

**Vấn đề:**
- `log(0)` → -inf
- `exp(large number)` → overflow
- Division by zero

**Giải pháp:**

1. **Clipping**:
   ```python
   # Avoid log(0)
   y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
   
   # Avoid exp overflow
   z = np.clip(z, -500, 500)
   ```

2. **Stable Sigmoid**:
   ```python
   def sigmoid(z):
       return np.where(
           z >= 0,
           1 / (1 + np.exp(-z)),
           np.exp(z) / (1 + np.exp(z))
       )
   ```

3. **Check for Zero Division**:
   ```python
   if denominator == 0:
       return 0.0
   return numerator / denominator
   ```

### Thách Thức 4: Memory Management

**Vấn đề:**
- Dữ liệu lớn
- Bị trùng lắp dữ liệu sau khi preprocess

**Giải pháp:**

1. **In-place Operations**:
   ```python
   # Creates new array
   X = X / std
   
   # Modifies in-place
   X /= std
   ```

2. **Generator Functions**:
   ```python
   def batch_generator(X, y, batch_size):
       for i in range(0, len(X), batch_size):
           yield X[i:i+batch_size], y[i:i+batch_size]
   ```
---

## Hướng Phát Triển

1. **Các kĩ thuật Resampling tốt hơn**
    - [ ] SMOTE (Synthetic Minority Over-sampling)
    - [ ] ADASYN (Adaptive Synthetic Sampling)
    - [ ] Combined methods (SMOTE + ENN)

2. **Mô hình mới**
    - [ ] Support Vector Machine (SVM)
    - [ ] Gradient Boosting (XGBoost-like)
    - [ ] Naive Bayes

3. **Tối ưu các tham số**
    - [ ] Grid Search
    - [ ] Random Search
    - [ ] Tối ưu Bayesian

4. **Feature Engineering**
    - [ ] Thời gian (hour, day, month)
    - [ ] Đặc trưng thống kê (rolling mean, std)

5. **Mô hình học sâu**
    - [ ] Mạng Neuron đơn giản
    - [ ] Hàm kích hoạt (ReLU, tanh)

---

## Contributors

### Thông Tin Tác Giả

- **Tên**: Nguyễn Thiên Ấn 
- **Trường**: Trường Đại học Khoa học Tự nhiên (HCMUS - VNUHCM)
- **Lớp**: CSC17104 - P4DS
---

### Liên Hệ

- **GitHub**: [@p1neapplechoco](https://github.com/p1neapplechoco)
- **Email**: ngthienaans@gmail.com

**Project Link**: [https://github.com/p1neapplechoco/FraudDetection-NumPy/](https://github.com/p1neapplechoco/FraudDetection-NumPy/)

---

## License

Được phân phối dưới **MIT License**. Xem [LICENSE](LICENSE) để biết thêm chi tiết.

```
MIT License

Copyright (c) 2025 Nguyen Thien An

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---
