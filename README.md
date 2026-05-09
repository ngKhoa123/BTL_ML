# Bài Tập Lớn Machine Learning

Bài tập lớn machine learning cho môn Học Máy tại Đại học Bách Khoa Thành phố Hồ Chí Minh, bao gồm hai bài toán: dự đoán khả năng rời bỏ của khách hàng và phân loại khối u não với BRISC.

---

## Thông Tin Môn Học

| Thông tin | Chi tiết |
|-----------|---------|
| **Tên môn học** | Học Máy |
| **Mã môn học** | CO3117 |
| **Học kỳ** | 252 |
| **Năm học** | 2025-2026 |
| **Trường** | Đại học Bách Khoa Thành phố Hồ Chí Minh (HCMUT) |

---

## Giảng Viên Hướng Dẫn

| Họ và Tên | Chức Danh |
|-----------|----------|
| TS. Trương Vĩnh Lân | Giảng viên hướng dẫn |

---

## Thông Tin Các Thành Viên Nhóm

| STT | Họ và Tên | MSSV | 
|-----|-----------|-----------------|
| 1 | Nguyễn Đăng Khoa | 2311614 | 
| 2 | Vũ Huy Gia Khang | 2311486 | 
| 3 | Nguyễn Văn Hiếu | 2310967 | 
| 4 | Dương Khôi Nguyên | 2312333| 
| 5 | Hồ Công Danh  | 2310410 | 


---

## Mục Tiêu Của Bài Tập Lớn

### Bài 1: Dự Đoán Khả Năng Rời Bỏ Của Khách Hàng 
- Xây dựng mô hình học máy để dự đoán khả năng khách hàng sẽ rời đi
- Áp dụng các kỹ thuật tiền xử lý và chuẩn hóa dữ liệu
- Huấn luyện và so sánh các mô hình phân loại khác nhau (Logistic Regression, Random Forest, XGBoost.)
- Đánh giá hiệu suất mô hình bằng các metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Tối ưu hóa hyperparameter để cải thiện performance

### Bài 3: Phân Loại Khối U Não
- Xây dựng mô hình học máy và học sâu để phân loại khối u não dựa trên ảnh MRI.
- Trích xuất đặc trưng bằng HOG, giảm chiều dữ liệu với PCA
- Huấn luyện và so sánh mô hình phân loại: HOG->PCA->SVM và EfficientnetB0
- Tối ưu hóa hyperparameter để cải thiện performance
---

## Cấu Trúc Thư Mục

```
BTL_ML/
├── notebooks/                          # Jupyter Notebooks
│   ├── Bài 1/
│   │   └── CHURN_PREDICTION.ipynb      # Notebook dự đoán churn
│   └── Bài 3/
│       └── BRISC.ipynb                 # Notebook phân tích BRISC
│
├── modules/                            # Các module Python
│   ├── preprocessing.py                # Tiền xử lý dữ liệu
│   ├── feature_extraction.py           # Trích xuất đặc trưng
│   ├── model_evaluation.py             # Đánh giá mô hình
│   ├── preprocessing_utils.py          # Các utility cho tiền xử lý
│   └── utils.py                        # Các hàm tiện ích chung
│
├── features/                           # Dữ liệu đặc trưng đã trích xuất
│   ├── Bài 1/
│   │   └── xgboost/                    # Đặc trưng cho bài 1
│   │       ├── X_train_features.npy
│   │       └── X_test_features.npy
│   └── Bài 3/
│       ├── EfficientNet/               # Đặc trưng từ EfficientNet
│       │   ├── train_embeddings.npy
│       │   ├── test_embeddings.npy
│       │   ├── train_labels.npy
│       │   └── test_labels.npy
│       └── SVM/                        # Đặc trưng PCA cho SVM
│           ├── X_train_pca.npy
│           └── X_test_pca.npy
│
├── model/                              # Các mô hình đã huấn luyện
│
├── requirements.txt                    # Danh sách các thư viện phụ thuộc
├── README.md                          
└── .gitignore

```

---

## Hướng Dẫn Cài Đặt và Chạy

### Yêu Cầu Hệ Thống

- **Python**: >= 3.8
- **Pip** hoặc **Conda** để quản lý package
- **Jupyter Notebook** để chạy các notebook

### Các Thư Viện Chính

- `numpy`: Xử lý mảng số
- `pandas`: Xử lý dữ liệu bảng
- `scikit-learn`: Các thuật toán machine learning
- `scikit-image`: Xử lý hình ảnh
- `opencv-python`: Thư viện xử lý hình ảnh
- `pillow`: Xử lý hình ảnh
- `jupyter`: Jupyter Notebook
- `matplotlib`, `seaborn`: Trực quan hóa dữ liệu
- `joblib`: Lưu và tải mô hình

### Cài Đặt

#### 1. Clone Repository
```bash
git clone <repository-url>
cd BTL_ML
```

#### 2. Tạo Môi Trường Ảo

**Với Conda:**
```bash
conda create -n ml-env python=3.9
conda activate ml-env
```

**Hoặc với venv:**
```bash
# Trên Windows
python -m venv venv
venv\Scripts\activate

# Trên Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

#### 3. Cài Đặt Các Thư Viện Phụ Thuộc
```bash
pip install -r requirements.txt
```

#### 4. Khởi Động Jupyter Notebook
```bash
jupyter notebook
```

Sau đó truy cập vào thư mục `notebooks/` và chọn notebook cần chạy:
- `Bài 1/CHURN_PREDICTION.ipynb` - Dự đoán churn
- `Bài 3/BRISC.ipynb` - Phân tích BRISC


---



## Liên Kết 

- **Báo cáo PDF**: [Thêm link báo cáo nếu có]
- **Google Colab**: 
  - [Bài 1 - Churn](https://colab.research.google.com/drive/16Gsz3SS9Dtbx7_fw6DTYm7pNJBPKMIlj?usp=sharing)
  - [Bài 3 - BRISC](https://colab.research.google.com/drive/17CoV9pgAfIIGcKIQoRSwJmpWJIk7UlmC?usp=sharing)

---

