import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Đọc và xử lý dữ liệu
def load_data():
    data = pd.read_csv('dermatology.csv', header=None)
    data.columns = [
        'ban_do', 'bong_vay', 'ranh_gioi_ro_rang', 'ngua', 'hien_tuong_koebner',
        'sang_dinh_da_giac', 'sang_dinh_dinh_nang_long', 'ton_thuong_niem_mac_mieng',
        'ton_thuong_dau_goi_va_khuu_tay', 'ton_thuong_da_dau', 'tien_su_gia_dinh',
        'mat_sac_to', 'bach_cau_eosin_trong_xam_nhap', 'xam_nhap_pnl',
        'xo_soi_dermis', 'xuat_bao', 'tang_san_bao', 'tang_san_sung',
        'sung_mo', 'sung_bong', 'keo_dai_rete', 'mong_lop_suprapapillary',
        'mu', 'apxe_vi_munro', 'sung_co_cuc_bo', 'mat_lop_hat',
        'ton_thuong_bao_tang', 'ton_thuong_nang', 'xam_nhap_bong',
        'nut_sung_dinh_sang', 'nut_sung_xung_quanh', 'viem_da_xam',
        'xam_nhap_dai', 'tuoi', 'lop'
    ]

    # Kiểm tra và loại bỏ các giá trị không hợp lệ
    data = data.apply(pd.to_numeric, errors='coerce')  # Chuyển các giá trị không phải số thành NaN
    data = data.dropna()  # Loại bỏ các hàng có giá trị NaN

    # Chuyển đổi cột nhãn 'class' sang số nguyên
    labelencoder = LabelEncoder()
    data['lop'] = labelencoder.fit_transform(data['lop'])

    return data

# Huấn luyện mô hình Random Forest
def train_model(data):
    X = data.iloc[:, :-1].values  # Tất cả cột trừ cột cuối
    Y = data.iloc[:, -1].values   # Cột cuối (class)

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=140)

    # Khởi tạo mô hình Random Forest
    model = RandomForestClassifier(criterion="gini", random_state=100, n_estimators=100, max_depth=10, min_samples_leaf=2)
    model.fit(X_train, y_train)

    return model

# Tải dữ liệu và huấn luyện mô hình
data = load_data()
model = train_model(data)

# Dữ liệu cần kiểm tra
test_data = [
    [2, 2, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 55],  # Class 2
    [3, 3, 3, 2, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 2, 0, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 8],   # Class 1
    [2, 1, 2, 3, 1, 3, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 3, 2, 0, 0, 2, 3, 26],  # Class 3
    [2, 2, 2, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 3, 0, 0, 2, 0, 3, 2, 2, 2, 2, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 40]   # Class 1
]

# Chạy dự đoán trên các dòng test
for i, data_point in enumerate(test_data):
    data_point = np.array(data_point).reshape(1, -1)  # Chuyển thành định dạng đúng
    prediction = model.predict(data_point)[0]  # Dự đoán class
    print(f"Dữ liệu test {i+1}: {data_point.flatten()} -> Kết quả dự đoán (class): {prediction + 1}")
