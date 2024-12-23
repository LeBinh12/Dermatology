from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sys
import io

# Đảm bảo hỗ trợ Unicode trên Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Import và xử lý dữ liệu
def load_data():
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv('dermatology.csv', header=None)

    # Đặt tên cột
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

    # Chuyển đổi cột nhãn 'lop' sang số nguyên
    labelencoder = LabelEncoder()
    data['lop'] = labelencoder.fit_transform(data['lop'])

    return data

# Huấn luyện mô hình Random Forest
def train_model(data):
    X = data.iloc[:, :-1].values  # Tất cả cột trừ cột cuối
    Y = data.iloc[:, -1].values   # Cột cuối (lop)

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=140)

    # Khởi tạo mô hình Random Forest
    model = RandomForestClassifier(
        criterion="gini", 
        random_state=100, 
        n_estimators=100, 
        max_depth=10, 
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)

    # Đánh giá mô hình
    y_pred = model.predict(X_test)
    print("Độ chính xác: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("Ma trận nhầm lẫn:\n", confusion_matrix(y_test, y_pred))
    print("Báo cáo phân loại:\n", classification_report(y_test, y_pred))

    return model

# Tải dữ liệu và huấn luyện mô hình
data = load_data()
model = train_model(data)

# Xây dựng route chính cho ứng dụng web
@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class = None

    # Ánh xạ số sang tên bệnh
    disease_mapping = {
        1: "Bệnh vẩy nến",
        2: "Viêm da tiết bã nhờn",
        3: "Bệnh liken phẳng",
        4: "Bệnh vảy phấn hồng",
        5: "Viêm da mãn tính",
        6: "Bệnh vảy nến đỏ nang lông"
    }

    if request.method == 'POST':
        # Lấy dữ liệu từ form
        user_input = []
        for feature in data.columns[:-1]:  # Bỏ cột 'lop'
            user_input.append(float(request.form.get(feature, 0)))
        # Kiểm tra nếu tất cả dữ liệu đều là 0
        if all(value == 0 for value in user_input):
            predicted_class = "Không có bệnh"
        else:
            # Chuyển đổi dữ liệu nhập vào thành định dạng phù hợp
            user_input = np.array(user_input).reshape(1, -1)

            # Dự đoán
            predicted_class = model.predict(user_input)[0]
            predicted_class = int(predicted_class) + 1

            # Chuyển số thành tên bệnh
            predicted_class = disease_mapping.get(predicted_class, "Không xác định")

    return render_template('index.html', columns=data.columns[:-1], predicted_class=predicted_class)

if __name__ == "__main__":
    app.run(debug=True)
