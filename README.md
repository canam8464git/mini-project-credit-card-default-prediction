# mini-project-credit-card-default-prediction
Phân loại vỡ nợ thẻ tín dụng (dữ liệu UCI 30.000 dòng, 23 biến)
# Dự đoán vỡ nợ thẻ tín dụng (Credit Card Default Classification)

## Tổng quan

Trong bối cảnh quản trị rủi ro tín dụng, việc nhận diện sớm khách hàng có nguy cơ vỡ nợ ở kỳ thanh toán tiếp theo (`Y = 1`) giúp giảm tổn thất nợ xấu. Dự án được thực hiện trên bộ dữ liệu **Default of Credit Card Clients** gồm `30.000` hồ sơ, `23` biến đặc trưng (`X1`–`X23`) và nhãn nhị phân `Y` bằng Python trên Google Colab, sử dụng các thư viện `pandas`, `numpy`, `scikit-learn`, `matplotlib` và `ucimlrepo`.

## Quy trình thực hiện

- **Kiểm tra và làm sạch dữ liệu:** Tải dữ liệu qua `ucimlrepo`, kiểm tra giá trị thiếu, phát hiện `35` dòng trùng lặp và rà soát miền giá trị của các nhóm biến (biến phân loại `X2`–`X4`, trạng thái thanh toán `X6`–`X11`, dư nợ sao kê `X12`–`X17` và số tiền thanh toán `X18`–`X23`).
- **Khám phá dữ liệu (EDA):** Phân tích phân bố tuổi (`X5`), phân bố nhãn `Y` và số tháng trễ hạn nhằm nắm rõ mức độ mất cân bằng lớp và hành vi trễ hạn của khách hàng.
- **Tạo đặc trưng, so sánh và huấn luyện mô hình:** Xây dựng đặc trưng theo nghiệp vụ tín dụng (`Feature Engineering`), chia tập train/test, chuẩn hóa dữ liệu, đối chiếu hiệu quả giữa các mô hình dự đoán khác nhau để ra mô hình phù hợp là `HistGradientBoosting` (sau `Feature Engineering`) với ngưỡng xác suất tối ưu `0,55` theo `F1-score` của lớp vỡ nợ.
- **Đánh giá và demo dự đoán:** Đánh giá mô hình chốt trên tập kiểm thử bằng `PR-AUC` và ma trận nhầm lẫn; xây dựng hàm `demo_predict_default` để kiểm tra dự đoán trên cả hồ sơ bình thường và hồ sơ vỡ nợ thực tế.

## So sánh mô hình và lựa chọn ngưỡng

- Dự án đối chiếu các phương án mô hình và thiết lập (trước/sau `Feature Engineering` và các mô hình phân loại đối chứng trên cùng bộ dữ liệu) để chọn mô hình phù hợp nhất cho bài toán mất cân bằng lớp của rủi ro tín dụng.
- **`HistGradientBoosting` sau `Feature Engineering`** được chọn làm mô hình chính nhờ khả năng nắm bắt quan hệ phi tuyến và tương tác giữa các đặc trưng hành vi thanh toán/dư nợ tốt hơn, kết hợp tinh chỉnh ngưỡng xác suất **`0,55`** thay vì ngưỡng mặc định `0,50` để cân bằng giữa việc phát hiện khách hàng vỡ nợ và hạn chế cảnh báo nhầm.

## Kết quả

- **PR-AUC** trên lớp vỡ nợ (`Y = 1`) của tập kiểm thử đạt **`0,5544`**.
- **Ma trận nhầm lẫn tại ngưỡng `0,55`:** `TN = 3.926`, `FP = 747`, `FN = 560`, `TP = 767` — tương ứng **Recall ≈ `0,578`** và **Precision ≈ `0,507`** cho lớp vỡ nợ (bắt được gần `58%` số ca vỡ nợ thực tế và khoảng một nửa số cảnh báo là chính xác).
- **Demo dự đoán:** Chạy thử trên các hồ sơ mẫu, bao gồm cả ca dự đoán đúng và ca bị bỏ sót (`False Negative`), giúp minh họa trực quan sự đánh đổi khi chọn ngưỡng quyết định.

## Cách chạy lại

Mở file `Default_of_credit_card_clients.ipynb` trong repository trên Google Colab, sau đó chạy tuần tự các cell từ trên xuống dưới.
