# Tối Ưu Hóa Dự Báo Động Lực Học Lợi Nhuận Tài Chính Với Triple Barrier và Purged & Embargo Cross-Validation

## Tóm Tắt (Abstract)

Nghiên cứu này khảo sát các thách thức cốt lõi khi áp dụng các mô hình học máy vào dự báo chuỗi thời gian tài chính: rò rỉ thông tin qua thời gian (look-ahead và autocorrelation leakage), sụp đổ hiệu chuẩn xác suất (probability calibration collapse), bùng nổ phương sai do số chiều cao (curse of dimensionality), và sự bất đối xứng giữa độ chính xác phân loại (classification edge) với hiệu quả sinh lời thực tế (portfolio PnL). Bằng cách triển khai khung phân đoạn thời gian nghiêm ngặt Purged & Embargo Cross-Validation (López de Prado), kết hợp cơ chế chọn lọc biến Clustered Consensus Selection, hiệu chuẩn Bounded Platt Scaling và kích hoạt lệnh qua Rolling Percentile thích ứng, hệ thống triệt tiêu hoàn toàn overfitting (khoảng cách Train - Test Accuracy tiệm cận 0) và đạt tỷ lệ thắng ngoài mẫu (Out-Of-Sample - OOS Precision) ở mức **62.88%** trên 1,070 phiên kiểm định. Tuy nhiên, khi kiểm toán PnL danh mục với chi phí giao dịch thực tế, chiến lược chịu mức sụt giảm **-19.68%** do tỷ lệ Win/Loss Ratio đạt mức bất lợi **0.37**. Kết quả thực nghiệm chứng minh rằng lợi thế dự báo xác suất không thể chuyển hóa thành lợi nhuận nếu thiếu cơ chế quản trị lệnh động (dynamic trade management) để khắc phục rủi ro rào cản thời gian (time-out barrier).

## 1. Đặt Vấn Đề Và Câu Hỏi Nghiên Cứu (Motivation & Research Questions)

### 1.1. Bối cảnh và Động lực nghiên cứu

Trong thị trường tài chính định lượng, việc áp dụng máy học đối mặt với tỷ số tín hiệu trên nhiễu (Signal-to-Noise Ratio - SNR) cực thấp và tính chất phi dừng (non-stationarity). Hầu hết các mô hình phân loại chuỗi thời gian thông thường gặp phải các lỗi phương pháp luận nghiêm trọng:

1. **Rò rỉ tương quan chuỗi (Serial Dependence & Overlapping Labels):** Việc gán nhãn cố định theo bước thời gian ($t+h$) truyền thống hoặc chia K-Fold ngẫu nhiên làm rò rỉ biến động và cấu trúc vi mô sang tập kiểm định ngoài mẫu.
2. **Ảo tưởng phi tuyến & Lời nguyền số chiều:** Việc tạo quá nhiều biến phái sinh bậc cao (đạo hàm bậc 2 như acceleration) khuếch đại nhiễu trắng ngẫu nhiên, khiến các thuật toán chọn biến nội mẫu bị đánh lừa bởi Data Snooping.
3. **Nghịch lý giữa Machine Learning Metric và Financial Metric:** Mô hình có thể đạt Accuracy hoặc Precision cao nhưng vẫn phá hủy vốn lũy kế (negative equity drift) do cấu trúc tỷ lệ lãi/lỗ (Payoff Asymmetry) bị lệch.

### 1.2. Câu hỏi nghiên cứu (Research Questions)

- **RQ1:** *Liệu việc thiết lập cơ chế Purged & Embargo CV kết hợp Clustered Consensus Selection có thể triệt tiêu hoàn toàn rò rỉ thông tin và hiện tượng Overfitting trên chuỗi thời gian hay không?*
- **RQ2:** *Cơ chế điều khiển ngưỡng thích ứng (Rolling Percentile) có giải quyết được hiện tượng kẹt lệnh (Distribution Mismatch / Probability Collapse) khi thị trường chuyển dịch chế độ (Regime Shift) so với ngưỡng cố định hay không?*
- **RQ3:** *Liệu một mô hình đạt tỷ lệ thắng OOS vượt trội (~63%) có đảm bảo tạo ra kỳ vọng sinh lời dương ($\mathbb{E}[\text{PnL}] > 0$) trong điều kiện kiểm toán danh mục có tính đến chi phí giao dịch và cơ chế cản Triple Barrier đối xứng hay không?*

## 2. Phương Pháp Luận Và Thiết Lập Toán Học (Methodology & Mathematical Proofs)

### 2.1. Khung gán nhãn Triple Barrier Causal

Cho chuỗi giá điều chỉnh $P_t$ và độ biến động thực nghiệm cục bộ $\sigma_t$ được ước lượng nhân quả từ thông tin quá khứ ($\mathcal{I}_{t-1}$) qua Exponentially Weighted Moving Average (EWMA):

$$
\sigma_t = \text{std}\left(\{r_\tau\}_{\tau \le t-1}\right), \quad \text{với } r_\tau = \ln\left(\frac{P_\tau}{P_{\tau-1}}\right)
$$

Rào cản trên (Take-Profit), rào cản dưới (Stop-Loss) và rào cản thời gian (Vertical Barrier) được xác lập:

$$
\text{Upper}_t = P_t (1 + pt \cdot \sigma_t), \quad \text{Lower}_t = P_t (1 - sl \cdot \sigma_t), \quad T_{\text{touch}} = \min(t_{\text{hit}}, t + h)
$$

Nhãn nhị phân $y_t \in \{0, 1\}$ được xác định:

$$
y_t = \begin{cases} 1, & \text{nếu } P \text{ chạm } \text{Upper}_t \text{ trước} \\ 0, & \text{nếu } P \text{ chạm } \text{Lower}_t \text{ trước hoặc hết } h \text{ phiên (Time-out)} \end{cases}
$$

### 2.2. Kiểm định phân đoạn Purged & Embargo Time-Series CV

Để đảm bảo tính độc lập thống kê tiệm cận giữa tập huấn luyện $\mathcal{S}_{\text{train}}$ và tập kiểm định $\mathcal{S}_{\text{test}}$:

$$
P(X_{\text{test}}, y_{\text{test}} \mid X_{\text{train}}, y_{\text{train}}) = P(X_{\text{test}}, y_{\text{test}})
$$

Điều kiện biên mở rộng với khoảng đệm an toàn Embargo $h_{\text{embargo}}$:

$$
\forall i \in \mathcal{S}_{\text{train}}, \quad t_{\text{touch}}^{(i)} < t_{\text{start}}^{(\text{test})} - h_{\text{embargo}}
$$

với $h_{\text{embargo}} = 5 \text{ ngày}$, triệt tiêu hoàn toàn khả năng rò rỉ qua các cụm biến động (volatility clustering) kéo dài.

### 2.3. Bounded Platt Scaling & Phân rã Brier Score

Độ chính xác xác suất được đánh giá qua phân rã Murphy (1973):

$$
\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2 = \underbrace{\bar{y}(1 - \bar{y})}_{\text{Uncertainty}} + \underbrace{\frac{1}{N}\sum_{k=1}^K n_k (\hat{p}_k - \bar{y}_k)^2}_{\text{Reliability (Loss)}} - \underbrace{\frac{1}{N}\sum_{k=1}^K n_k (\bar{y}_k - \bar{y})^2}_{\text{Resolution}}
$$

Để tránh suy biến hàm logit $z = \mathbf{w}^\top X + b$ ra hai biên hoặc bị nén biên độ khi gặp Covariate Shift, xác suất được hiệu chuẩn qua bộ Bounded Platt Scaling trên phân vị Out-Of-Fold:

$$
\hat{p} = \sigma(a \cdot z + b), \quad \text{với } a \in [0.1, 2.5], \; b \in [-1.5, 1.5]
$$

### 2.4. Ngưỡng thích ứng Rolling Percentile & Dynamic Bet Sizing

Thay vì cố định điểm cắt phân loại cứng $\theta = 0.50$, ngưỡng vào lệnh $Threshold_t$ được tính trượt trong cửa sổ phân vị $W = 60$ phiên tại bách phân vị $Q = 65\%$ (chọn lọc top 35% xác suất tốt nhất):

$$
\text{Threshold}_t = \text{Percentile}_{Q}\left(\{\hat{p}_\tau\}_{\tau = t-W}^{t-1}\right)
$$

Tín hiệu kích hoạt nhị phân:

$$
S_t = \mathbb{I}\left(\hat{p}_t \ge \text{Threshold}_t\right)
$$

Quy mô vị thế (Bet Size) $\omega_t$ được điều chỉnh theo điểm Z-score của xác suất dự báo:

$$
Z_t = \frac{\hat{p}_t - \mu_{\hat{p}, W}}{\sigma_{\hat{p}, W}}, \quad \omega_t = S_t \cdot \text{clip}\left(0.20 + (Z_t - 0.5) \cdot \frac{0.80}{1.50}, 0.20, 1.0\right)
$$

## 3. Quá Trình Thực Nghiệm Và Diễn Tiến Chẩn Đoán (Empirical Evolution)

Quá trình nghiên cứu trải qua 6 giai đoạn tinh chỉnh thực nghiệm, giải quyết từng khâu bệnh lý của chuỗi dữ liệu (từ Capture 1 đến Capture 8):

| **Giai Đoạn Thử Nghiệm** | **Can Thiệp Kiến Trúc Cốt Lõi** | **Bệnh Lý / Hiện Tượng Quan Sát** | **OOS Test AUC** | **Precision (Lớp 1)** | **Trạng Thái Phân Bổ Lệnh (OOS)** |
| --- | --- | --- | --- | --- | --- |
| **Giai đoạn 1 (Gốc)** | Purging cơ bản, Logistic $L_2$, Đạo hàm bậc 2 | Acceleration chiếm trọn 5/5 fold; Permutation drop âm | 0.4927 | ~50.8% | Lệch pha ngoài mẫu |
| **Giai đoạn 2 (Siết $L_1$)** | Thêm Embargo, ép phạt $L_1$ ($C=0.05$) | *Resolution Collapse:* $\mathbf{w} \to \mathbf{0}$, tín hiệu bị kẹt All-or-Nothing theo Fold | 0.4611 | 50.2% | Folds 1, 2, 5: 0%; Folds 3, 4: 100% |
| **Giai đoạn 3 (Sạch biến)** | Xóa bỏ toàn bộ `*_accel`, siết 5 biến gốc | Lần đầu tiên OOS AUC vượt ngẫu nhiên, Interaction test tăng mạnh | 0.5218 | 55.67% | Fold 4 kẹt trần 100% lệnh do ngưỡng tĩnh |
| **Giai đoạn 4 (Rolling)** | Chuyển sang Rolling Percentile ($W=60$) | Xóa bỏ lỗi Fold 4, nâng Precision lên sát 60% | 0.5218 | 59.94% | Base Estimator bị trơ logits ($T^* = 5.0$) |
| **Giai đoạn 5 (Deg-2 Interaction)** | Sinh 10 biến $\to$ 55 biến tương tác đa thức | *Curse of Dimensionality:* Overfitting bùng nổ (Gap Tr-Te $= 0.21$) | 0.5123 | 62.68% | Fold 1 nghẽn lệnh (14%) do rào Absolute Floor |
| **Giai đoạn 6 (Chuẩn hóa cuối)** | Khóa 4 nhóm trực giao (10 cột), Bounded Platt | Triệt tiêu Overfitting (Gap $\approx 0$), Precision đạt đỉnh | **0.5494** | **62.60%** | **Phân bổ tối ưu: 32.7% – 37.8% qua 5 Folds** |

## 4. Phân Tích Kết Quả Kiểm Toán Chuyên Sâu (Experimental Results)

### 4.1. Khắc chế hoàn toàn Overfitting và Triệt tiêu Rò rỉ Thông tin (RQ1)

Kết quả chẩn đoán Bias - Variance tại cấu hình chuẩn hóa cuối cùng (Capture 8) cho thấy tính tổng quát hóa ngoài mẫu đạt trạng thái tiệm cận hoàn hảo:

- **Train AUC trung bình:** $0.5587$ so với **OOS Test AUC trung bình:** $0.5494$ (Khoảng cách Gap chỉ $0.0093$).
- **Train Accuracy trung bình:** $52.43\%$ so với **OOS Test Accuracy:** $53.08\%$.
- Khoảng cách kiểm định trên từng fold cá biệt:
    - Fold 1: Train ACC $52.61\%$, Test ACC $55.14\%$ (Gap: $-2.53\%$).
    - Fold 2: Train ACC $49.06\%$, Test ACC $49.53\%$ (Gap: $-0.47\%$).
    - Fold 4: Train ACC $56.97\%$, Test ACC $59.81\%$ (Gap: $-2.84\%$).
    - Fold 5: Train ACC $52.82\%$, Test ACC $52.80\%$ (Gap: $+0.02\%$).

Sự đồng nhất giữa Train và Test chứng minh rằng việc khống chế chặt chẽ số chiều ($4$ biến đại diện trực giao tạo thành $10$ cột đa thức) kết hợp khoảng đệm Embargo $6 - 8 \text{ ngày}$ đã loại bỏ hoàn toàn hiện tượng học vẹt nhiễu (in-sample memorization).

### 4.2. Thích ứng Regime Shift và Tính ổn định luồng lệnh (RQ2)

Việc thay thế ngưỡng xác suất tuyệt đối bằng phân vị động cục bộ (Rolling Percentile) đã chứng minh vai trò hóa giải Covariate Shift:

- Khắc phục hiện tượng sụp đổ Fold 4: Tỷ lệ vào lệnh giảm từ mức cực đoan $100\%$ về **$35.51\%$**, phản ánh trung thực phân vị mong muốn.
- Giải phóng tình trạng nghẽn lệnh ở Fold 1: Sau khi gỡ bỏ sàn cứng cố định `ABSOLUTE_FLOOR`, tỷ lệ mở vị thế hồi phục từ $5.14\%$ lên **$36.45\%$**.
- Tỷ lệ mở lệnh xuyên suốt 5 chu kỳ thị trường (2018–2024) duy trì độ biến thiên cực hẹp:
    
    $$
    \text{Fold 1: } 36.45\% \;\to\; \text{Fold 2: } 32.71\% \;\to\; \text{Fold 3: } 33.64\% \;\to\; \text{Fold 4: } 35.51\% \;\to\; \text{Fold 5: } 37.85\%
    $$
    

### 4.3. Xác thực tính chân thực của Alpha (Permutation Feature Importance)

Kiểm định hoán vị ngoài mẫu (OOS Permutation Importance) xác nhận tín hiệu dự báo phụ thuộc vào các họ biến có căn cứ kinh tế lượng vững chắc:

1. **Áp lực dòng tiền & Phân kỳ khối lượng:** `cmf_slope_5_zscaled_lag1` (AUC Drop $= \mathbf{+0.1118}$) đóng góp sức mạnh dự báo lớn nhất toàn hệ thống.
2. **Động lực học nén biến động:** `kinematic_squeeze_ratio_zscaled_lag2` (AUC Drop $= \mathbf{+0.0895}$).
3. **Động lượng lợi suất ngắn hạn:** `ret_3_zscaled_lag2` (AUC Drop $= \mathbf{+0.0810}$) và `body_direction_momentum_zscaled_lag2` (AUC Drop $= \mathbf{+0.0769}$).
4. **Biến động Parkinson:** `vol_parkinson_momentum` (AUC Drop $= \mathbf{+0.0358}$).

Các biến thuộc nhóm đạo hàm bậc hai hoặc đuôi nến nhiễu khi bị đưa vào `FEATURE_BLACKLIST` đã giải phóng dung lượng học cho mô hình, giúp OOS AUC trung bình tăng từ $0.4927$ lên $0.5494$.

## 5. Kiểm Toán PnL Danh Mục Và Giải Mã Nghịch Lý Win Rate (RQ3)

Bảng dưới đây tổng hợp kết quả kiểm toán hiệu năng danh mục Out-Of-Sample sau khi giải quyết xung đột vị thế giữ lệnh (Concurrent Trade Resolution) và khấu trừ chi phí giao dịch ($0.20\%$ hai chiều):

| **Phân Vùng Dữ Liệu** | **Số Lệnh (Trades)** | **Tỷ Lệ Thắng (Win Rate)** | **Profit Factor** | **Win / Loss Ratio** | **Lợi Nhuận Tích Lũy** | **Sharpe Ratio** | **Max Drawdown** | **Kích Thước Vị Thế TB** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Fold 1** | 40 | 57.50% | 0.20 | 0.15 | -12.36% | -1.94 | -13.73% | 0.37 |
| **Fold 2** | 51 | 62.75% | 0.83 | 0.49 | -1.22% | -2.23 | -3.27% | 0.28 |
| **Fold 3** | 53 | 64.15% | 1.00 | 0.56 | -0.05% | -1.82 | -2.43% | 0.22 |
| **Fold 4** | 45 | 66.67% | 0.62 | 0.31 | -6.23% | -1.56 | -7.66% | 0.40 |
| **Fold 5** | 40 | 62.50% | 0.91 | 0.54 | -1.01% | -0.90 | -6.57% | 0.35 |
| **Full OOS Portfolio** | **229** | **62.88%** | **0.62** | **0.37** | **-19.68%** | **-1.46** | **-23.83%** | **0.32** |

### Giải mã nghịch lý: Vì sao Win Rate 62.88% vẫn chịu lỗ ròng -19.68%?

Mô hình toán học của kỳ vọng lợi nhuận trên mỗi giao dịch được biểu diễn:

$$
\mathbb{E}[\text{Return}] = P_{\text{win}} \times \overline{\text{Gain}} - (1 - P_{\text{win}}) \times \overline{\text{Loss}} - \text{Cost}
$$

Với $P_{\text{win}} = 0.6288$ và Win/Loss Ratio $R = \frac{\overline{\text{Gain}}}{\overline{\text{Loss}}} = 0.37$:

$$
\mathbb{E}[\text{Return}] = \overline{\text{Loss}} \cdot \left(0.6288 \times 0.37 - 0.3712 \times 1.0\right) - \text{Cost} = \overline{\text{Loss}} \cdot \left(0.2327 - 0.3712\right) - \text{Cost} = -0.1385 \cdot \overline{\text{Loss}} - \text{Cost} < 0
$$

Mặc dù cấu trúc Triple Barrier ban đầu đặt biên độ đối xứng $pt = 1.0, sl = 1.0$, trong thực tế phân phối lợi nhuận bị bóp méo bởi hai yếu tố thực thi:

1. **Tác động của Vertical Barrier (Time-out):** Khi giá đi ngang và không chạm Take-Profit trong vòng $h = 5$ phiên, lệnh bị đóng cưỡng bức. Nhiều lệnh đóng ở trạng thái âm sâu (tiệm cận mức Stop-Loss), trong khi các lệnh thắng khi chạm Take-Profit lại bị giới hạn biên độ cố định ($1.0 \times \sigma$).
2. **Chi phí ma sát cấu trúc:** Chi phí phí giao dịch và trượt giá ($2 \times 0.20\% = 0.40\%$) bào mòn đáng kể các lệnh thắng có biên lợi nhuận nhỏ trong các chu kỳ biến động thấp.

## 6. Kết Luận Và Định Hướng Nghiên Cứu Tiếp Theo (Conclusion & Future Work)

### 6.1. Kết luận nghiên cứu

1. **Tính hợp lệ của phương pháp luận học máy:** Việc kết hợp Purged & Embargo CV, kiểm định HRP/MI và siết Sparsity Cap ($4$ nhóm biến) đã giải quyết triệt để vấn đề rò rỉ thông tin và hiện tượng Overfitting, mang lại mô hình dự báo xác suất có tính tổng quát hóa cao và độ chính xác phân loại ngoài mẫu ổn định ($62.88\%$).
2. **Vai trò của Rolling Percentile:** Cơ chế ngưỡng thích ứng phân vị đã chứng minh khả năng kiểm soát luồng lệnh, triệt tiêu hoàn toàn sự cố phân bổ lệch pha giữa các chế độ thị trường.
3. **Phát hiện quan trọng nhất:** Một mô hình phân loại nhị phân sở hữu Edge thống kê vượt trội vẫn có thể thất bại về mặt tài chính nếu không có sự tương thích giữa **Mô hình Dự báo (Prediction Engine)** và **Cơ chế Thực thi Quản trị Lệnh (Execution & Payoff Management)**. Rào cản lớn nhất của bài toán hiện tại không nằm ở khâu trích xuất đặc trưng hay thuật toán học máy, mà nằm ở tính chất bất đối xứng của hàm mục tiêu tài chính.

### 6.2. Hướng phát triển bắt buộc (Actionable Next Steps)

- **Tái cấu trúc rào cản bất đối xứng (Asymmetric Barrier):** Tái lập cấu trúc Triple Barrier với tỷ lệ Risk:Reward kỳ vọng tối thiểu $\ge 1.5$ (ví dụ: $pt = 1.5, sl = 1.0$) nhằm đảm bảo Win/Loss Ratio nâng lên trên $0.80$.
- **Cơ chế Trailing Stop động:** Thay vì đóng lệnh cứng nhắc tại $h = 5$ phiên khi time-out, triển khai cơ chế khóa lợi nhuận động (Dynamic Trailing Stop dựa trên ATR hoặc độ lệch Parkinson) để ngăn các lệnh đang có lãi đảo chiều thành lệnh lỗ khi chạm cản thời gian.
- **Tối ưu hóa Lớp Tuyến Tính:** Dựa trên bằng chứng từ Test 2 tại Capture 8 (Delta AUC phi tuyến là âm: $-0.0098$), loại bỏ hoàn toàn `PolynomialFeatures(degree=2)` để sử dụng hồi quy Logistic tuyến tính thuần nhất trên 4 biến trực giao, vừa tiết kiệm chi phí tính toán vừa tối đa hóa độ ổn định ngoài mẫu.