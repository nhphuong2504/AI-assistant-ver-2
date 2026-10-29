# Technical review and resume evidence

**Update:** The subsequent [decision-support extension](DECISION_REVIEW.md) adds development-selected classifiers, budgeted policy evaluation, spending challengers and current resume bullets. This file preserves the initial review and Cox baseline results; use the extension for the latest project positioning.

Reviewed 2026-10-29, from implementation and locally reproduced results. Perspective: Senior Data Scientist / Hiring Manager. Canonical metrics: `evaluation.json`; per-customer prediction CSVs allow independent recalculation.

## Assessment

Project có câu hỏi kinh doanh rõ ràng và tích hợp được nhiều bước: giao dịch → đặc trưng khách hàng → mô hình → API → giao diện hội thoại. Điểm mạnh nhất sau review là thiết kế đánh giá theo thời gian, sửa leakage và phân biệt ranking với calibration. Không nên giới thiệu đây là hệ thống tối ưu doanh thu retention đã được chứng minh hoặc production-ready.

Implementation ban đầu có nhiều thành phần hữu ích nhưng chưa đủ bằng chứng để bảo vệ các tuyên bố về độ chính xác. Benchmark mới cho thấy churn ranking có giá trị dự báo vượt baseline. CLV có giá trị xếp hạng, song còn yếu ở dự báo tổng doanh thu. Hai kết luận này cần được trình bày cùng nhau trong phỏng vấn.

## Những gì đã làm tốt

- Chọn đúng bài toán non-contractual retail: mua lặp lại, giá trị khách hàng và inactivity. BG/NBD + Gamma-Gamma là lựa chọn có cơ sở, không cần thêm deep learning để tăng độ phức tạp.
- Có phân biệt raw/audit transactions và clean transactions; tổng hợp invoice trước khi tính đặc trưng, tránh đếm số dòng sản phẩm thành số đơn hàng.
- Có mô hình cơ sở Kaplan-Meier, Cox, RFM và mô phỏng BG/NBD; có ý thức phân biệt relative risk với probability.
- Tách ETL, analytics, data cache và API; LLM gọi công cụ tính toán thay vì tự thực hiện mọi phép tính trong câu trả lời.
- Giao diện hoạt động như một lớp truy cập analytics. Giá trị Data Science nằm ở công cụ và validation phía sau, không chỉ ở tích hợp LLM.

## Các vấn đề quan trọng và xử lý

| Vấn đề trong implementation ban đầu | Vì sao quan trọng | Thay đổi thực hiện |
|---|---|---|
| CLV được scale theo doanh thu/số mua thực tế của chính kỳ test trong script calibration; serving dùng các hệ số 1.1/1.7 không có nguồn calibration độc lập | Tổng dự báo gần actual sau scaling không chứng minh khả năng dự báo; còn trộn doanh thu khách mới với cohort khách cũ | Serving dùng hệ số trung tính 1.0; benchmark không truyền target actual, chỉ chấm đúng cohort có mặt tại cutoff |
| Một số metrics revenue chỉ tính khách có actual revenue > 0 | Loại bỏ false positives và khách không quay lại mua, làm đánh giá lạc quan | Giữ cả khách có zero purchases/revenue trong tất cả metrics |
| Cutoff dạng `YYYY-MM-DD` bị chuyển thành 00:00 | Bỏ toàn bộ giao dịch còn lại trong ngày được gọi là inclusive | Một utility dùng hết ngày cutoff, test riêng giao dịch cuối ngày |
| BG/NBD đếm invoice trong khi thời gian tính theo ngày, Gamma-Gamma dùng cả lần mua đầu | Không khớp quy ước daily transaction summary; frequency và monetary không đồng nhất | Tổng hợp purchase-day, frequency = repeat purchase-days, monetary = mean repeat-day spend, đối chiếu trực tiếp với summary của lifetimes |
| Gamma-Gamma penalization .01 dẫn đến q=1 khi ép q≥1 trên development | Population mean không hữu hạn; không thể âm thầm dùng dự báo NaN | Diagnostic chỉ trên development; chuyển monetary model về unpenalized, kiểm tra q>1 và fail rõ ràng nếu không thỏa |
| BG/NBD development có nghiệm biên a,b gần 0 gây lỗi closed-form | Một số khách mới có dự báo NaN; không được bỏ họ khỏi evaluation | Tính tích phân expected intensity tương đương cho các trường hợp không hữu hạn/âm; test độc lập với closed form tại tham số thông thường |
| Cox dùng đặc trưng tích lũy ở cuối kỳ để giải thích thời gian từ lần mua đầu; random split khách | Covariates sử dụng lịch sử tương lai so với time origin; C-index không chứng minh dự báo tương lai | Mô hình serving chuyển sang historical landmark: freeze features, chỉ dùng follow-up đã hoàn tất, dự báo ở landmark mới |
| Event time dựa trên lần mua cuối, bỏ qua inactivity xảy ra rồi khách quay lại | Đổi định nghĩa event sau khi đã biết tương lai; nhầm current status với first event | Nhãn prospective là first future inactivity crossing, dù khách tái kích hoạt sau đó; KM cũng có first-event table riêng |
| Nội suy tuyến tính survival và kéo phẳng tail tới hàng nghìn ngày | Survival estimate là step function; ngoài follow-up không đủ bằng chứng | Dùng step evaluation cho landmark Cox, chặn horizon >90 ngày |
| PH diagnostic tự tính correlation từ residual và mask khó bảo đảm index alignment | Kiểm định được báo cáo không đáng tin cậy | Dùng `lifelines.statistics.proportional_hazard_test`; lưu p-values và công khai vi phạm |
| Duplicate raw rows và missing numeric values chưa được loại sạch | Gây đếm lặp và giá trị không hợp lệ | Gắn duplicate/nonfinite flags, giữ audit rows, loại khỏi analytics |
| H_days của ERL chỉ là nhãn; cap không áp dụng cho mọi stopping path | Output không đúng estimand, có thể vượt cap | H_days thực sự giới hạn restricted mean, cap mọi path; thêm Monte Carlo SE, cap fraction, stable seed theo customer |
| Segmentation không truyền đầy đủ inactivity_days/H_days | Người dùng yêu cầu một định nghĩa nhưng kết quả theo định nghĩa khác | Truyền tham số nhất quán; integration test threshold 60 với restriction 30 |
| JSON chứa NaN cho single-purchase customers; SQLite chỉ dựa regex để bảo vệ read-only | API có thể lỗi serialization; bảo vệ dữ liệu nên nằm ở connection | Missing monetary → null; SQLite mở mode=ro, query_only, đóng connection; fix trailing SQL semicolon |

Thay đổi nhỏ bổ sung: frontend thực sự đọc VITE_API_URL; requirements phản ánh `create_agent` của LangChain 1.x; ETL chạy được từ repo root và không lỗi in Unicode trên console Windows.

## Dữ liệu và đơn vị đo

- Nguồn local: UCI Online Retail, 2010-12-01 08:26 đến 2011-12-09 12:50.
- 541,909 raw rows; 392,692 clean rows; 4,338 distinct customers; 18,532 invoices.
- 5,268 exact duplicates trong raw; 5,192 trong số đó vượt qua các filter clean cũ. Loại duplicate là giả định có công khai: không có event ID để phân biệt chắc chắn hai dòng giống hệt nhưng hợp lệ.
- 135,080 rows thiếu customer ID; 9,288 cancellation rows; 10,624 nonpositive/missing quantity rows; 2,517 nonpositive/missing price rows. Các flag có overlap, không được cộng để suy ra số dòng loại.
- Tổng positive gross sales sau cleaning: £8,887,208.894. Không phải net revenue: returns/cancellations bị loại và không được offset. Không có margin để tính profit CLV.
- SHA-256 nguồn và dependency versions nằm trong evaluation.json. Không winsorize outlier sau khi xem holdout để làm metrics đẹp hơn; wholesale orders có thể là giá trị kinh doanh thật.

## Thiết kế đánh giá mới

| Fold | Scoring cutoff | Kết thúc outcome 90 ngày | CLV cohort | Active churn cohort |
|---|---|---|---:|---:|
| Development | 2011-06-09 | 2011-09-07 | 2,787 | 1,961 |
| Final retrospective test | 2011-09-09 | 2011-12-08 | 3,362 | 1,899 |

Với mỗi cutoff, BG/NBD và Gamma-Gamma chỉ fit trên lịch sử tới cutoff. Cox fit tại landmark trước cutoff 90 ngày, chỉ sử dụng outcomes hoàn tất tới cutoff; features tại scoring cutoff không sử dụng follow-up tương lai. Chỉ khách active tại scoring cutoff được đưa vào churn risk set. Các khách đã có inactivity episode trước đây nhưng đang active có thể quay lại risk set; estimand là **next episode**, không phải permanent death.

Churn features gồm log số invoices, log mean invoice value, log product diversity, days since last purchase và tenure. Cox dùng L2 penalty .1, BG/NBD .05; không chạy feature/hyperparameter search theo final scores. Gamma-Gamma unpenalized được chọn vì kiểm tra finite-mean trên development, không vì tối ưu final MAE. Historical customers có thể xuất hiện ở nhiều fold: phù hợp bài toán rescoring existing customers, không chứng minh generalization tới khách hoàn toàn mới.

Baseline được tính với cùng cohort/outcome: last-90-day purchase-days/gross spend; historical prevalence; recency bins; continuous recency-only logistic fit trên historical labels. Có 1,000 paired customer bootstrap resamples để ước lượng chênh lệch MAE/Brier. CIs này có điều kiện trên model đã fit và fold đang xét; không bao gồm refit, seasonality, hoặc uncertainty do chọn protocol.

Đây là **retrospective out-of-time benchmark**. Cửa sổ cuối từng được workflow cũ sử dụng, và đã được chạy lại khi sửa lỗi số học. Không gọi nó là pristine unseen test hoặc một prospective experiment. Không có thêm dữ liệu ngoài dataset để chứng minh external validity.

## Kết quả có thể bảo vệ

### Churn: tín hiệu tốt, xác suất còn hạn chế

| Metric | Development Cox | Development recency logistic | Final Cox | Final recency logistic |
|---|---:|---:|---:|---:|
| ROC-AUC | 0.8165 | 0.7310 | 0.8327 | 0.7707 |
| Average precision | 0.8502 | 0.7977 | 0.8333 | 0.7922 |
| Brier score ↓ | 0.1794 | 0.2046 | 0.1721 | 0.2016 |
| Top-20% precision | 91.86% | 90.84% | 91.58% | 90.00% |

Final Brier thấp hơn continuous recency logistic **14.61%**. Paired Brier difference (Cox − baseline) = **−0.02944**, 95% CI **[−0.03468, −0.02396]**. Đây là cải thiện predictive error có bằng chứng tốt nhất của project, so với baseline đã fit đúng thời gian.

Ở final fold có 940 events/1,899 khách, event rate 49.50%. Chọn 380 khách đứng đầu theo Cox, 348 khách có event trong follow-up: precision 91.58%, recall 37.02%, lift khoảng 1.85 so với random selection. Recency-only đã đạt precision 90%; vì vậy không nên phóng đại phần tăng thêm của mô hình ở đúng budget 20%.

Một số PH tests thất bại, đặc biệt recency. Mean predicted churn 55.52% so với actual 49.50% cho thấy overprediction. Calibration deciles được lưu để kiểm tra; không tuyên bố calibrated probabilities hoặc causal hazard effects. Cải thiện Brier không đồng nghĩa giả định mô hình đúng hoàn toàn. Dùng cho exploratory ranking có cơ sở hơn dùng làm cam kết xác suất từng khách.

Sensitivity với threshold 60/90/180 ngày cho thấy event prevalence thay đổi đáng kể (66.20% / 49.50% / 21.91%) và cohort cũng đổi (1,491 / 1,899 / 2,779). ROC-AUC 180 ngày cao không được chọn làm headline: nhiều khách có thời điểm crossing gần như quyết định từ recency trong horizon 90 ngày. Không diễn giải chênh lệch các threshold là improvement trên cùng một bài toán.

### CLV: ranking có ích, forecast tổng chưa đạt

| Final 90-day metric | BG/NBD + Gamma-Gamma | Last-90-day baseline |
|---|---:|---:|
| Purchase-day MAE | 0.9256 | 0.9405 |
| Purchase-day RMSE | 1.4875 | 1.4741 |
| Gross sales MAE/customer | £579.50 | £601.99 |
| Gross sales RMSE/customer | £2,988.09 | £2,825.91 |
| Aggregate sales bias | −40.47% | −30.36% |
| Top-20% future-sales capture | 69.36% | 65.78% |
| Top-20% lift vs random | 3.46× | 3.29× |

Final cohort actual gross sales £2,657,588.311, model total £1,581,980.329. MAE giảm point estimate 3.73% nhưng CI của chênh lệch £MAE là **[−£58.69, +£11.13]**, chứa 0. Purchase MAE difference CI cũng chứa 0. Không đưa “statistically significant CLV improvement” vào resume. RMSE và aggregate bias còn kém baseline; seasonal holiday demand là một giả thuyết phù hợp với thời điểm nhưng chưa được xác lập nguyên nhân.

69.36% capture là **doanh thu đã quan sát trong holdout của nhóm được xếp hạng bằng dự báo**, không phải doanh thu mô hình tạo thêm, giữ lại, hoặc tối ưu nhờ campaign. Model chọn 673/3,362 khách (ceil 20%); lift dùng đúng fraction 673/3,362.

Single-purchase cohort có 1,406 khách; AOV fallback là mean repeat-day spend của cohort train. Trong nhóm này top-20% revenue lift chỉ ~1.12×: khả năng cá nhân hóa yếu hơn nhóm repeat buyers. Spearman correlation giữa frequency và repeat monetary ~0.182 không chứng minh đầy đủ giả định independence của Gamma-Gamma. Dữ liệu cũ và customer history bắt đầu từ đầu dataset gây giới hạn về acquisition history/left truncation.

### ERL, segmentation và retention score

Đã sửa tính đúng restriction và thêm simulation diagnostics. Monte Carlo SE chỉ đo simulation noise; chưa có prospective validation đủ dài cho lifetime forecast, parameter uncertainty hoặc đủ dữ liệu để khẳng định các bucket VIP >720 ngày có ý nghĩa kinh doanh. Giữ ERL như exploratory model estimate, không dùng làm headline về độ chính xác.

`CLV × churn probability` chỉ là heuristic ranking, có thể kết hợp hai yếu tố đã liên quan đến dropout; không phải causal value-at-risk hay expected incremental profit. Discount 50%, loyalty actions và các recommended actions là giả thuyết chưa được A/B test. Muốn khẳng định ROI cần treatment/control, response uplift và cost/margin data. Không thêm uplift model khi dataset không có treatment outcomes.

## Software và phần chưa xác minh

- 18 regression/integration tests pass bằng unittest. Bao phủ cutoff, daily RFM reference equivalence, future-data invariance, reactivation, event/censor boundaries, numerical fallback, cap/seed consistency, read-only DB, JSON null, API/evaluation consistency, KM và segmentation parameters.
- Frontend `npm run build` pass. Browser end-to-end và live LLM generation không nằm trong verification đã chạy. Frontend test mẫu `expect(true)` không là bằng chứng correctness.
- Agent results hiện có ghi 37/40 (92.5%) theo grader cũ. Grader chỉ match string/numbers ở bất kỳ đâu; một số customer ID có ngay trong câu hỏi nên có thể false positive. Targets đã cũ sau khi thay model/cohort. Không tái dùng metric này cho resume; không gọi paid API trong review này.
- Cache dùng snapshot in-memory, chưa có data-version invalidation đầy đủ; restart backend sau ETL. Cox cache key dùng cutoff, threshold và số dòng, chưa phải fingerprint nội dung.
- `clear_memory()` hiện vẫn là no-op và UI dùng thread mặc định dùng chung. Đây là hạn chế app cần giải quyết trước multi-user deployment. Chưa có authentication, persistent session ownership, production monitoring hoặc load test.
- Setup scripts cũ có bước xóa/recreate venv; không chạy chúng trong review. Sử dụng trực tiếp environment đã có.
- `backend/test/` giữ các exploratory scripts cũ; `pytest.ini` chỉ định maintained suite tại `backend/tests/`. Các assert RFM trong script cũ dùng định nghĩa monetary trước review không còn là reference đúng.
- Có runtime warning về covariance BG/NBD tại nghiệm biên và deprecation warnings từ dependencies. Point predictions được kiểm tra hữu hạn; không coi parameter standard errors đó là inference đáng tin.

Những điểm trên không được che bằng metric đẹp. Phần app không liên quan trực tiếp chất lượng Data Science được ghi lại thay vì mở rộng sang một dự án production engineering.

## Góc nhìn Hiring Manager

Project thể hiện tốt feature/label design, customer-level aggregation, survival modeling, probabilistic purchase/value modeling, temporal validation, baseline discipline, uncertainty assessment và khả năng đưa model vào API. Điều giúp nổi bật là phát hiện/sửa leakage, kiểm tra cả non-buyers, đối chiếu serving với offline evaluation và thừa nhận kết quả aggregate CLV còn yếu.

Thứ tự kể trong phỏng vấn: business question → unit of prediction → information available at cutoff → label/horizon → baseline → metrics/uncertainty → decision/limitations. Chỉ nói về LLM sau khi giải thích được tính đúng của analytics. Không dùng số lượng model hoặc 12 segment như bằng chứng hiệu quả kinh doanh.

## Resume bullets (initial review; superseded)

- Engineered an auditable retail analytics pipeline for 541,909 transaction records, applying cancellation, missing-value, and duplicate checks to produce 392,692 clean records across 4,338 customers.
- Developed a historical-landmark Cox model with temporally frozen customer features, achieving 0.833 ROC-AUC and reducing Brier score by 14.6% versus a recency-only logistic baseline on a 90-day temporal holdout of 1,899 customers.
- Modeled 90-day customer spending with BG/NBD and Gamma-Gamma, ranking the top 20% of 3,362 customers to capture 69.4% of observed future gross sales versus 65.8% using recent-spend ranking.

Không thay “observed future gross sales” bằng generated/recovered revenue. Nếu cần rút còn 2 bullets, ưu tiên churn và CLV ranking, đưa data scale vào project description. Khi phỏng vấn phải nói rõ final benchmark là retrospective, PH violations còn tồn tại và CLV aggregate underprediction 40.5%.

## Reproduce

Chạy từ repo root (Windows):

```powershell
.\venv\Scripts\python.exe backend/etl/load_online_retail.py
.\venv\Scripts\python.exe backend/evaluate.py
.\venv\Scripts\python.exe -m unittest discover -s backend/tests -v
```

Frontend build: `cd frontend`, sau đó `npm run build`. Dependency constraints: `backend/requirements-tested.txt`. `evaluation.json` và prediction CSVs là nguồn số liệu; chỉnh code thì chạy lại trước khi thay resume metrics.

Method references: [lifetimes daily summaries and repeat-transaction convention](https://lifetimes.readthedocs.io/en/latest/lifetimes.html), [lifelines time-varying covariates](https://lifelines.readthedocs.io/en/latest/Time%20varying%20survival%20regression.html), [lifelines proportional-hazards diagnostics](https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html).
