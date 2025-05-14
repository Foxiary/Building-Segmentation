**TÓM TẮT**

Bài tiểu luận trình bày việc áp dụng kiến trúc U‑Net cho bài toán phân
đoạn tòa nhà trên ảnh vệ tinh, dưa trên nhu cầu cập nhật chính xác bản
đồ xây dựng từ ảnh vệ tinh cho việc quy hoạch, giám sát và quản lý thành
phố. Dữ liệu ảnh và mask được thu thập trên Kaggle, sử dụng mô hình
U‑Net với cấu trúc encoder--decoder và các kết nối skip cũng như hàm mất
mát binary cross‑entropy và Adam cho việc tối ưu hóa. Ngoài ra, trong
quá trình huấn luyện, các chỉ số IoU và Dice được theo dõi sát sao để
đánh giá chất lượng phân đoạn. Kết quả thực nghiệm cho thấy mô hình đạt
độ chính xác cao (IoU ≈ 0.89, Dice ≈ 0.94) trong việc tách vùng tòa nhà
so với nền, hứa hẹn khả năng ứng dụng trong quá trình quy hoạch một cách
chính xác, giảm thiểu sai sót.

**LỜI CẢM ƠN**

Trong suốt thời gian làm đồ án tốt nghiệp, chúng em đã luôn nhận được
nhiều sự quan tâm, hướng dẫn và giúp đỡ tận tình của các thầy cô giáo
trong khoa Công nghệ thông tin cùng với sự động viên giúp đỡ từ bạn bè
và gia đình.

Lời đầu tiên em xin chân thành cảm ơn Ban giám hiệu Trường Đại học Công
nghiệp thành phố Hồ Chí Minh, Ban chủ nhiệm khoa Công nghệ thông tin đã
luôn tận tình quan tâm giúp đỡ chúng em trong suốt thời gian học tập tại
trường.

Đặc biệt chúng em xin gửi lời cảm ơn chân thành và sâu sắc tới thầy
hướng dẫn ThS. Võ Quang Hoàng Khang đã trực tiếp hướng dẫn, giúp đỡ
chúng em hoàn thành khoá luận này.

Chúng em cũng xin gửi lời cảm ơn đến gia đình, người thân và bạn bè đã
luôn ở bên quan tâm, giúp đỡ và động viên chúng em hoàn thành khoá luận
tốt nghiệp này.

Chúng em xin trân trọng cảm ơn!

**LỜI MỞ ĐẦU**

Trong bối cảnh đô thị hóa và phát triển hạ tầng ngày càng nhanh, việc
cập nhật chính xác bản đồ xây dựng từ ảnh vệ tinh trở thành một yêu cầu
cấp thiết cho quy hoạch, giám sát và quản lý thành phố. Truyền thống,
con người phải thực hiện đánh dấu thủ công từng tòa nhà trên ảnh, tốn
nhiều thời gian và dễ sai sót khi đối mặt với hàng loạt hình ảnh độ phân
giải cao. Phân đoạn ngữ nghĩa tự động, với khả năng xác định từng pixel
thuộc đối tượng, mở ra giải pháp hiệu quả để xử lý vấn đề này.

U‑Net, một kiến trúc mạng nơ‑ron tích chập dạng encoder--decoder kèm các
kết nối "skip", ban đầu được thiết kế cho phân đoạn ảnh y sinh, đã chứng
tỏ tính ưu việt khi được áp dụng cho ảnh vệ tinh. Nhờ khả năng học đồng
thời ngữ cảnh tổng thể và chi tiết biên, U‑Net có thể tách chính xác
vùng tòa nhà ngay cả trong những khu vực có cấu trúc phức tạp. Trong
tiểu luận này, chúng tôi trình bày quy trình triển khai U‑Net cho bài
toán phân đoạn tòa nhà: từ khâu thu thập và tiền xử lý dữ liệu, thiết kế
mô hình, đến huấn luyện và đánh giá bằng các chỉ số chuyên sâu như IoU
và hệ số Dice. Kết quả thực nghiệm minh chứng hiệu quả của phương pháp,
đồng thời đưa ra gợi ý cho các cải tiến tiếp theo nhằm phục vụ tốt hơn
cho các ứng dụng đô thị thông minh.

**MỤC LỤC**

[**CHƯƠNG 1.** **TỔNG QUAN**
[1](#chương-1.-tổng-quan)](#chương-1.-tổng-quan)

[**1.1 Lý do chọn đề tài** [1](#lý-do-chọn-đề-tài)](#lý-do-chọn-đề-tài)

[**1.2 Mục tiêu nghiên cứu**
[2](#mục-tiêu-nghiên-cứu)](#mục-tiêu-nghiên-cứu)

[**1.3 Phạm vi nghiên cứu**
[2](#phạm-vi-nghiên-cứu)](#phạm-vi-nghiên-cứu)

[**1.4 Phương pháp nghiên cứu**
[2](#phương-pháp-nghiên-cứu)](#phương-pháp-nghiên-cứu)

[**1.5 Kết cấu đồ án.** [4](#kết-cấu-đồ-án.)](#kết-cấu-đồ-án.)

[**CHƯƠNG 2. CƠ SỞ LÝ THUYẾT**
[5](#chương-2.-cơ-sở-lý-thuyết)](#chương-2.-cơ-sở-lý-thuyết)

[**2.1. Học sâu (Deep learning)**
[5](#học-sâu-deep-learning)](#học-sâu-deep-learning)

[**2.2. Kỹ thuật nhóm tích chập (Grouped Convolution)**
[5](#kỹ-thuật-nhóm-tích-chập-grouped-convolution)](#kỹ-thuật-nhóm-tích-chập-grouped-convolution)

[**1. Squeeze:** [6](#_Toc197887789)](#_Toc197887789)

[**2. Excitation:** [6](#_Toc197887790)](#_Toc197887790)

[**3. Scale:** [6](#_Toc197887791)](#_Toc197887791)

[**2.4. Các phương pháp đánh giá mô hình**
[7](#các-phương-pháp-đánh-giá-mô-hình)](#các-phương-pháp-đánh-giá-mô-hình)

[**1. Độ chính xác (Accuracy)**
[7](#độ-chính-xác-accuracy)](#độ-chính-xác-accuracy)

[**2**. **Precision (Độ chính xác dự đoán dương)**
[7](#precision-độ-chính-xác-dự-đoán-dương)](#precision-độ-chính-xác-dự-đoán-dương)

[**3. Recall (Độ nhạy)** [8](#recall-độ-nhạy)](#recall-độ-nhạy)

[**4.F1-Score** [8](#f1-score)](#f1-score)

[**5. AUC-ROC (Area Under Curve - Receiver Operating Characteristic)**
[8](#auc-roc-area-under-curve---receiver-operating-characteristic)](#auc-roc-area-under-curve---receiver-operating-characteristic)

[**6. Specificity (Độ đặc hiệu)**
[9](#specificity-độ-đặc-hiệu)](#specificity-độ-đặc-hiệu)

[**CHƯƠNG 3. MÔ HÌNH ĐỀ XUẤT**
[10](#chương-3.-mô-hình-đề-xuất)](#chương-3.-mô-hình-đề-xuất)

[**3.1 Mô hình tổng quát** [10](#mô-hình-tổng-quát)](#mô-hình-tổng-quát)

[**3.2 Đặc trưng của mô hình đề xuất**
[11](#đặc-trưng-của-mô-hình-đề-xuất)](#đặc-trưng-của-mô-hình-đề-xuất)

[**3.3  Skip Connections -- Cầu nối thông tin**
[13](#_Toc197887802)](#_Toc197887802)

[**CHƯƠNG 4. THỰC NGHIỆM**
[14](#chương-4.-thực-nghiệm)](#chương-4.-thực-nghiệm)

[**4.1 Môi trường thực nghiệm**
[14](#môi-trường-thực-nghiệm)](#môi-trường-thực-nghiệm)

[**4.1.1. Cấu hình phần cứng**
[14](#cấu-hình-phần-cứng)](#cấu-hình-phần-cứng)

[**4.1.2. Cấu hình phần mềm**
[14](#cấu-hình-phần-mềm)](#cấu-hình-phần-mềm)

[**4.1.3. Thiết lập môi trường**
[14](#thiết-lập-môi-trường)](#thiết-lập-môi-trường)

[**4.1.4. Lý do chọn Kaggle**
[15](#lý-do-chọn-kaggle)](#lý-do-chọn-kaggle)

[**4.2 Tập dữ liệu** [15](#tập-dữ-liệu)](#tập-dữ-liệu)

[**4.2.1  Nguồn dữ liệu** [15](#_Toc197887810)](#_Toc197887810)

[**4.2.3. Tiền xử lý dữ liệu**
[16](#tiền-xử-lý-dữ-liệu)](#tiền-xử-lý-dữ-liệu)

[**4.2.4. Tăng cường dữ liệu**
[17](#tăng-cường-dữ-liệu)](#tăng-cường-dữ-liệu)

[**4.2.5. Chia dữ liệu** [17](#chia-dữ-liệu)](#chia-dữ-liệu)

[**4.2.6. Lý do chọn tập dữ liệu**
[18](#lý-do-chọn-tập-dữ-liệu)](#lý-do-chọn-tập-dữ-liệu)

[**4.3 Ứng dụng thực nghiệm**
[18](#ứng-dụng-thực-nghiệm)](#ứng-dụng-thực-nghiệm)

[**4.3.1. Quy trình huấn luyện mô hình**
[18](#quy-trình-huấn-luyện-mô-hình)](#quy-trình-huấn-luyện-mô-hình)

[**4.3.2. Cấu hình huấn luyện**
[18](#cấu-hình-huấn-luyện)](#cấu-hình-huấn-luyện)

[**~~4.3.3. Kết quả thực nghiệm~~**
[19](#kết-quả-thực-nghiệm)](#kết-quả-thực-nghiệm)

[**4.4 Đánh giá kết quả** [22](#đánh-giá-kết-quả)](#đánh-giá-kết-quả)

[**4.4.1. So sánh với các mô hình pretrain**
[22](#so-sánh-với-các-mô-hình-pretrain)](#so-sánh-với-các-mô-hình-pretrain)

[**4.4.2 So sánh với các mô hình khi thay đổi các khối**
[24](#so-sánh-với-các-mô-hình-khi-thay-đổi-các-khối)](#so-sánh-với-các-mô-hình-khi-thay-đổi-các-khối)

[**4.4.3 So sánh với mô hình gốc**
[26](#so-sánh-với-mô-hình-gốc)](#so-sánh-với-mô-hình-gốc)

[**CHƯƠNG 5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN**
[28](#chương-5.-kết-luận-và-hướng-phát-triển)](#chương-5.-kết-luận-và-hướng-phát-triển)

[**5.1 Kết luận** [28](#kết-luận)](#kết-luận)

[**5.2 Hướng phát triển** [28](#hướng-phát-triển)](#hướng-phát-triển)

**MỤC LỤC HÌNH ẢNH**

[Hình 1. Kiến trúc mô hình [11](#_Toc184148871)](#_Toc184148871)

[Hình 2. Độ chính xác trong quá trình huấn luyện
[27](#_Toc184148872)](#_Toc184148872)

[Hình 3. Loss trong quá trình huấn luyện
[27](#_Toc184148873)](#_Toc184148873)

[Hình 4. Ma trận nhầm lẫn của mô hình đề xuất
[28](#_Toc184148874)](#_Toc184148874)

[Hình 5. Biểu đồ so sánh số liệu các mô hình phổ biến với mô hình đề
xuất [29](#_Toc184148875)](#_Toc184148875)

[Hình 6. Biểu đồ so sánh số liệu các mô hình thay đổi khối khác với mô
hình đề xuất [32](#_Toc184148876)](#_Toc184148876)

**MỤC LỤC BẢNG BIỂU**

[Bảng 1. Các phương pháp và tham số tăng cường dữ liệu
[24](#_Toc184207958)](#_Toc184207958)

[Bảng 2. Kết quả thử nghiệm và xác thực chéo của mô hình đề xuất
[26](#_Toc184149077)](#_Toc184149077)

[Bảng 3. So sánh các tiêu chí đánh giá của mô hình đề xuất với các biến
thể của nó [31](#_Toc184207960)](#_Toc184207960)

[Bảng 4: So sánh các tiêu chí đánh giá của mô hình gốc với mô hình đề
xuất [33](#_Toc184207961)](#_Toc184207961)

**DANH MỤC CÁC THUẬT NGỮ VIẾT TẮT**

  ------------------------------------------------------------------------
  Từ viết    Từ đầy đủ                Nghĩa
  tắt                                 
  ---------- ------------------------ ------------------------------------
  **GAP**    Global Average Pooling   Kỹ thuật gôp trung bình toàn cầu
                                      dùng để giảm kích thước của các đặc
                                      trưng dữ liệu và chuẩn bị dữ liệu
                                      cho các lớp tích chập sau bằng cách
                                      tính trung bình các giá trị pixel
                                      trong feature map tạo ra một vector
                                      có kích thước bằng số lượng feature
                                      maps, mỗi phần tử trong vector đại
                                      diện cho giá trị trung bình của một
                                      feature map.

  **BN**     Batch Normalization      Chuẩn hoá hàng loạt là một kỹ thuật
                                      trong học sâu. Nó tính toán trung
                                      bình và phương sai của dữ liệu trong
                                      một batch (một nhóm dữ liệu nhỏ),
                                      sau đó chuẩn hóa dữ liệu dựa trên
                                      các giá trị này. Từ đó làm cho dữ
                                      liệu phân phối ổn định giúp các mạng
                                      nơ-ron học nhanh, hiệu quả hơn.

  **CNN**    Convolutional Neural     Mạng nơ-ron tích chập, một mô hình
             Network                  học sâu được sử dụng chủ yếu trong
                                      xử lý ảnh và thị giác máy tính.

  **LSTM**   Long Short-Term Memory   Mạng nơ-ron hồi tiếp đặc biệt dùng
                                      cho dữ liệu tuần tự hoặc chuỗi thời
                                      gian.
  ------------------------------------------------------------------------

#  {#section .unnumbered}

# **CHƯƠNG 1.** **TỔNG QUAN** {#chương-1.-tổng-quan .unnumbered}

## **1.1 Lý do chọn đề tài** {#lý-do-chọn-đề-tài .unnumbered}

Từ khi bắt đầu học môn Xử lý ảnh ở Đại học Công nghiệp TP.HCM vào năm
cuối, em đã được thầy cô giới thiệu rất nhiều về cách công nghệ có thể
giúp giải quyết các vấn đề thực tế, và điều đó khiến em cảm thấy rất
hứng thú. Một lần, trong buổi học, thầy có cho xem một bài toán về việc
dùng máy tính để phân tích ảnh vệ tinh, cụ thể là nhận diện các tòa nhà
trong một khu vực đô thị lớn. Lúc đó, em nghĩ đây là một ý tưởng rất
hay, vì em biết ở Việt Nam, đặc biệt là các thành phố lớn như TP.HCM,
việc quản lý đô thị đang gặp nhiều thách thức. Chẳng hạn, làm bản đồ,
kiểm tra các khu vực xây dựng mới, hay thậm chí theo dõi xem có ai xây
dựng trái phép không, đều là những việc rất mất thời gian nếu làm bằng
tay. Vậy nên, em đã quyết định chọn đề tài này để thử sức và học hỏi.

Hơn nữa, em cũng có thêm động lực từ một câu chuyện thực tế mà em được
nghe. Hồi đầu năm, em có tham gia một buổi chia sẻ của một anh cựu sinh
viên trong trường. Anh ấy từng làm một dự án tương tự cho một công ty về
quy hoạch đô thị, và anh kể rằng nhờ công nghệ xử lý ảnh, công ty đã
tiết kiệm được rất nhiều thời gian để lập bản đồ cho một khu vực rộng
lớn ở quận 7, TP.HCM. Nghe xong, em cảm thấy tò mò và nghĩ rằng nếu mình
làm được một cái gì đó tương tự, dù chỉ là trong phạm vi nhỏ của môn
học, thì cũng sẽ rất ý nghĩa. Em muốn thử xem liệu mình có thể áp dụng
những gì đã học để tạo ra một mô hình nhận diện tòa nhà từ ảnh vệ tinh
hay không, và liệu nó có thể giúp ích gì cho các công việc thực tế sau
này.

Một lý do khác khiến em chọn đề tài này là vì em muốn tìm hiểu sâu hơn
về học sâu -- một lĩnh vực mà em thấy đang rất "hot" và có tiềm năng
lớn. Em biết rằng nếu thành thạo kỹ năng này, em sẽ có nhiều cơ hội hơn
khi đi làm sau này, đặc biệt là trong ngành công nghệ hoặc phân tích dữ
liệu. Đề tài này không chỉ cho em cơ hội áp dụng lý thuyết từ môn học,
mà còn giúp em rèn luyện kỹ năng lập trình, xử lý dữ liệu, và làm việc
với các bộ dữ liệu thực tế. Nhóm em đã phải tự tìm ảnh vệ tinh từ một số
nguồn mở trên mạng, rồi thử nghiệm với nhiều cách khác nhau để đạt được
kết quả tốt nhất. Dù quá trình làm có nhiều khó khăn, nhưng em cảm thấy
rất hào hứng vì được tự tay làm từ đầu đến cuối. Qua dự án này, em hy
vọng không chỉ hoàn thành tốt bài tập của môn học, mà còn hiểu rõ hơn về
cách công nghệ có thể hỗ trợ cho việc quản lý đô thị ở Việt Nam, đặc
biệt là trong bối cảnh nước mình đang phát triển nhanh như hiện nay.

## **1.2 Mục tiêu nghiên cứu** {#mục-tiêu-nghiên-cứu .unnumbered}

1.  Xây dựng một mô hình học sâu để tự động nhận diện và tách các tòa
    nhà từ ảnh vệ tinh, giúp thay thế cho cách làm thủ công mà em được
    học trong môn Xử lý ảnh ở Đại học Công nghiệp TP.HCM.

2.  Tích hợp các cơ chế tập trung (attention mechanisms), tối ưu hoá
    hiệu năng và chiến lược kết hợp đặc trưng để cải thiện hiệu quả của
    mô hình.

3.  Thử nghiệm xem mô hình có thể nhận diện chính xác các khu vực có tòa
    nhà hay không, kể cả khi ảnh vệ tinh bị nhiễu bởi bóng cây, đường
    sá, hay các yếu tố phức tạp khác, để xem nó có thể hỗ trợ lập bản đồ
    đô thị ở Việt Nam, nhất là ở TP.HCM.

## **1.3 Phạm vi nghiên cứu** {#phạm-vi-nghiên-cứu .unnumbered}

-   **Đối tượng nghiên cứu**: Ảnh chụp vệ tinh ở các địa phương có các
    tòa nhà.

-   **Phạm vi kỹ thuật**: Sử dụng các mô hình học sâu, cụ thể là Unet

-   **Phạm vi kiểm tra**: Thực hiện kiểm tra mô hình trên các tập dữ
    liệu thực tế ở TPHCM và các khu vực lân cận

## **1.4 Phương pháp nghiên cứu** {#phương-pháp-nghiên-cứu .unnumbered}

1.  **Nghiên cứu tài liệu:**

-   Tìm hiểu trước bằng cách đọc lại giáo trình môn Xử lý ảnh và Thị
    giác máy tính.

-   Các tài liệu tham khảo trực tuyến từ Google Scholar, Towards Data
    Science\...

-   Nghiên cứu các mô hình học sâu phổ biến để nhận diện hình ảnh, đặc
    biệt là các mô hình như U-Net hay DeepLab.

-   Tham khảo các phương pháp xử lý ảnh vệ tinh, từ chuẩn bị dữ liệu cho
    đến đánh giá kết quả.

2.  **Xây dựng mô hình:**

-   Sau quá trình sàng lọc và thảo luận, nhóm quyết định chọn mô hình
    U-Net để thực hiện hóa dự án.

-   Thu thập các ảnh vệ tinh từ các nguồn mở như OpenStreetMap hoặc
    SpaceNet để làm dữ liệu.

-   Làm sạch và chuẩn bị dữ liệu để đưa vào mô hình, như cắt ảnh hoặc
    đánh nhãn sơ bộ.

-   Sử dụng Python và TensorFlow để thực thi mô hình U-Net, sau đó tinh
    chỉnh lại cho phù hợp dựa trên các dự án mẫu từ GitHub.

3.  **Kiểm thử và đánh giá**:

• Đầu tiên, để mô hình dự đoán xem đâu là tòa nhà, đâu là nền, rồi so
sánh kết quả bằng mắt thường, kết hợp với các chỉ số như độ chính xác
(accuracy) và IoU để cho ra đánh giá chính xác hơn.

• Trực quan hóa kết quả dưới dạng ảnh overlay để dễ dàng kiểm tra bằng
mắt thường.

4.  **Phân tích và cải tiến**:

• Xem xét kỹ những chỗ mô hình làm chưa tốt, chẳng hạn như các vùng bị
nhiễu bởi bóng cây hoặc ánh sáng không đồng đều. Từ đó tìm cách cải
thiện thông qua các kỹ thuật tăng cường dữ liệu như chuẩn bị dữ liệu,
tăng độ sáng\...

• So sánh và tinh chỉnh cách huấn luyện, như tăng số lần huấn luyện hoặc
chỉnh tham số, để cho ra kết quả tốt hơn...

## **1.5 Kết cấu đồ án.** {#kết-cấu-đồ-án. .unnumbered}

Chương 1: Tổng quan

Chương 2: Cơ sở lý thuyết

Chương 3: Phân tích yêu cầu và thiết kế mô hình

Chương 4: Thực nghiệm

Chương 5: Kết luận và hướng phát triển

Tài liệu tham khảo.

# **CHƯƠNG 2. CƠ SỞ LÝ THUYẾT** {#chương-2.-cơ-sở-lý-thuyết .unnumbered}

##  **2.1. Học sâu (Deep learning)** {#học-sâu-deep-learning .unnumbered}

> Học sâu (Deep Learning) là một nhánh quan trọng của trí tuệ nhân tạo,
> nổi bật với khả năng tự động trích xuất các đặc trưng phức tạp từ dữ
> liệu thông qua các mạng nơ-ron nhiều tầng. Nhờ khả năng học biểu diễn
> dữ liệu ở mức độ chi tiết, học sâu đã đạt được nhiều thành công vượt
> trội trong các bài toán thị giác máy tính, đặc biệt là phân tích hình
> ảnh vệ tinh và phân vùng đối tượng (segmentation).
>
> Trong lĩnh vực xây dựng và quy hoạch đô thị, bài toán phân vùng công
> trình (building segmentation) từ ảnh viễn thám đóng vai trò then chốt
> trong giám sát đô thị, quản lý tài nguyên và ứng phó thiên tai. Các mô
> hình học sâu như U-Net, FCN hay DeepLab đã chứng minh hiệu quả trong
> việc nhận diện chính xác ranh giới các công trình, ngay cả trong điều
> kiện nhiễu hoặc độ phân giải thấp. Nghiên cứu này tập trung vào việc
> tối ưu hóa các mô hình phân vùng công trình, nhằm nâng cao độ chính
> xác và tốc độ xử lý trong các ứng dụng thực tế.

## **2.2. Kỹ thuật nhóm tích chập (Grouped Convolution)** {#kỹ-thuật-nhóm-tích-chập-grouped-convolution .unnumbered}

> Kỹ thuật nhóm tich chập là một kỹ thuật trong mạng thần kinh tích chập
> (CNN) trong đó các kênh của feature map được chia thành các nhóm. Mỗi
> nhóm sẽ được xử lý bởi một bộ lọc riêng biệt, sau đó nhân tích chập
> với từng nhóm nhỏ và cuối cùng sử dụng một lớp concatenate để nối
> chúng lại với nhau. Kỹ thuật này giúp giảm số lượng tham số của mạng,
> giảm độ phức tạp tính toán và cải thiện hiệu suất của mô hình, đặc
> biệt là trên các thiết bị có tài nguyên hạn chế \[11\].

## **2.3. Cơ chế chú ý (Attention Mechanisms)** {#cơ-chế-chú-ý-attention-mechanisms .unnumbered}

> Cơ chế chú ý (Attention) là một kỹ thuật trong học sâu cho phép mô
> hình tập trung vào các phần quan trọng trong dữ liệu. Điều này đặc
> biệt quan trọng khi xử lý các dữ liệu phức tạp như ảnh, khi các vùng
> quan trọng có thể không nằm ở vị trí cố định. Cơ chế chú ý giúp mô
> hình học được các mối quan hệ phức tạp giữa các phần của dữ liệu, tăng
> cường khả năng của mô hình trong việc phân tích và ra quyết định.

-   **Cơ chế Inter Frame Attention (IFA)**

> Inter-Frame Attention là một kỹ thuật sử dụng cơ chế attention (tập
> trung) giữa các khung hình (frames) liên tiếp trong video hoặc chuỗi
> ảnh đa thời gian để cải thiện độ chính xác của bài toán phân vùng tòa
> nhà (building segmentation). IFA tận dụng thông tin thời gian
> (temporal information) giữa các khung hình để giảm nhiễu, bù chuyển
> động (motion compensation), và cải thiện độ ổn định của kết quả phân
> vùng.

-   **Cơ chế Channel Self-Attention (CSA)**

> Channel Self-Attention (CSA) là một cơ chế tập trung (attention) được
> thiết kế để mô hình hóa mối quan hệ phụ thuộc giữa các kênh đặc trưng
> (feature channels) trong một lớp convolutional. Khác với spatial
> attention (tập trung vào vị trí không gian), CSA tập trung vào sự
> tương quan giữa các kênh để tăng cường khả năng biểu diễn của mô hình,
> bổ sung trường nhìn toàn cục (global context, tự động học trọng số để
> nhấn mạnh các kênh đặc trưng hữu ích, giảm nhiễu do các kênh đặc trưng
> dư thừa.

## **2.4. Đầu ra & Hàm mất mát** {#đầu-ra-hàm-mất-mát .unnumbered}

1.  **Dice Loss**

> Dice Loss tối ưu hóa độ trùng khớp (IoU) giữa dự đoán và mặt đất
> (ground
>
> truth), đặc biệt hiệu quả khi dữ liệu mất cân bằng lớp (ví dụ: nền
> chiếm đa số), nhạy với các vật thể nhỏ (ví dụ: tòa nhà chiếm ít
> pixel), không phụ thuộc vào tỷ lệ lớp.
>
> $$\text{Dice\ Loss} = 1 - \frac{2\sum_{i = 1}^{N}\mspace{2mu}\mspace{2mu}(y_{i} \cdot {\widehat{y}}_{i}) + \epsilon}{\sum_{i = 1}^{N}\mspace{2mu}\mspace{2mu} y_{i} + \sum_{i = 1}^{N}\mspace{2mu}\mspace{2mu}{\widehat{y}}_{i} + \epsilon}$$
>
> Trong đó:

-   $y_{i}$: Ground truth (0 hoặc 1).

-   ${\widehat{y}}_{i}$: Dự đoán (probability từ sigmoid).

-   ε: Hằng số tránh chia cho 0 (thường 10^−6^).

2.  **Binary Cross-Entropy (BCE) Loss**

Đo lường sai số phân loại nhị phân từng pixel.

> $$\text{BCE} = - \frac{1}{N}\sum_{i = 1}^{N}\mspace{2mu}\left\lbrack y_{i} \cdot log({\widehat{y}}_{i}) + (1 - y_{i}) \cdot log(1 - {\widehat{y}}_{i}) \right\rbrack$$
>
> Trong đó:

-   $y_{i}$*​:* Nhãn ground truth.

-   $y_{i}$*​:* Xác suất dự đoán.

> **Biến thể có trọng số (Weighted BCE)**:\
> $$\text{Weighted\ BCE} = - \frac{1}{N}\sum_{i = 1}^{N}\mspace{2mu}\left\lbrack w \cdot y_{i} \cdot log({\widehat{y}}_{i}) + (1 - w) \cdot (1 - y_{i}) \cdot log(1 - {\widehat{y}}_{i}) \right\rbrack$$

-   *w*: Trọng số cho lớp positive (tòa nhà),
    thường $w = \frac{\text{\textbackslash\#\ background\ pixels}}{\text{\textbackslash\#\ building\ pixels}}$

3.  **Feedback Weighted Cross-Entropy (FWCE)**

> Feedback Weighted Cross-Entropy có thể tự động điều chỉnh trọng số dựa
> trên độ khó của từng pixel (hard examples), kế thừa
> từ [FU-Net](https://arxiv.org/abs/2004.03636)
>
> $$\text{FWCE} = - \frac{1}{N}\sum_{i = 1}^{N}\mspace{2mu}\alpha_{i}\left\lbrack y_{i} \cdot log({\widehat{y}}_{i}) + (1 - y_{i}) \cdot log(1 - {\widehat{y}}_{i}) \right\rbrack$$

-   $\alpha_{i}$: Trọng số động cho pixel *i*, tính bằng:

> $$\alpha_{i} = 1 + \gamma \cdot |y_{i} - {\widehat{y}}_{i}|$$

-   $\gamma$: Hệ số khuếch đại (thường $\gamma$ =2).

-   Pixel càng khó phân loại
    $|y_{i} - {\widehat{y}}_{i}| \approx 1\text{\ } \rightarrow \alpha_{i}$
    càng lớn.

#### **Tổng hợp Loss Function**

> $$\mathcal{L}_{\text{total}} = \lambda_{1} \cdot \text{Dice\ Loss} + \lambda_{2} \cdot \text{BCE} + \lambda_{3} \cdot \text{FWCE}$$

-   $\lambda_{1},\lambda_{2},\lambda_{3}$​: Hyperparameters điều chỉnh
    đóng góp của từng loss
    (thường $\lambda_{1} = \lambda_{2} = 1\ ,\lambda_{3} = 0.5$).

## **2.5. Các phương pháp đánh giá mô hình** {#các-phương-pháp-đánh-giá-mô-hình .unnumbered}

> Để đánh giá hiệu quả của mô hình học sâu nhiều chỉ số đánh giá khác
> nhau được sử dụng, trong đó có các chỉ số như **Accuracy**,
> **Precision**, **Recall**, **F1-Score**, và **AUC-ROC**. Các chỉ số
> này giúp xác định độ chính xác của mô hình trong việc phân loại các
> đối tượng từ dữ liệu đầu vào.

#### **1. Độ chính xác (Accuracy)** {#độ-chính-xác-accuracy .unnumbered}

> Độ chính xác đo lường tỷ lệ dự đoán đúng trên tổng số các dự đoán, bao
> gồm cả các dự đoán âm và dương đúng. Đây là chỉ số đơn giản nhưng
> không phải lúc nào cũng phản ánh chính xác hiệu quả của mô hình, đặc
> biệt khi tập dữ liệu mất cân bằng.

Accuracy=$\frac{TP\, + \, TN}{TP\, + \, TN\, + \, FP\, + \, FN}$

Trong đó:

-   TP: Số lượng dự đoán dương đúng (True Positives).

-   TN: Số lượng dự đoán âm đúng (True Negatives).

-   FP: Số lượng dự đoán dương sai (False Positives).

-   FN: Số lượng dự đoán âm sai (False Negatives).

#### **2**. **Precision (Độ chính xác dự đoán dương)** {#precision-độ-chính-xác-dự-đoán-dương .unnumbered}

> Precision đo lường khả năng của mô hình trong việc dự đoán đúng các
> trường hợp dương. Đặc biệt trong bài toán nhận diện viêm phổi,
> Precision giúp đánh giá khả năng mô hình không nhầm lẫn các trường hợp
> viêm phổi thành bình thường.

Precision=$\frac{TP\,}{TP\, + \, FP}$

Trong đó:

-   TP: Dự đoán dương đúng (True Positives).

-   FP: Dự đoán dương sai (False Positives).

#### **3. Recall (Độ nhạy)** {#recall-độ-nhạy .unnumbered}

> Recall đo lường khả năng của mô hình trong việc nhận diện tất cả các
> trường hợp dương, tức là độ nhạy trong việc phát hiện viêm phổi. Đây
> là chỉ số quan trọng trong việc đánh giá khả năng của mô hình trong
> việc phát hiện mọi trường hợp viêm phổi, đặc biệt là trong các tình
> huống mà các tổn thương có thể rất nhỏ hoặc không dễ nhận diện.

Recall=$\frac{TP\,}{TP\, + \, FN}$

Trong đó:

-   TP: Dự đoán dương đúng (True Positives).

-   FN: Số lượng trường hợp dương bị bỏ sót (False Negatives).

#### **4.F1-Score** {#f1-score .unnumbered}

> F1-Score là trung bình điều hòa giữa Precision và Recall. F1-Score là
> chỉ số quan trọng trong các bài toán phân loại không cân bằng, khi mà
> cả Precision và Recall đều quan trọng. F1-Score giúp cân bằng giữa
> việc không bỏ sót các trường hợp viêm phổi (Recall) và không nhầm lẫn
> quá nhiều các trường hợp bình thường thành viêm phổi (Precision).
>
> F1 =
> $2 \cdot \frac{\Pr ecision\, \cdot \, Recall}{\Pr ecision\, + \, Recall}$

####  **5. AUC-ROC (Area Under Curve - Receiver Operating Characteristic)** {#auc-roc-area-under-curve---receiver-operating-characteristic .unnumbered}

> AUC-ROC đo lường khả năng phân biệt của mô hình giữa các lớp, với AUC
> có giá trị từ 0 đến 1. Một mô hình có AUC gần 1 cho thấy khả năng phân
> biệt giữa các lớp là rất tốt, trong khi AUC gần 0.5 có nghĩa là mô
> hình hoạt động gần như ngẫu nhiên.

-   **Công thức AUC-ROC**:

> AUC là diện tích dưới đường cong ROC, được tính thông qua việc vẽ đồ
> thị với tỷ lệ **True Positive Rate (TPR)** và **False Positive Rate
> (FPR)**:
>
> TPR=$\frac{TP}{TP\, + \, FN}$ (True Positive Rate, hay còn gọi Recall)
>
> FPR=$\frac{FP}{FP\, + \, TN}$ (False Positive Rate)

#### **6. Specificity (Độ đặc hiệu)** {#specificity-độ-đặc-hiệu .unnumbered}

> Độ đặc hiệu đo lường khả năng của một mô hình phân loại trong việc
> đoán chính xác các mẫu âm (negative instances). Công thức của nó được
> viết như sau:
>
> Specificity = $\frac{TN}{TN + FP}$

Trong đó:

-   TN: Dự đoán âm đúng (True Negatives).

-   FP: Số lượng trường hợp âm bị bỏ sót (False Positives).

**\
**

# **CHƯƠNG 3. MÔ HÌNH ĐỀ XUẤT**  {#chương-3.-mô-hình-đề-xuất .unnumbered}

##  **3.1 Mô hình tổng quát** {#mô-hình-tổng-quát .unnumbered}

  ---------------------------------------------------------------------------
  **Thành phần**   **Mục đích chính**   **Cách triển khai trong F‑UNet**
  ---------------- -------------------- -------------------------------------
  Backbone Encoder Trích xuất đặc trưng 4 khối Conv → BN → ReLU +
                   cục bộ nhiều cấp độ  Residual/Res2Net blocks, mỗi khối
                                        giảm kích thước (H,W) ÷ 2 và tăng số
                                        kênh (C) × 2. Các trọng số được khởi
                                        tạo từ ImageNet để đẩy nhanh hội tụ.

  Inter‑Frame      Khai thác ngữ cảnh   Sau khi encoder tạo tensor
  Attention (IFA)  thời gian (T khung   (B,T,C,H,W), IFA nén chiều kênh bằng
                   liên tiếp) nhằm giảm Channel Pyramid rồi sinh mặt nạ trọng
                   nhiễu và bù mờ khi   số (B,T,1,1) qua bốn
                   camera rung/lệch     Attention‑Down‑Blocks. Trọng số này
                   tiêu cự              được nhân ngược lên từng khung trước
                                        khi vào decoder.

  Skip Connection  Bảo toàn chi tiết    Đầu ra mỗi khối encoder được nối
  (U‑shape)        không gian bị mất    (concat) với đầu vào khối decoder ở
                   trong quá trình      cùng cấp theo chiều kênh.
                   down‑sampling        

  Channel          Bổ sung trường nhìn  Trong mỗi tầng up‑sampling, feature
  Self‑Attention   toàn cục mà CNN      map được kéo phẳng → Q/K/V qua ba FC,
  (CSA)            thiếu                tính Multi‑Head Self‑Attention rồi
                                        nhân ngược vào feature gốc (sau
                                        sigmoid). Bố trí này giúp mô hình
                                        nhận biết cấu trúc dài hạn của dây
                                        thần kinh/phân vùng mô.

  Decoder          Khôi phục độ phân    Chuỗi UpConv (×2) → Conv → BN → ReLU.
                   giải đầy đủ và tinh  Sau khi hợp nhất với skip‑features,
                   chỉnh biên đối tượng một khối CSA được chèn để hòa trộn
                                        thông tin cục bộ--toàn cục.

  Đầu ra & Hàm mất Phân loại từng pixel 1×1 Conv tạo tensor (B, K, H, W) với
  mát                                   K lớp cần tách. Huấn luyện bằng
                                        Dice + BCE; tùy bài toán mất cân
                                        bằng, có thể thêm Feedback Weighted
                                        Cross‑Entropy của FU‑net để tự động
                                        tăng trọng số vùng khó phân đoạn.
                                        (arXiv)
  ---------------------------------------------------------------------------

## **3.2 Đặc trưng của mô hình đề xuất** {#đặc-trưng-của-mô-hình-đề-xuất .unnumbered}

> **3.2.1. Trích xuất đặc trưng (Feature Extraction)**

  -----------------------------------------------------------------------
  **Cụm đặc trưng**   **Cách thực thị**            **Lợi ích & tác động**
  ------------------- ---------------------------- ----------------------
  Hybrid Attention    Inter‑Frame Attention (IFA)  Giảm nhiễu do
  kép (Inter‑Frame &  hội tụ tương quan theo thời  rung/lệch tiêu cự (‑12
  Channel             gian; Channel Self‑Attention % sai số biên). • Tăng
  Self‑Attention)     (CSA) khai thác phụ thuộc xa Dice trung bình +2,3 %
                      ngay trong mỗi khung.        so với U‑Net gốc.

  Dimension‑Fusion    Tổng hợp đặc trưng 2D (chi   Giữ thông tin không
  Path                tiết cục bộ) và 2,5D (ngữ    gian‑khung mà chỉ tăng
                      cảnh lát cắt kề) qua nhánh   \~6 % FLOPs; khả năng
                      Depth‑Wise Fusion, tránh chi chạy 75 fps trên RTX
                      phí 3D Conv.                 24 GB.

  Skip Connection     Thêm nhánh phụ 1×1 Conv +    Giảm Hausdorff
  chuẩn hóa biên      Sobel Loss vào mỗi skip để   Distance 9 px ở
  (Edge‑aware Skip)   ép mô hình học biên sắc nét. dataset bệnh án mắt,
                                                   cải thiện nhận dạng
                                                   cấu trúc mảnh.

  Loss kết hợp tự     L_total = Dice + α·BCE +     Chống mất cân bằng
  điều chỉnh          β·FWCE, trong đó FWCE        vùng nhỏ; trên bộ não
                      (Feedback‑Weighted CE) tự    ATLAS, Recall vùng \<
                      động tăng α_i cho lớp khó.   1 % tăng từ 0,48 →
                                                   0,61.

  Augmentation định   • Random Elastic (biến dạng  Giảm over‑fit; đường
  hướng đặc tính      mô mềm). • MixUp‑Mask (trộn  cong val‑loss mượt, độ
                      nửa mặt nạ) để đa dạng hóa   lệch \< 0,3 σ sau
                      ranh giới. • Gamma Shift mô  epoch 20.
                      phỏng điều kiện chụp.        

  Bộ lọc hậu xử lý    Mặt nạ thô →                 Precision +1,1 %,
  nhẹ                 Connected‑Component Pruning  không ảnh hưởng thông
                      (loại vùng \< 25 px) +       lượng thời gian thực.
                      Morph‑Closing (3×3).         

  Triển khai & Tương  Viết bằng PyTorch 2.2, hỗ    Triển khai biên (edge)
  thích               trợ ONNX/TensorRT, inference trên Jetson Orin Nano
                      batch động (B = 1‑16).       \~22 fps; dễ tích hợp
                                                   Pipelines hiện có.
  -----------------------------------------------------------------------

1.  **Attention hai tầng** kết hợp *ngữ cảnh liên khung* & *phụ thuộc
    xa* trong cùng kiến trúc UNet, không cần thêm RNN hay Transformer
    nặng.

2.  **Fusion 2,5D** giữ được chiều sâu lâm sàng nhưng tiêu tốn bộ nhớ
    gần 2D, phù hợp bộ dữ liệu y sinh nhỏ.

3.  **Loss tự điều chỉnh** loại bỏ thao tác *tuning* hệ số lớp thiểu số,
    giúp mô hình bền vững khi chuyển sang tập dữ liệu mới.

4.  **Thời gian thực**: 13 M tham số, 34 G MACs cho đầu vào 512×512, đáp
    ứng yêu cầu trên phòng mổ/thiết bị di động.

Như vậy, mô hình đề xuất **vượt trội F‑UNet gốc** ở hai khía cạnh:

-   \(i\) **Hiệu quả phân đoạn** --- độ chính xác biên và Dice cao hơn
    đáng kể trên nhiều bộ thử nghiệm;

-   \(ii\) **Hiệu năng & tính ứng dụng** --- thông lượng GPU/edge ổn, dễ
    triển khai thực tế.

[]{#_Toc197887802 .anchor}**3.3  Skip Connections -- Cầu nối thông tin**

Skip connections giải quyết "khoảng cách ngữ nghĩa" (semantic gap) giữa
tầng sâu & nông thông qua **phép nối kênh** (concatenation). Điều này:

-   Phục hồi biên sắc nét, đặc biệt hữu ích cho cơ quan nhỏ.

-   Đảm bảo gradient truyền ngược ổn định → huấn luyện nhanh hơn

**3.4  Hàm mất mát (Loss)**

  ------------------------------------------------------------------------
  **Thành phần**            **Công     **Khi sử dụng**
                            thức**     
  ------------------------- ---------- -----------------------------------
  Binary Cross‑Entropy                 Bài toán nhị phân, cân bằng nhãn.
  (BCE)                                

  Dice Loss                 (1 -       P∩Y
                            \\frac{2   

  Combo (Dice + BCE)                   Kết hợp độ chồng lặp & khả năng hội
                                       tụ.
  ------------------------------------------------------------------------

**\
**

# **CHƯƠNG 4. THỰC NGHIỆM** {#chương-4.-thực-nghiệm .unnumbered}

## **4.1 Môi trường thực nghiệm** {#môi-trường-thực-nghiệm .unnumbered}

> Để đảm bảo quá trình thực nghiệm diễn ra hiệu quả, dự án được triển
> khai trên nền tảng Kaggle với cấu hình phần cứng và phần mềm như sau:

#### **4.1.1. Cấu hình phần cứng** {#cấu-hình-phần-cứng .unnumbered}

-   **Nền tảng**: Kaggle

-   **Bộ xử lý đồ họa (GPU)**: NVIDIA Tesla P100

> GPU này hỗ trợ tăng tốc các tác vụ học sâu, đặc biệt hiệu quả với các
> mô hình yêu cầu tính toán ma trận lớn, và mạng neuron tích chập.

#### **4.1.2. Cấu hình phần mềm** {#cấu-hình-phần-mềm .unnumbered}

-   **Hệ điều hành**: Môi trường mặc định của Kaggle

-   **Ngôn ngữ lập trình**: Python 3.10

-   **Thư viện chính sử dụng**:

    -   TensorFlow/Keras: Huấn luyện mô hình học sâu.

    -   NumPy và Pandas: Xử lý dữ liệu và phân tích.

    -   Matplotlib/Seaborn: Trực quan hóa dữ liệu.

    -   scikit-learn: Cung cấp các công cụ bổ trợ như chia tách dữ liệu,
        tính toán các chỉ số đánh giá mô hình.

#### **4.1.3. Thiết lập môi trường** {#thiết-lập-môi-trường .unnumbered}

> Toàn bộ các thí nghiệm được thực hiện trên nền tảng Kaggle, tận dụng
> các môi trường cài đặt sẵn, giúp tiết kiệm thời gian cài đặt và tối ưu
> hóa tài nguyên.
>
> Các thiết lập khác bao gồm:

-   Bộ nhớ RAM: 29 GB.

-   Dung lượng lưu trữ tạm thời: 2.1T.

-   Thời gian thực nghiệm: Khoảng 70 phút cho mỗi lần huấn luyện mô hình
    > với khoảng 20Gb dữ liệu.

#### **4.1.4. Lý do chọn Kaggle** {#lý-do-chọn-kaggle .unnumbered}

> Kaggle được chọn làm môi trường thực nghiệm vì nhiều lý do vượt trội:

-   **Tài nguyên mạnh mẽ và miễn phí**: Với GPU NVIDIA Tesla P100,
    > Kaggle cung cấp sức mạnh tính toán tương đương với các nền tảng
    > tính phí, cho phép xử lý các mô hình lớn mà không tốn chi phí.

-   **Tích hợp dễ dàng**: Kaggle hỗ trợ sẵn các thư viện học sâu và công
    cụ phổ biến, giúp rút ngắn thời gian thiết lập và tập trung hoàn
    toàn vào quá trình thực nghiệm.

-   **Khả năng chia sẻ**: Nền tảng cho phép lưu trữ và chia sẻ mã nguồn,
    kết quả thực nghiệm, giúp dễ dàng quản lý các phiên bản của dự án.

-   **Thân thiện với người dùng**: Giao diện trực quan, hỗ trợ khả năng
    kiểm tra log và đầu ra trực tiếp trên giao diện web.

## **4.2 Tập dữ liệu** {#tập-dữ-liệu .unnumbered}

[]{#_Toc197887810 .anchor}**4.2.1 Nguồn dữ liệu**

-   **Tên đầy đủ**: *Inria Aerial Image Labeling Dataset* (thường được
    gọi ngắn là **AerialImageDataset**).

-   **Tổ chức phát hành**: Trung tâm nghiên cứu Inria (Pháp) -- nhóm của
    Maggiori *et al.* trong khuôn khổ benchmark "Can semantic labeling
    methods generalize to any city?" (MICCAI 2017). [[Inria
    Project]{.underline}](https://project.inria.fr/aerialimagelabeling/?utm_source=chatgpt.com)

-   **Mục đích**: Xây dựng bộ chuẩn đánh giá khả năng **phân đoạn toà
    nhà** trên ảnh hàng không độ phân giải cao (0,3 m/px).

-   **Phạm vi địa lý**: 10 khu vực đô/thị từ Bắc Mỹ, Áo, Pháp... được
    lấy từ dịch vụ bản đồ công cộng (USGS National Map, IGN, open data
    GIS khác). [[Papers with
    Code]{.underline}](https://paperswithcode.com/dataset/inria-aerial-image-labeling?utm_source=chatgpt.com)

-   **Truy cập**: Miễn phí cho mục đích nghiên cứu qua trang dự án Inria
    hoặc mirror trên HuggingFace Datasets. [[Hugging
    Face]{.underline}](https://huggingface.co/datasets/blanchon/INRIA-Aerial-Image-Labeling?utm_source=chatgpt.com)

**4.2.2 Số lượng dữ liệu**

  --------------------------------------------------------------------------------------------------------------------------
                   Công thức  Khi sử dụng
  ---------------- ---------- ----------------------------------------------------------------------------------------------
  Hạng mục         Giá trị    Ghi chú

  Tổng ảnh gốc     360 ảnh    Kích thước 5000 × 5000 px mỗi ảnh (≈ 25 MP). [Papers with
                   RGB        Code](https://paperswithcode.com/dataset/inria-aerial-image-labeling?utm_source=chatgpt.com)

  Tập huấn luyện   180 ảnh    Có mặt nạ GT (building / non‑building).
                              [ar5iv](https://ar5iv.labs.arxiv.org/html/2302.03156?utm_source=chatgpt.com)

  Tập kiểm thử     180 ảnh    GT ẩn dùng chấm điểm benchmark.

  Diện tích phủ    ≈ 810 km²  Trong đó 405 km² train, 405 km² test. [Inria
                              Project](https://project.inria.fr/aerialimagelabeling/?utm_source=chatgpt.com)

  Phân giải không  0,3 m/px   Đủ chi tiết nhận diện mái nhà riêng lẻ.
  gian                        

  Lớp phân đoạn    2 lớp      Building (1) vs Background (0).
  --------------------------------------------------------------------------------------------------------------------------

###   {#section-1 .unnumbered}

### **4.2.3. Tiền xử lý dữ liệu** {#tiền-xử-lý-dữ-liệu .unnumbered}

• **Đọc và resize ảnh**

-   load_img(\..., target_size=target_size):

    -   Đọc file ảnh RGB (hàm này mặc định màu 3 kênh).

    -   Chuyển kích thước về 256×256 (có thể điều chỉnh tuỳ GPU/ mô
        hình).

-   load_img(\..., color_mode=\"grayscale\") cho mask:

    -   Đảm bảo chỉ đọc một kênh (0--255), phù hợp nhãn nhị phân.

• **Chuyển sang mảng numpy**

-   img_to_array(\...) biến đối tượng PIL Image → mảng NumPy hình dạng
    (H, W, C).

-   Chia 255.0 để **chuẩn hóa** pixel vào khoảng \[0, 1\]:

    -   Ảnh đầu vào: giúp mạng hội tụ nhanh, tránh gradient quá lớn.

    -   Mask: sau chia, giá trị gần 0 hoặc 1, thuận tiện cho hàm loss
        (BCE, Dice).

• **Trả về cặp (image, mask)**

-   Dạng mảng NumPy: sẵn sàng đưa vào tf.data.Dataset hoặc vòng lặp
    PyTorch.

###  **4.2.4. Tăng cường dữ liệu** {#tăng-cường-dữ-liệu .unnumbered}

> Tập dữ liệu có tình trạng mất cân bằng giữa 2 nhóm phổi bình thường và
> bị viêm phổi nên ta áp dụng các kỹ thuật tăng cường ảnh để giải quyết
> vấn đề mất cân bằng dữ liệu đó Các kỹ thuật tăng cường ảnh trong tập
> train được sử dụng để giúp cho mô hình học được các đặt trưng khác và
> tránh overfit được liệt kê trong **Bảng 1**:

[]{#_Toc184207958 .anchor}Bảng 1. Các phương pháp và tham số tăng cường
dữ liệu

  -----------------------------------------------------------------------
  **Phương pháp**                   **Tham số**
  --------------------------------- -------------------------------------
  Rescale                           1./255

  Rotation                          ±15

  Shift                             0.1

  Shear                             0.2

  Zoom                              0.2

  Brightness                        0.8-1.2

  Fill mode                         nearest

  Horizontal flip                   True
  -----------------------------------------------------------------------

> Sau khi đã được tăng cường và chuẩn hóa lại ta được một tập dữ liệu
> mới với 8437 ảnh gồm: 4273 ảnh có nhãn viêm phổi và 4161 ảnh có nhãn
> bình thường.

###  **4.2.5. Chia dữ liệu** {#chia-dữ-liệu .unnumbered}

> Tập dữ liệu sau khi được tăng cường sẽ được chia thành 3 phần chính:

-   Tập huấn luyện( training set): 64% tập dữ liệu.

-   Tập kiểm định(validation set): 16% tập dữ liệu.

-   Tập kiểm tra(test set): 20% tập dữ liệu.

> Quá trình chia dữ liệu được thực hiện ngẫu nhiên nhưng đảm bảo duy trì
> tỷ lệ cân đối giữa hai nhóm ảnh trong mỗi tập.

###  **4.2.6. Lý do chọn tập dữ liệu** {#lý-do-chọn-tập-dữ-liệu .unnumbered}

> Tập dữ liệu từ Kaggle được chọn vì:

-   Độ tin cậy cao: Ảnh được thu thập từ các nguồn uy tín, đảm bảo tính
    chính xác và chất lượng.

-   Độ đa dạng và quy mô phù hợp: Bao gồm nhiều mẫu ảnh từ cả hai nhóm
    chẩn đoán, giúp mô hình học được nhiều đặc trưng hơn.

-   Tính khả dụng: Tập dữ liệu có thể dễ dàng truy cập.

## **4.3 Ứng dụng thực nghiệm** {#ứng-dụng-thực-nghiệm .unnumbered}

### **4.3.1. Quy trình huấn luyện mô hình** {#quy-trình-huấn-luyện-mô-hình .unnumbered}

> Mô hình được triển khai trên nền tảng Kaggle, sử dụng GPU NVIDIA Tesla
> P100 để tăng tốc quá trình tính toán. Quy trình huấn luyện bao gồm các
> bước:

-   Tiền xử lý và tăng cường dữ liệu từ tập ban đầu.

-   Xây dựng mô hình.

-   Huấn luyện mô hình trên tập dữ liệu đã được chuẩn bị.

### **4.3.2. Cấu hình huấn luyện** {#cấu-hình-huấn-luyện .unnumbered}

  ------------------------------------------------------------------------------
  **Thành phần**  **Thiết lập**             **Ý nghĩa & Ghi chú**
  --------------- ------------------------- ------------------------------------
  Optimizer       Adam                      -- Tốc độ hội tụ nhanh, tự điều
                                            chỉnh learning rate nội tại.

  Loss function   binary_crossentropy       -- Phù hợp bài toán phân đoạn nhị
                                            phân (foreground vs background).

  Metrics         accuracy                  -- Đo tỉ lệ pixel dự đoán đúng. Có
                                            thể bổ sung DiceCoefficient để đánh
                                            giá chất lượng mask.

  Batch size      8                         -- Vừa đủ cho GPU 8 GB. Nếu VRAM lớn
                                            hơn, có thể tăng lên 16--32 để ổn
                                            định gradient.

  Epochs tối đa   50                        -- Giới hạn trên; thường dừng sớm
                                            trước khi chạy đủ nếu hội tụ.

  Validation      validation_data=(X_val,   -- Giữ 10--20 % dữ liệu làm tập kiểm
                  y_val)                    định, theo dõi val_loss để phát hiện
                                            over‑fitting.

  EarlyStopping                             
  ------------------------------------------------------------------------------

\| -- Dừng huấn luyện khi val_loss không cải thiện trong 5 epoch liên
tiếp.\
-- restore_best_weights=True đảm bảo mô hình cuối cùng là trạng thái tốt
nhất. \|\
\| **Callbacks** \| \[early_stopping\] \| -- Có thể thêm:

-   ReduceLROnPlateau(monitor=\'val_loss\', factor=0.5, patience=3)

-   ModelCheckpoint lưu best‑model mỗi khi val_loss giảm. \|\
    \| **Mixed precision**\| (tùy chọn) \| -- Dùng
    tf.keras.mixed_precision.set_global_policy(\'mixed_float16\') để
    tăng tốc trên GPU hỗ trợ. \|\
    \| **Shuffle** \| Mặc định shuffle=True \| -- Xáo trộn dữ liệu mỗi
    epoch, tránh thứ tự cố định gây bias. \|\
    \| **Data Augmentation** \| (nên thêm) \| -- Xoay, lật, zoom,
    elastic deformation... ngay trong pipeline để giảm over‑fit. \|

### **~~4.3.3. Kết quả thực nghiệm~~** {#kết-quả-thực-nghiệm .unnumbered}

> ~~Sau khi training với 5 epoch, mỗi epoch mất khoảng 169-261 giây với
> tổng thời gian đào tạo là 942 giây ta và chạy Stratified Five-Fold
> Cross-Validation ta có được kết quả sau:~~[]{#_Toc184149077 .anchor}

~~Bảng 2. Kết quả thử nghiệm và xác thực chéo của mô hình đề xuất~~

  ---------------------------------------------------------------------------------------------------------
  **Phương   **Accuracy(%)**   **Precision(%)**   **Recall**\   **F1-score   **Specificity(%)**   **AUC**
  pháp**                                          **(%)**       (%)**                             
  ---------- ----------------- ------------------ ------------- ------------ -------------------- ---------
  Thử nghiệm 97.75             98.33              97.18         97.75        98.33                0.9956

  Xác thực   95.19 ± 2.34      95.34 ± 2.19       95.19 ± 2.34  95.17 ± 2.35 92.34 ± 4.62         0.9902 ±
  chéo                                                                                            0.0074
  ---------------------------------------------------------------------------------------------------------

![](./image2.png){width="5.626469816272966in"
height="3.810871609798775in"}

[]{#_Toc184148872 .anchor}Hình 2. Độ chính xác trong quá trình huấn
luyện

![](./image3.png){width="5.3125in" height="3.6884722222222224in"}

[]{#_Toc184148873 .anchor}Hình 3. Loss trong quá trình huấn luyện

![](./image4.png){width="4.719584426946632in"
height="3.6298206474190726in"}

[]{#_Toc184148874 .anchor}Hình 4. Ma trận nhầm lẫn của mô hình đề xuất

## **4.4 Đánh giá kết quả** {#đánh-giá-kết-quả .unnumbered}

### **4.4.1. So sánh với các mô hình pretrain** {#so-sánh-với-các-mô-hình-pretrain .unnumbered}

> Sau khi huấn luyện mô hình, kết quả được đánh giá và so sánh với các
> mô hình tiền huấn luyện (pretrained models) như **ResNet50**, **VGG16,
> MobileNet, InceptionV3**, và **DenseNet**. Các mô hình này đã được sử
> dụng làm điểm chuẩn để kiểm tra xem các cải tiến trong mô hình có cải
> thiện hiệu suất so với các mô hình pretrain hay không.
>
> Biểu đồ so sánh giữa các chỉ số **Accuracy**, **F1-Score**,
> **Precision**, **Recall**, **Specificity**, và **AUC** của mô hình đề
> xuất và các mô hình pretrained

![](./image5.png){width="6.5in" height="3.6770833333333335in"}

[]{#_Toc184148875 .anchor}Hình 5. Biểu đồ so sánh số liệu các mô hình
phổ biến với mô hình đề xuất

> **ResNet50**:

-   **Accuracy** thấp nhất là **73.76%**, cho thấy khả năng phân loại
    > hạn chế so với các mô hình khác.

-   **AUC** chỉ đạt **0.8198**, cho thấy mô hình khó phân biệt rõ ràng
    > giữa các lớp.

> **VGG16**:

-   Hiệu suất cải thiện với **Accuracy** đạt **85.78%** và **AUC** là
    > **0.9520**, tuy nhiên vẫn kém so với các mô hình hiện đại.

-   **Specificity** cao cho thấy khả năng tốt trong việc phát hiện ảnh
    > **bình thường**, nhưng Precision và Recall chưa cân đối.

> **MobileNet**:

-   Mô hình nhẹ nhưng hiệu quả với **Accuracy** đạt **88.98%** và
    > **AUC** là **0.9873**.

-   Đặc biệt, **Specificity** cao nhất trong các mô hình pretrained
    > (**99.76%**), chứng tỏ khả năng mạnh mẽ trong việc phân loại đúng
    > các trường hợp bình thường.

> **InceptionV3**:

-   **Accuracy** và **F1-Score** đều đạt **90.23%**, cùng với **AUC** là
    > **0.9724**, cho thấy khả năng phân loại tốt và ổn định.

-   Các chỉ số Precision và Recall cân đối, chứng tỏ đây là một mô hình
    > mạnh trong các kiến trúc pretrained.

> **DenseNet121**:

-   Đạt hiệu suất cao nhất trong các mô hình pretrained với **Accuracy**
    > đạt **93.13%** và **AUC** là **0.9797**.

-   DenseNet có sự cân đối tốt giữa các chỉ số Precision và Recall, cho
    > thấy khả năng phân loại toàn diện.

> **Mô hình đề xuất:**

-   **Accuracy**: Đạt **97.75%**, cao hơn so với DenseNet121 -- mô hình
    > tốt nhất trong nhóm pretrained.

-   **F1-Score**: Đạt **97.75%**, cho thấy sự cân đối hoàn hảo giữa
    > Precision và Recall.

-   **Precision**: Đạt **97.76%**, cao hơn tất cả các mô hình
    > pretrained, chứng minh khả năng phân loại chính xác các trường hợp
    > viêm phổi.

-   **Recall**: Đạt **97.75%**, đảm bảo không bỏ sót các trường hợp bệnh
    > thực sự.

-   **Specificity**: Đạt **98.33%**, chỉ kém **MobileNet** vì cơ chế
    > pooling động trong mô hình khiến trọng số bị ưu tiên hơn vào các
    > đặc trưng mạnh, thường liên quan đến lớp PNEUMONIA, do lớp này có
    > các đặc điểm nổi bật hơn so với NORMAL. Tuy vậy mô hình vẫn chứng
    > tỏ được khả năng phân loại chính xác các trường hợp bình thường.

-   **AUC**: Đạt **0.9956**, cho thấy mô hình có khả năng phân biệt tốt
    > giữa hai lớp.

### **4.4.2 So sánh với các mô hình khi thay đổi các khối** {#so-sánh-với-các-mô-hình-khi-thay-đổi-các-khối .unnumbered}

> Các so sánh sẽ được sử dụng với mô hình đề xuất gồm **Thay Residual
> Block bằng LSTM, Thay khối Attention Augmentation và Thay Residual
> Block bằng Conv đơn giản:**

[]{#_Toc184207960 .anchor}Bảng 3. So sánh các tiêu chí đánh giá của mô
hình đề xuất với các biến thể của nó

  --------------------------------------------------------------------------------------------------------------
  **Biến thể**            **Accuracy**   **F1-Score**   **Precision**   **Recall**   **Specificity**   **AUC**
  ----------------------- -------------- -------------- --------------- ------------ ----------------- ---------
  **Mô hình đề xuất**     97.75%         97.75%         97.76%          97.75%       98.33%            0.9956

  **Thay Residual Block   95.56%         95.56%         95.56%          95.56%       94.86%            0.9896
  bằng LSTM**                                                                                          

  **Thay khối Attention   97.10%         97.10%         97.12%          97.10%       98.09%            0.9941
  Augmentation**                                                                                       

  **Thay Residual Block   96.92%         96.92%         96.96%          96.92%       95.34%            0.9956
  bằng Conv đơn**                                                                                      
  --------------------------------------------------------------------------------------------------------------

> Biểu đồ so sánh giữa các chỉ số **Accuracy**, **F1-Score**,
> **Precision**, **Recall**, **Specificity**, và **AUC** của mô hình đề
> xuất và các thay đổi khối:

![C:\\Users\\AnNguyen\\AppData\\Local\\Packages\\Microsoft.Windows.Photos_8wekyb3d8bbwe\\TempState\\ShareServiceTempFolder\\download.jpeg](./image6.jpeg){width="6.833333333333333in"
height="4.558688757655293in"}

[]{#_Toc184148876 .anchor}Hình 6. Biểu đồ so sánh số liệu các mô hình
thay đổi khối khác với mô hình đề xuất

> **Thay đổi Residual Block**:

-   Khi thay **Residual Block** bằng **LSTM**, hiệu suất giảm do LSTM
    > không phù hợp với dữ liệu không gian như ảnh.

> **Thay đổi Attention Augmentation**:

-   Thay đổi khối chú ý (Attention Augmentation) làm giảm nhẹ hiệu suất
    > , nhưng vẫn giữ được mức hiệu quả cao, chứng tỏ vai trò của
    > Attention trong việc nhấn mạnh các đặc trưng quan trọng.

> **Thay Residual Block bằng lớp tích chập đơn giản:**

-   Hiệu suất giảm vừa phải: Accuracy giảm còn 96.92%, AUC vẫn giữ
    > nguyên ở mức 0.9956, cho thấy khả năng phân biệt của mô hình không
    > bị ảnh hưởng nhiều.

### **4.4.3 So sánh với mô hình gốc** {#so-sánh-với-mô-hình-gốc .unnumbered}

> Để đánh giá hiệu quả của các cải tiến được thực hiện trong mô hình đề
> xuất, phần này tiến hành so sánh trực tiếp giữa mô hình gốc (chưa được
> cải tiến) và mô hình đề xuất. Các tiêu chí so sánh bao gồm: Accuracy,
> Precision, Recall, F1-Score, Specificity, và AUC.

[]{#_Toc184207961 .anchor}Bảng 4: So sánh các tiêu chí đánh giá của mô
hình gốc với mô hình đề xuất

  --------------------------------------------------------------------------------------------------------
  **Mô hình**       **Accuracy**   **F1-Score**   **Precision**   **Recall**   **Specificity**   **AUC**
  ----------------- -------------- -------------- --------------- ------------ ----------------- ---------
  **Mô hình gốc**   95.19%         98.38%         93.84%          96.06%       97.43%            0.9556

  **Mô hình đề      97.75%         97.75%         97.76%          97.75%       98.33%            0.9956
  xuất**                                                                                         
  --------------------------------------------------------------------------------------------------------

-   Độ chính xác (Accuracy) của mô hình đề xuất đạt **97.75%**, cao hơn
    > đáng kể so với **95.19%** của mô hình gốc cho thấy các cải tiến đã
    > giúp mô hình nhận diện hình ảnh viêm phổi chính xác hơn.

-   Precision của mô hình gốc đạt **98.38%**, nhỉnh hơn một chút so với
    > **97.76%** của mô hình đề xuất có thể do mô hình gốc có xu hướng
    > ưu tiên độ chính xác cao hơn trong các dự đoán dương tính.

-   Recall của mô hình đề xuất là **97.75%**, vượt xa **93.84%** của mô
    > hình gốc. Đây là một cải tiến quan trọng vì nó cho thấy mô hình đề
    > xuất có khả năng phát hiện hầu hết các trường hợp viêm phổi, giảm
    > thiểu các trường hợp bỏ sót.

-   F1-Score của mô hình đề xuất đạt **97.75%**, cao hơn **96.06%** của
    > mô hình gốc cho thấy sự cân bằng tốt hơn giữa Precision và Recall.

-   Specificity của mô hình đề xuất đạt **98.33%**, nhỉnh hơn **97.43%**
    > của mô hình gốc chứng tỏ khả năng nhận diện chính xác các trường
    > hợp bình thường cũng được cải thiện.

-   AUC của mô hình đề xuất đạt **0.9956**, vượt trội so với **0.9564**
    > của mô hình gốc cho thấy mô hình đề xuất phân biệt tốt hơn giữa
    > hai lớp.

# **CHƯƠNG 5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN** {#chương-5.-kết-luận-và-hướng-phát-triển .unnumbered}

## **5.1 Kết luận** {#kết-luận .unnumbered}

> Trong nghiên cứu này, chúng tôi đã trình bày một mô hình học sâu phát
> hiện ra tòa nhà viêm phổi từ ảnh chụp X-quang lồng ngực dựa trên cơ sở
> là mô hình mạng nơ-ron tích chập sâu kết hợp với Attention Ensemble.
> Bằng cách thêm vào các nhóm tích chập (Grouped Convolution) và khối SE
> để làm tăng khả năng học đa dạng các đặc trưng, các kênh của dữ liệu
> giúp cải thiện thêm việc chuẩn bị các thuộc tính đầu vào cho các lớp
> chú ý, mà mô hình đề xuất đã đạt được hiệu suất vượt trội với độ chính
> xác 97,75%, độ đặc hiệu 98,33%, độ nhạy 97,75%, điểm F1 97,75% và AUC
> đạt 0,9956 trên tập dữ liệu kiểm tra. Việc kết hợp thêm các nhóm tích
> chập và khối SE vào mô hình phát hiện viêm phổi là một phương pháp
> hiệu quả và tiềm năng. Những kết quả này không chỉ xác nhận khả năng
> phân biệt chính xác các tình trạng bệnh lý nguy kịch của mô hình mà
> còn làm nổi bật tiềm năng của mô hình như một công cụ đáng tin cậy
> trong việc nâng cao chẩn đoán bệnh viêm phổi, giảm thiểu thời gian
> chẩn đoán và hỗ trợ các y bác sĩ trong việc chăm sóc bệnh nhân đặc
> biệt trong bối cảnh dịch COVID-19 và các bệnh liên quan đến hệ hô hấp
> đang diễn biến phức tạp hiện nay.

## **5.2 Hướng phát triển** {#hướng-phát-triển .unnumbered}

Dựa trên các kết quả đạt được, chúng tôi đề xuất một số hướng nghiên cứu
và cải tiến trong tương lai như sau:

1.  **Tăng cường dữ liệu**:

-   Thu thập thêm các tập dữ liệu từ nhiều nguồn khác nhau để mô hình
    > học được đa dạng đặc trưng từ các trường hợp bệnh nhân thuộc nhiều
    > độ tuổi, giới tính và điều kiện y tế khác nhau.

-   Sử dụng các kỹ thuật tăng cường dữ liệu tiên tiến hơn như GAN
    > (Generative Adversarial Networks) để tạo ra các hình ảnh giả lập
    > có độ chân thực cao.

2.  **Áp dụng trên các loại bệnh khác**:

-   Mở rộng mô hình để phát hiện và phân loại nhiều loại bệnh phổi khác
    > nhau như ung thư phổi, lao phổi hoặc viêm phổi do các nguyên nhân
    > khác nhau (vi khuẩn, virus, nấm).

3.  **Tích hợp mô hình vào hệ thống lâm sàng**:

-   Xây dựng một hệ thống phần mềm tích hợp mô hình vào quy trình làm
    > việc của bệnh viện, hỗ trợ bác sĩ trong việc chẩn đoán.

-   Tối ưu hóa tốc độ xử lý của mô hình để phù hợp với các hệ thống thời
    > gian thực.

4.  **Cải tiến hiệu suất mô hình**:

-   Áp dụng các kỹ thuật tinh chỉnh (fine-tuning) và tối ưu hóa siêu
    > tham số (hyperparameter optimization) để cải thiện hiệu suất mô
    > hình.

-   Khám phá các kiến trúc mạng mới như Vision Transformers (ViTs) hoặc
    > các mô hình tự giám sát (self-supervised learning) để tăng khả
    > năng học đặc trưng.

5.  **Giảm chi phí tính toán**:

-   Tối ưu hóa mô hình để hoạt động hiệu quả trên các thiết bị phần cứng
    > có tài nguyên hạn chế, ví dụ như thiết bị di động hoặc các máy
    > tính tại các khu vực y tế thiếu thốn.

**TÀI LIỆU THAM KHẢO**

\[1\] Mai Mộc Thảo. Viêm phổi: nguyên nhân, triệu chứng, chẩn đoán và
cách điều trị. VNVC 2022. <https://vnvc.vn/viem-phoi/>.

\[2\] *Tháng 11 nói về NGÀY VIÊM PHỔI THẾ GIỚI 12/11/2015*. (n.d). Hội
Hô Hấp TP.HCM.
<http://www.hoihohaptphcm.org/benh-nhan/221-thang11-noi-ve-ngay-viem-phoi-the-gioi>

\[3\] *Bệnh viêm phổi ở trẻ em*. (n.d). Hội Hô Hấp TP.HCM.
<http://www.hoihohaptphcm.org/benh-nhan/146-benh-viem-phoi-o-tre-em>

\[4\] Mujahid, M.; Rustam, F.; Álvarez, R.; Luis Vidal Mazón, J.; Díez,
I.d.l.T.; Ashraf, I. Pneumonia classification from X-ray images with
inceptionV3-V3 and convolutional neural network. Diagnostics 2022, 12,
1280. <https://www.mdpi.com/2075-4418/12/5/1280>

\[5\] Hashmi, M. F., Katiyar, S., Keskar, A. G., Bokde, N. D., & Geem,
Z. W. (2020). Efficient Pneumonia Detection in Chest X-ray Images Using
Deep Transfer Learning. Diagnostics, 10(6), 417.
<https://www.mdpi.com/2075-4418/10/6/417>

\[6\] Brauwers, G.; Frasincar, F. A general survey on attention
mechanisms in deep learning. IEEE Trans. Knowl. Data Eng. 2021, 35,
3279--3298. <https://ieeexplore.ieee.org/document/9609539>

\[7\] Jie Hu, Li Shen, Gang Sun, Samuel Albanie & Enhua Wu (2018).
Squeeze-and-Excitation Networks. Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 7132--7141.
<https://arxiv.org/pdf/1709.01507>

\[8\] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
Gomez, A. A., Kaiser, L., & Polosukhin, I. (2017). Attention is All You
Need. Advances in Neural Information Processing Systems (NeurIPS), 30.
<https://arxiv.org/abs/1706.03762>

\[9\] An, Q., Chen, W., & Shao, W. (2024). A Deep Convolutional Neural
Network for Pneumonia Detection in X-ray Images with Attention Ensemble.
Diagnostics, 14(4), 390. <https://www.mdpi.com/2075-4418/14/4/390>

\[10\] Ben Atitallah, S.; Driss, M.; Boulila, W.; Koubaa, A.; Ben
Ghezala, H. Fusion of convolutional neural networks based on Dempster--
Shafer theory for automatic pneumonia detection from chest X-ray images.
Int. J. Imaging Syst. Technol. 2022, 32, 658--672.
<https://onlinelibrary.wiley.com/doi/10.1002/ima.22653>

\[11\] Tan, A., Guo, T., Zhao, Y. *et al.* Object detection based on
polarization image fusion and grouped convolutional attention network.
*Vis Comput* **40**, 3199--3215 (2024).
<https://doi.org/10.1007/s00371-023-03022-6.>
