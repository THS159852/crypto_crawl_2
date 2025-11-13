# Hướng dẫn Deploy Bot Telegram lên Render (Miễn phí)

Đây là các bước chi tiết để đưa bot crypto của bạn lên mạng và chạy 247. Chúng ta sẽ sử dụng dịch vụ [Render](httpsrender.com) vì nó có gói miễn phí rất phù hợp cho bot (không giống Vercel).

## Bước 1 Chuẩn bị mã nguồn trên GitHub

Render hoạt động bằng cách kết nối với một kho lưu trữ (repository) trên GitHub.

1.  Tạo tài khoản GitHub Nếu bạn chưa có, hãy tạo một tài khoản miễn phí tại [github.com](httpsgithub.com).
2.  Tạo kho lưu trữ mới (New Repository)
     Trên trang chủ GitHub, nhấn vào dấu `+` ở góc trên bên phải và chọn New repository.
     Đặt tên cho nó, ví dụ `crypto-telegram-bot`.
     Chọn Private (Riêng tư) để giữ an toàn cho code của bạn.
     Nhấn Create repository.
3.  Tải 2 file lên kho lưu trữ
     Trong kho lưu trữ mới tạo, nhấn vào Add file - Upload files.
     Tải lên 2 file sau từ máy tính của bạn
        1.  `crypto_bot.py` (File script bot ở trên).
        2.  `requirements.txt` (File liệt kê các thư viện ở trên).
     Sau khi tải lên, nhấn Commit changes.

Bây giờ bạn đã có một kho lưu trữ trên GitHub chứa 2 file cần thiết.

## Bước 2 Cấu hình và triển khai trên Render

1.  Tạo tài khoản Render
     Truy cập [render.com](httpsrender.com) và đăng ký một tài khoản mới.
     Kết nối tài khoản GitHub của bạn với Render khi được yêu cầu.

2.  Tạo Dịch vụ mới (New Service)
     Trên trang Dashboard của Render, nhấn New + - Background Worker. (Lưu ý Không chọn Web Service).
     Kết nối kho lưu trữ GitHub của bạn. Tìm và chọn kho `crypto-telegram-bot` mà bạn vừa tạo.
     Đặt một tên duy nhất cho dịch vụ của bạn (ví dụ `my-crypto-bot`).

3.  Điền thông tin cài đặt
     Region Chọn `Singapore (Southeast Asia)` để có tốc độ nhanh nhất từ Việt Nam.
     Branch Để là `main` (hoặc `master` tùy tên nhánh của bạn).
     Build Command Gõ vào `pip install -r requirements.txt`
     Start Command Gõ chính xác lệnh sau vào ô này `python crypto_bot.py`

4.  Thêm Biến Môi trường (Rất quan trọng)
     Kéo xuống phần Advanced và nhấn vào Add Environment Variable.
     Bạn cần thêm 2 biến sau
         Biến 1 (BOT_TOKEN)
             Key `BOT_TOKEN`
             Value `YOUR_BOT_TOKEN` (Dán Bot Token thật của bạn vào đây).
         Biến 2 (Fix lỗi font) Biến này để đảm bảo thư viện `matplotlib` tìm thấy font chữ và vẽ biểu đồ không bị lỗi.
             Key `MPLCONFIGDIR`
             Value `tmpmatplotlib`

5.  Chọn Gói miễn phí (Free Plan)
     Ở cuối trang, đảm bảo bạn đã chọn gói Free.

6.  Khởi chạy
     Nhấn nút Create Background Worker.

Render sẽ bắt đầu xây dựng dịch vụ. Quá trình này sẽ mất vài phút vì nó cần cài đặt các thư viện `pandas`, `ccxt`, `mplfinance`...

## Bước 3 Kiểm tra và Hoàn tất

 Sau khi xây dựng xong, bạn có thể nhấn vào tab Logs để xem log chạy của bot.
 Nếu bạn thấy các dòng chữ như `Bắt đầu khởi tạo bot Telegram...` và `Bot đã sẵn sàng và đang lắng nghe...`, điều đó có nghĩa là bot của bạn đã chạy thành công!
 Bây giờ, hãy vào Telegram và thử gõ một lệnh (ví dụ `start` hoặc `$btc`) để kiểm tra. Bot sẽ phản hồi và hoạt động 247.