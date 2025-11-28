import os
import sys
from telegram.ext import Application
import json
import logging
from dotenv import load_dotenv # <--- ĐÃ THÊM DÒNG NÀY

# Bật logging để theo dõi lỗi
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# THAY ĐỔI ĐƯỜNG DẪN NÀY CHO ĐÚNG VỚI VỊ TRÍ FILE crypto_bot.py CỦA BẠN
path = '/home/yourusername/your-project-folder' 
if path not in sys.path:
    sys.path.append(path)

# --- THÊM PHẦN TẢI BIẾN MÔI TRƯỜNG TỪ FILE .ENV ---
try:
    # Tải biến môi trường từ file .env trong thư mục dự án
    load_dotenv(os.path.join(path, '.env')) 
    logging.info(".env file loaded from project directory.")
except Exception as e:
    logging.warning(f"Could not load .env file: {e}")
# ----------------------------------------------------

# Lấy BOT_TOKEN từ biến môi trường (Bây giờ bao gồm cả biến từ .env)
BOT_TOKEN = os.environ.get('BOT_TOKEN')
if not BOT_TOKEN:
    logging.error("BOT_TOKEN is not set in environment variables (or .env file).")

# Import hàm get_application từ file bot của bạn
try:
    from crypto_bot import get_application 
except ImportError as e:
    logging.error(f"Error importing bot: {e}")
    raise e

# Tạo bot application (LƯU Ý: Đây là đối tượng toàn cục)
try:
    application = get_application()
    logging.info("Telegram Application object initialized.")
except Exception as e:
    logging.error(f"Error calling get_application(): {e}")
    # Nếu có lỗi, vẫn khởi tạo một đối tượng application rỗng để tránh crash
    application = Application.builder().token(BOT_TOKEN or "DUMMY").build() 

# Hàm xử lý yêu cầu Webhook (Là entry point của WSGI)
def application(environ, start_response):
    """Xử lý yêu cầu Webhook từ Telegram."""
    
    # Lấy BOT_TOKEN từ môi trường (để dùng làm secret path)
    current_token = os.environ.get('BOT_TOKEN')
    WEBHOOK_PATH = f'/{current_token}'
    
    # Chỉ xử lý POST request và đường dẫn Webhook
    if environ['REQUEST_METHOD'] == 'POST':
        if environ['PATH_INFO'] == WEBHOOK_PATH:
            try:
                # Lấy nội dung request (Update từ Telegram)
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(request_body_size)
                update_data = json.loads(request_body.decode('utf-8'))
                
                # Đưa Update vào hàng đợi để bot xử lý bất đồng bộ
                application.update_queue.put(update_data)
                
                # Trả về phản hồi 200 OK ngay lập tức (quan trọng!)
                status = '200 OK'
                response_headers = [('Content-type', 'text/plain')]
                start_response(status, response_headers)
                return [b'OK']
                
            except Exception as e:
                logging.error(f"Error processing update: {e}")
                status = '500 Internal Server Error'
                start_response(status, [('Content-type', 'text/plain')])
                return [b'Error']

    # Xử lý các request khác (GET request, dùng cho UptimeRobot/Ping)
    status = '200 OK'
    response_headers = [('Content-type', 'text/plain')]
    start_response(status, response_headers)
    return [b'Hello, Bot is running!']