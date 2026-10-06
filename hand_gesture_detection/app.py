import gradio as gr
import cv2
import numpy as np

# Nhập (import) các module hiện có trong dự án của bạn
# Giả sử bạn có class Detector trong detector.py và hàm draw_landmarks trong visualizer.py
from detector import GestureDetector
from visualizer import draw_landmarks

# Khởi tạo mô hình AI (chỉ chạy 1 lần khi khởi động web)
model_path = 'gesture_recognizer.task'
detector = GestureDetector(model_path)

def process_frame(frame):
    """
    Hàm này sẽ nhận từng khung hình (frame) từ trình duyệt web của người dùng,
    phân tích cử chỉ, vẽ kết quả và trả lại khung hình đã xử lý.
    """
    # Gradio truyền ảnh dưới dạng mảng NumPy (RGB), OpenCV thường dùng (BGR)
    # Tuy nhiên MediaPipe yêu cầu định dạng RGB, nên bạn có thể truyền trực tiếp
    
    # Đưa khung hình qua file detector.py của bạn để nhận diện
    results = detector.recognize(frame)
    
    # Đưa kết quả và khung hình qua file visualizer.py để vẽ khung và chữ
    processed_image = draw_landmarks(frame, results)
    
    # Trả khung hình đã được vẽ kết quả về cho trình duyệt
    return processed_image

# Thiết lập giao diện web bằng Gradio
interface = gr.Interface(
    fn=process_frame, # Gọi hàm xử lý phía trên
    inputs=gr.Image(sources=["webcam"], streaming=True), # Mở webcam và truyền video trực tiếp
    outputs="image", # Hiển thị hình ảnh kết quả
    title="Nhận diện Cử chỉ Tay bằng AI",
    description="Vẫy tay hoặc làm các cử chỉ trước camera để hệ thống nhận diện theo thời gian thực.",
    live=True # Cập nhật kết quả liên tục không cần bấm nút Submit
)

if __name__ == "__main__":
    # Khởi chạy máy chủ web
    interface.launch()