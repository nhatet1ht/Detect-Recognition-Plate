import cv2
import torch
from ultralytics import YOLO
import pytesseract
from pytesseract import Output
import os
import re

def clean_text(text):
    # Loại bỏ các ký hiệu đặc biệt chỉ giữ lại chữ cái, chữ số, '-', và '.'
    cleaned_text = re.sub(r'[^A-Za-z0-9-.]', '', text)
    return cleaned_text

def detect_license_plate(image_path, output_dir):
    # Load mô hình YOLOv8
    model = YOLO('C:/diagram/plate_yolov8n.pt')

    # Đọc ảnh và phát hiện biển số
    img = cv2.imread(image_path)
    results = model(img)

    base_name = os.path.basename(image_path)
    output_image_path = os.path.join(output_dir, os.path.splitext(base_name)[0] + '_output.jpg')

    for result in results:  # Kết quả dự đoán
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Cắt biển số từ ảnh gốc
            plate_img = img[y1:y2, x1:x2]

            # Sử dụng Tesseract để đọc văn bản
            d = pytesseract.image_to_data(plate_img, output_type=Output.DICT)
            combined_text = '-'.join([clean_text(d['text'][i]) for i in range(len(d['text'])) if clean_text(d['text'][i])])

            # Vẽ bounding box và text lên ảnh gốc
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, combined_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Lưu ảnh đã ghi biển số lên bounding box
    cv2.imwrite(output_image_path, img)
    print(f"Saved output image to {output_image_path}")

def process_all_images(directory, output_dir):
    # Đảm bảo thư mục xuất tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Duyệt qua tất cả các file ảnh trong thư mục
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            detect_license_plate(image_path, output_dir)
            print(f"Processed {filename}")

# Đường dẫn tới thư mục chứa ảnh và thư mục xuất kết quả
input_dir = 'truth'
output_dir = 'img_tess'
process_all_images(input_dir, output_dir)
