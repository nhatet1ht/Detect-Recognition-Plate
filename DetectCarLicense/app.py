from flask import Flask, render_template, request, send_from_directory, jsonify
import cv2
from ultralytics import YOLO
import os
import easyocr
import re

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

model = YOLO('C:/diagram/plate_yolov8n.pt')
reader = easyocr.Reader(['vi'])

COUNT = 0

def clean_text(text):
    # Loại bỏ các ký hiệu đặc biệt chỉ giữ lại chữ cái và chữ số
    cleaned_text = re.sub(r'[^A-Za-z0-9]', '', text)
    return cleaned_text

def replace_characters(text):
    # Chuyển đổi ký tự theo yêu cầu
    char_map_pos3 = {'0': 'D', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S','n':'A','g':'G','h':'H'}
    char_map_others = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'D': '0','o':'0'}
    
    text = list(text)  # Chuyển chuỗi thành danh sách ký tự để dễ dàng thao tác
    
    if len(text) > 2 and text[2] in char_map_pos3:
        text[2] = char_map_pos3[text[2]]
    
    for i in range(len(text)):
        if i != 2 and text[i] in char_map_others:
            text[i] = char_map_others[text[i]]
    
    return ''.join(text)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']
    if not img:
        return jsonify({"error": "No file provided or file name is empty"}), 400

    img_path = f'static/{COUNT}.jpg'
    img.save(img_path)    
    img_arr = cv2.imread(img_path)
    
    results = model(img_path)
    if len(results) == 0 or len(results[0].boxes) == 0:
        return jsonify({"error": "No license plate detected"}), 400
    
    # Assuming the first detection is the license plate
    detection = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, detection.xyxy[0])
    
    # Crop the license plate from the image
    plate_img = img_arr[y1:y2, x1:x2]
    cropped_plate_path = 'static/cropped_plate.jpg'
    cv2.imwrite(cropped_plate_path, plate_img)
    
    detected_img_path = f'runs/detect/predict/{COUNT}.jpg'
    os.makedirs(os.path.dirname(detected_img_path), exist_ok=True)

    # Vẽ khung bao quanh biển số xe
    cv2.rectangle(img_arr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(detected_img_path, img_arr)
    
    ocr_results = reader.readtext(plate_img)
    combined_text = ""
    for (bbox, text, prob) in ocr_results:
        cleaned = clean_text(text)
        if cleaned:
            combined_text += cleaned
    
    # Thực hiện biến đổi sau khi gộp các chuỗi
    final_text = replace_characters(combined_text)
    
    COUNT += 1
    return render_template('detection.html', license_plate_text=final_text)

@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('runs/detect/predict', f'{COUNT-1}.jpg')

@app.route('/img')
def img():
    global COUNT
    return send_from_directory('static', f"{COUNT-1}.jpg")

if __name__ == '__main__':
    app.run(debug=True)
