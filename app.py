from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# 모델 로드
model = load_model("musculoskeletal_model.h5")

# 클래스 이름 (형식: 1호: 키보드마우스)
class_names = [
    "1호: 키보드마우스",
    "2호: 반복동작",
    "3호: 팔머리위",
    "4호: 허리목굽힘",
    "5호: 쪼그리기",
    "6호: 손가락쥠",
    "7호: 한손무거움",
    "8호: 25kg10회",
    "9호: 10kg25회",
    "10호: 4.5kg반복",
    "11호: 충격작업"
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '이미지 파일이 필요합니다'}), 400

    file = request.files['file']
    try:
        # 이미지 처리
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # 예측
        predictions = model.predict(image)[0]
        top_idx = int(np.argmax(predictions))
        top_class = class_names[top_idx]
        confidence = float(predictions[top_idx])

        return jsonify({
            'label': top_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "근골격계 부담
