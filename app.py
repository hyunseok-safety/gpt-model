from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# 모델 로드
model = load_model("musculoskeletal_model.h5")

# 클래스 이름 정의
class_names = [
  "키보드마우스",   # 1호
  "반복동작",      # 2호
  "팔머리위",      # 3호
  "허리목굽힘",    # 4호
  "쪼그리기",      # 5호
  "손가락쥠",      # 6호
  "한손무거움",    # 7호
  "25kg10회",      # 8호
  "10kg25회",      # 9호
  "4.5kg반복",     # 10호
  "충격작업"       # 11호
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '이미지 파일이 필요합니다'}), 400

    file = request.files['file']
    try:
        # 이미지 전처리
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # 예측
        predictions = model.predict(image)[0]
        result = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
        top_class = class_names[np.argmax(predictions)]

        return jsonify({
            'top_prediction': top_class,
            'probabilities': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route("/", methods=["GET"])
def home():
    return "근골격계 부담작업 분류 모델 작동 중입니다."
