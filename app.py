from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# 모델 로드
model = load_model("musculoskeletal_model.h5")

# 클래스 이름 (1호: 키보드마우스 형식)
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
    # 파일 유효성 검사
    file = request.files.get('file')
    if file is None:
        print(f"❌ 'file' 파라미터 없음. request.files.keys(): {list(request.files.keys())}")
        return jsonify({'error': "'file' 필드가 필요합니다"}), 400

    try:
        # 이미지 전처리
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # 예측
        predictions = model.predict(image)[0]
        top_idx = int(np.argmax(predictions))
        top_class = class_names[top_idx]
        confidence = float(predictions[top_idx])

        print(f"✅ 예측 완료: {top_class} ({confidence:.2f})")

        return jsonify({
            'label': top_class,
            'confidence': confidence
        })

    except Exception as e:
        print(f"❌ 예측 중 오류: {str(e)}")
        return jsonify({'error': f"이미지 처리 오류: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def home():
    return "근골격계 부담작업 분류 모델 작동 중입니다."
