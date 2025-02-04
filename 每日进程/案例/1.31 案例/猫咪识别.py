from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app, resources={
    r"/detect_cat": {
        "origins": "*",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# 加载预训练的猫咪检测模型
cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')

@app.route('/detect_cat', methods=['POST', 'OPTIONS'])
def detect_cat():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
        
    try:
        # 获取上传的图片文件
        file = request.files['image']
        
        # 将图片文件转换为OpenCV格式
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 转换为灰度图像进行检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 检测猫咪
        cats = cat_cascade.detectMultiScale(gray, 
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30, 30))
        
        # 在图片上标记猫咪位置
        for (x, y, w, h) in cats:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
        # 将处理后的图片转换回bytes
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = buffer.tobytes()
        
        # 返回处理后的图片和检测到的猫咪位置信息
        return jsonify({
            'processed_image': img_bytes.decode('latin1'),
            'cat_locations': cats.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
