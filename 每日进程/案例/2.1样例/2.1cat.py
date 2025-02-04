from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch

app = Flask(__name__)
CORS(app, resources={
    r"/detect_cat": {
        "origins": "*",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# 加载YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

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
        
        # 使用YOLOv5进行检测
        results = model(img)
        
        # 获取只包含猫的检测结果
        cat_detections = []
        for det in results.xyxy[0]:  # 遍历检测结果
            if int(det[5]) == 15:  # COCO数据集中猫的类别ID为15
                x1, y1, x2, y2, conf = det[:5]
                cat_detections.append([
                    int(x1.item()), int(y1.item()),
                    int(x2.item()-x1.item()), int(y2.item()-y1.item())
                ])
                # 在图片上画框
                cv2.rectangle(img, 
                            (int(x1.item()), int(y1.item())), 
                            (int(x2.item()), int(y2.item())), 
                            (0, 255, 0), 2)
        
        # 将处理后的图片转换回bytes
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = buffer.tobytes()
        
        # 返回处理后的图片和检测到的猫咪位置信息
        return jsonify({
            'processed_image': img_bytes.decode('latin1'),
            'cat_locations': cat_detections
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
