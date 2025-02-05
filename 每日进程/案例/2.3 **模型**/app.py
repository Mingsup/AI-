from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import cv2
import torchvision.transforms as T
import os
import io
import base64

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 加载模型（全局变量，这样只需要加载一次）
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = None

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_model():
    global model
    if model is None:
        model = get_model(num_classes=2)
        checkpoint = torch.load('models/cat_detector_epoch_10.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    return model

def detect_cats(image, confidence_threshold=0.5):
    # 确保模型已加载
    model = load_model()
    
    # 转换图片
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 进行检测
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # 转换图片为OpenCV格式
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 检测到的猫的数量
    cat_count = 0
    detections = []
    
    # 在图片上标注检测结果
    for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
        if score > confidence_threshold:
            cat_count += 1
            x1, y1, x2, y2 = box.cpu().numpy()
            # 画矩形框
            cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # 添加置信度文本
            text = f'Cat: {score:.2f}'
            cv2.putText(img_cv, text, (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            detections.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(score)
            })
    
    # 在图片左上角显示检测到的猫的总数
    cv2.putText(img_cv, f'Total Cats: {cat_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 将结果图片转换为base64
    _, buffer = cv2.imencode('.jpg', img_cv)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        'cat_count': cat_count,
        'detections': detections,
        'image': img_base64
    }

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # 获取上传的图片
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '没有选择图片'}), 400
        
        # 读取和处理图片
        image = Image.open(file.stream).convert('RGB')
        
        # 进行检测
        result = detect_cats(image)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import numpy as np
    app.run(debug=True, port=5000)