import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import cv2
import torchvision.transforms as T
import os

def get_model(num_classes):
    # 加载预训练模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # 修改分类器以适应我们的类别数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def detect_cats(image_path, model_path, confidence_threshold=0.5):
    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = get_model(num_classes=2)  # 2类：背景和猫
    
    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 加载和转换图片
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 进行检测
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # 在图片上画框
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # 检测到的猫的数量
    cat_count = 0
    
    # 在图片上标注检测结果
    for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
        if score > confidence_threshold:
            cat_count += 1
            x1, y1, x2, y2 = box.cpu().numpy()
            # 画矩形框
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # 添加置信度文本
            text = f'Cat: {score:.2f}'
            cv2.putText(img, text, (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 在图片左上角显示检测到的猫的总数
    cv2.putText(img, f'Total Cats: {cat_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 保存结果图片
    output_path = f'result_{os.path.basename(image_path)}'
    cv2.imwrite(output_path, img)
    print(f"检测到 {cat_count} 只猫")
    print(f"结果已保存到: {output_path}")
    
    return cat_count, output_path

if __name__ == "__main__":
    # 使用最后一个epoch的模型（通常效果最好）
    model_path = 'models/cat_detector_epoch_10.pth'
    
    # 可以处理单张图片
    image_path = input("请输入要检测的图片路径: ")
    
    if not os.path.exists(image_path):
        print(f"错误：找不到图片 {image_path}")
    else:
        try:
            cat_count, output_path = detect_cats(image_path, model_path)
            print(f"处理完成！")
        except Exception as e:
            print(f"处理图片时出错: {str(e)}")import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import cv2
import torchvision.transforms as T
import os

def get_model(num_classes):
    # 加载预训练模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # 修改分类器以适应我们的类别数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def detect_cats(image_path, model_path, confidence_threshold=0.5):
    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = get_model(num_classes=2)  # 2类：背景和猫
    
    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 加载和转换图片
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 进行检测
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # 在图片上画框
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # 检测到的猫的数量
    cat_count = 0
    
    # 在图片上标注检测结果
    for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
        if score > confidence_threshold:
            cat_count += 1
            x1, y1, x2, y2 = box.cpu().numpy()
            # 画矩形框
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # 添加置信度文本
            text = f'Cat: {score:.2f}'
            cv2.putText(img, text, (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 在图片左上角显示检测到的猫的总数
    cv2.putText(img, f'Total Cats: {cat_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 保存结果图片
    output_path = f'result_{os.path.basename(image_path)}'
    cv2.imwrite(output_path, img)
    print(f"检测到 {cat_count} 只猫")
    print(f"结果已保存到: {output_path}")
    
    return cat_count, output_path

if __name__ == "__main__":
    # 使用最后一个epoch的模型（通常效果最好）
    model_path = 'models/cat_detector_epoch_10.pth'
    
    # 可以处理单张图片
    image_path = input("请输入要检测的图片路径: ")
    
    if not os.path.exists(image_path):
        print(f"错误：找不到图片 {image_path}")
    else:
        try:
            cat_count, output_path = detect_cats(image_path, model_path)
            print(f"处理完成！")
        except Exception as e:
            print(f"处理图片时出错: {str(e)}")