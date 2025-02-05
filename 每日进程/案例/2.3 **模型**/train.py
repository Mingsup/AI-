import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.utils.data
from PIL import Image
import os
import numpy as np
import time
from datetime import datetime

# 定义数据集类
class CatDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        # 加载图片
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        # 加载标注
        anno_path = os.path.join(self.root, "annotations", self.annotations[idx])
        boxes = []
        with open(anno_path) as f:
            for line in f:
                x1, y1, x2, y2 = map(float, line.strip().split(','))
                boxes.append([x1, y1, x2, y2])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # 所有标签都是1（猫）
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model(num_classes):
    # 加载预训练模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # 修改分类器以适应我们的类别数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_model():
    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 数据转换
    from torchvision.transforms import transforms as T
    transform = T.Compose([
        T.ToTensor(),
    ])
    
    # 加载数据集
    print("正在加载数据集...")
    dataset = CatDataset(root="dataset", transforms=transform)
    
    # 划分训练集和验证集
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(len(dataset) * 0.8)  # 80%用于训练
    dataset_train = torch.utils.data.Subset(dataset, indices[:train_size])
    dataset_val = torch.utils.data.Subset(dataset, indices[train_size:])
    
    print(f"数据集总大小: {len(dataset)}")
    print(f"训练集大小: {len(dataset_train)}")
    print(f"验证集大小: {len(dataset_val)}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=2, 
        shuffle=True, 
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x))
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # 创建模型
    print("初始化模型...")
    num_classes = 2  # 背景 + 猫
    model = get_model(num_classes)
    model.to(device)
    
    # 设置优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # 训练循环
    num_epochs = 10
    print("\n开始训练...")
    print("=" * 50)
    
    # 创建保存模型的目录
    os.makedirs('models', exist_ok=True)
    
    # 记录训练开始时间
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # 创建进度条
        total_batches = len(data_loader_train)
        
        for i, (images, targets) in enumerate(data_loader_train):
            batch_start_time = time.time()
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            
            # 计算进度和预计剩余时间
            progress = (i + 1) / total_batches
            batch_time = time.time() - batch_start_time
            remaining_batches = total_batches - (i + 1)
            eta = remaining_batches * batch_time
            
            # 打印进度信息
            if i % 5 == 0:  # 每5个batch更新一次信息
                print(f"\r进度: [{i+1}/{total_batches}] "
                      f"({progress*100:.1f}%) "
                      f"Loss: {losses.item():.4f} "
                      f"ETA: {eta/60:.1f}分钟", end="")
        
        # 更新学习率
        lr_scheduler.step()
        
        # 计算平均损失
        avg_loss = total_loss / len(data_loader_train)
        epoch_time = time.time() - epoch_start_time
        
        print(f"\n\nEpoch {epoch+1} 统计:")
        print(f"平均损失: {avg_loss:.4f}")
        print(f"耗时: {epoch_time/60:.1f}分钟")
        
        # 保存模型
        save_path = f'models/cat_detector_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)
        print(f"模型已保存到: {save_path}")
    
    total_time = time.time() - start_time
    print("\n训练完成！")
    print(f"总训练时间: {total_time/3600:.1f}小时")
    print(f"模型保存在 'models' 目录下")
    
    return model

if __name__ == "__main__":
    # Windows下多进程的保护
    import multiprocessing
    multiprocessing.freeze_support()
    
    # 记录开始时间
    start_datetime = datetime.now()
    print(f"开始训练时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        model = train_model()
        print("\n训练成功完成！")
    except Exception as e:
        print(f"\n训练过程中出现错误: {str(e)}")
    finally:
        end_datetime = datetime.now()
        training_duration = end_datetime - start_datetime
        print(f"结束训练时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总计用时: {training_duration}")