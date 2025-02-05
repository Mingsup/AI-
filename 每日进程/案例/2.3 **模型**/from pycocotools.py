import requests
import os
import tarfile
from PIL import Image
import xml.etree.ElementTree as ET
import shutil

def download_pet_dataset():
    # 创建目录
    os.makedirs('dataset/images', exist_ok=True)
    os.makedirs('dataset/annotations', exist_ok=True)
    
    # 下载数据集
    image_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotation_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    
    def download_and_extract(url, filename):
        print(f"开始下载 {filename}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                downloaded += len(chunk)
                # 显示下载进度
                percent = int(downloaded * 100 / total_size)
                print(f"\r下载进度: {percent}%", end='')
        print("\n下载完成！")
        
        print(f"正在解压 {filename}...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall()
        
        os.remove(filename)
        print(f"{filename} 解压完成！")
    
    # 下载并解压数据
    download_and_extract(image_url, 'images.tar.gz')
    download_and_extract(annotation_url, 'annotations.tar.gz')
    
    # 处理图片和标注
    def process_data():
        print("开始处理数据...")
        processed_count = 0
        xml_dir = 'annotations/xmls'
        
        # 定义猫的品种列表
        cat_breeds = [
            'persian', 'maine_coon', 'bombay', 'bengal', 'siamese', 
            'british_shorthair', 'sphynx', 'ragdoll', 'birman', 
            'abyssinian', 'russian_blue', 'egyptian_mau'
        ]
        
        if not os.path.exists(xml_dir):
            print(f"错误：找不到目录 {xml_dir}")
            return
                
        total_files = len([f for f in os.listdir(xml_dir) if f.endswith('.xml')])
        print(f"找到 {total_files} 个XML文件")
        
        for xml_file in os.listdir(xml_dir):
            if not xml_file.endswith('.xml'):
                continue
            
            # 检查是否是猫的图片
            is_cat = any(breed in xml_file.lower() for breed in cat_breeds)
            if not is_cat:
                continue
                
            # 解析XML文件
            xml_path = os.path.join(xml_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 获取对应的图片
            img_name = xml_file.replace('.xml', '.jpg')
            img_path = os.path.join('images', img_name)
            
            if not os.path.exists(img_path):
                print(f"警告：找不到图片 {img_path}")
                continue
            
            try:
                # 复制图片
                img_save_path = f'dataset/images/{img_name}'
                Image.open(img_path).save(img_save_path)
                
                # 保存标注
                anno_save_path = f'dataset/annotations/{img_name.replace(".jpg", ".txt")}'
                with open(anno_save_path, 'w') as f:
                    for obj in root.findall('object'):
                        bbox = obj.find('bndbox')
                        x1 = float(bbox.find('xmin').text)
                        y1 = float(bbox.find('ymin').text)
                        x2 = float(bbox.find('xmax').text)
                        y2 = float(bbox.find('ymax').text)
                        f.write(f"{x1},{y1},{x2},{y2}\n")
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"已处理 {processed_count} 张猫咪图片")
            except Exception as e:
                print(f"处理文件 {img_name} 时出错: {e}")
        
        print(f"总共处理了 {processed_count} 张猫咪图片")
        return processed_count
    
    # 处理数据
    processed_count = process_data()
    
    # 清理临时文件
    print("清理临时文件...")
    if os.path.exists('images'):
        shutil.rmtree('images')
    if os.path.exists('annotations'):
        shutil.rmtree('annotations')
    
    # 验证结果
    image_count = len(os.listdir('dataset/images'))
    anno_count = len(os.listdir('dataset/annotations'))
    print(f"\n数据集处理完成！")
    print(f"保存位置：")
    print(f"- 图片：dataset/images/ ({image_count} 个文件)")
    print(f"- 标注：dataset/annotations/ ({anno_count} 个文件)")
    
    if image_count == 0 or anno_count == 0:
        print("\n警告：数据集目录为空。可能出现了问题。")
        print("请检查：")
        print("1. 网络连接是否正常")
        print("2. 下载的文件是否完整")
        print("3. 是否有足够的磁盘空间")

if __name__ == '__main__':
    download_pet_dataset()