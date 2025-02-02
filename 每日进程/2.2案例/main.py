from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from typing import Dict
import tempfile
import os
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建保存截图的文件夹
HITS_FOLDER = "hits_captures"
if not os.path.exists(HITS_FOLDER):
    os.makedirs(HITS_FOLDER)

# 挂载静态文件夹
app.mount("/hits", StaticFiles(directory=HITS_FOLDER), name="hits")

async def detect_hits(video_path: str, video_name: str) -> Dict[str, any]:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="无法打开视频文件")

        hits_count = 0
        prev_frame = None
        frame_buffer = []
        frame_number = 0
        hit_frames = []
        last_hit_frame = 0  # 记录上一次击球的帧号
        MIN_FRAMES_BETWEEN_HITS = 15  # 两次击球之间的最小帧数间隔
        
        video_folder = os.path.join(HITS_FOLDER, video_name.split('.')[0])
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_number += 1
            original_frame = frame.copy()
            
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if prev_frame is not None:
                frame_diff = cv2.absdiff(gray, prev_frame)
                thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                significant_motion = False
                for contour in contours:
                    if cv2.contourArea(contour) > 1000:  # 可以调整这个阈值
                        significant_motion = True
                        break
                
                frame_buffer.append(significant_motion)
                if len(frame_buffer) > 10:
                    frame_buffer.pop(0)
                    
                    # 改进的击球检测逻辑
                    if (len(frame_buffer) >= 5 and 
                        sum(frame_buffer[-5:]) >= 3 and 
                        frame_number - last_hit_frame > MIN_FRAMES_BETWEEN_HITS):
                        
                        hits_count += 1
                        last_hit_frame = frame_number
                        
                        # 保存击球瞬间的截图
                        timestamp = datetime.now().strftime("%H%M%S")
                        image_name = f"hit_{hits_count}_{timestamp}.jpg"
                        image_path = os.path.join(video_folder, image_name)
                        cv2.imwrite(image_path, original_frame)
                        
                        hit_frames.append({
                            "hit_number": hits_count,
                            "frame_number": frame_number,
                            "image_path": f"/hits/{video_name.split('.')[0]}/{image_name}"
                        })
                        
                        frame_buffer = []  # 重置缓冲区
                    
            prev_frame = gray
            
        cap.release()
        
        return {
            "hits_count": hits_count,
            "hit_frames": hit_frames
        }

    except Exception as e:
        logger.error(f"视频处理错误: {str(e)}")
        raise HTTPException(status_code=500, detail="视频处理失败")

@app.post("/analyze-badminton")
async def analyze_badminton_video(video: UploadFile = File(...)) -> Dict:
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="不支持的视频格式")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as temp_file:
        try:
            content = await video.read()
            if not content:
                raise HTTPException(status_code=400, detail="空文件")
                
            temp_file.write(content)
            temp_path = temp_file.name
            
            logger.info(f"开始分析视频: {video.filename}")
            result = await detect_hits(temp_path, video.filename)
            logger.info(f"视频分析完成，检测到 {result['hits_count']} 次击球")
            
            return result
            
        except Exception as e:
            logger.error(f"处理视频时出错: {str(e)}")
            raise HTTPException(status_code=500, detail="视频处理失败")
        finally:
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.error(f"删除临时文件失败: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "羽毛球视频分析API服务正在运行"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
