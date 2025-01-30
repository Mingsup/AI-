from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import os

# 初始化 Flask 应用
app = Flask(__name__)

# 配置 SQLite 数据库（开发环境）
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///videos.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 允许跨域（如前端访问）
from flask_cors import CORS
CORS(app)

# 初始化数据库
db = SQLAlchemy(app)

# 创建 `uploads` 目录
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# 定义数据库模型
class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(50), default="processing")
    analysis_result = db.Column(db.JSON, nullable=True)

# 创建数据库表（仅第一次运行时需要）
with app.app_context():
    db.create_all()
