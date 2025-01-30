from flask import Flask, request, jsonify
from flask_cors import CORS  # 添加这行
from itertools import permutations

app = Flask(__name__)
CORS(app, resources={
    r"/sort_and_permute": {
        "origins": "*",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})  # 添加详细的CORS配置

@app.route('/sort_and_permute', methods=['POST', 'OPTIONS'])  # 添加OPTIONS方法
def sort_and_permute():
    if request.method == "OPTIONS":  # 处理OPTIONS请求
        return jsonify({"status": "ok"})
        
    try:
        data = request.get_json()
        numbers = data.get('numbers', [])
        
        if len(numbers) != 10:
            return jsonify({'error': '请输入10个数字'})
        
        # 排序
        sorted_numbers = sorted(numbers)
        
        # 生成排列（限制返回前100个）
        perms = list(permutations(numbers))[:100]
        perms = [list(p) for p in perms]
        
        return jsonify({
            'sorted_numbers': sorted_numbers,
            'permutations': perms
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
