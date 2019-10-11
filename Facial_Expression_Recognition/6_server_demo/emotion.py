from flask import Flask, request
from flask import render_template
import time
from expression_demo import expression_predict  # 表情预测项目
import os

app = Flask(__name__)  # 创建一个 Flask 实例，使用单一模块，应该使用 __name__



system_path = os.path.dirname(__file__)
@app.route('/')
def hello(imgPath=None):
    return render_template('index.html', imgPath=system_path+"/static/image/logo.jpg")


@app.route('/upload', methods=['POST'])
def upload(imgPath=None, result="None"):
    """
    :param imgPath: 上传的图片会保存在服务器里
    :param result: 预测的结果
    :return:
    """
    file = request.files['file']
    fileName = file.filename
    filePath = system_path+"/static/image/"+fileName  #"/tmp/flask_img_predict/static/image/" + fileName  # 图片路径
    print(filePath)
#     print(filePath)
    if file:
        file.save(filePath)
        result = expression_predict(filePath)
        if result is None:
            result = "could not found your beauty face"
        return render_template('index.html', imgPath=system_path+"/static/image/"+fileName, result=result)
    else:
        return render_template('index.html', imgPath=system_path+"/static/image/logo.jpg")

    
    
if __name__ == '__main__':
    # 使你的服务器公开可用，WIN+R-> cmd -> ipconfig/all 可以看见主机名：在任意电脑输入 主机名:5000 即可看到效果
    app.run(host="0.0.0.0") # 
