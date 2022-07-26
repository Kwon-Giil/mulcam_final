import tensorflow as tf
from tensorflow import keras
import numpy as np
from flask import Flask, render_template, url_for, request
from StyleGAN import style_transfer
import os, time
import urllib3, json, base64
import cv2


def detect_api(fname):
    openApiURL = "http://aiopen.etri.re.kr:8000/ObjectDetect"
    accessKey = "c2ef0cf7-9c10-4e92-8f80-ca8823807873"
    imageFilePath = "./flask_project/static/img/" + fname
    type = "jpg"
    
    file = open(imageFilePath, "rb")
    imageContents = base64.b64encode(file.read()).decode("utf8")
    file.close()
    
    requestJson = {
        "access_key": accessKey,
        "argument": {
            "type": type,
            "file": imageContents
        }
    }
    
    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        openApiURL,
        headers={"Content-Type": "application/json; charset=UTF-8"},
        body=json.dumps(requestJson)
    )

    image = cv2.imread(imageFilePath, cv2.COLOR_BGR2RGB) 
    if str(response.status) == '200':
        json_data = json.loads(response.data)
        print(json_data)
        for i in range(len(json_data['return_object']['data'])):
            cl = json_data['return_object']['data'][i]['class']
            confi = json_data['return_object']['data'][i]['confidence']
            s_point = (int(json_data['return_object']['data'][i]['x']), int(json_data['return_object']['data'][i]['y']))
            e_point = (int(json_data['return_object']['data'][i]['x'])+int(json_data['return_object']['data'][i]['width']), int(json_data['return_object']['data'][i]['y'])+int(json_data['return_object']['data'][i]['height']))
            if cl == 'person':
                color  = (0,0,255)
            else:
                color = (255,0,0)
            thickness = 1
            cv2.putText(image, cl, (int(json_data['return_object']['data'][i]['x']), int(json_data['return_object']['data'][i]['y'])-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,color)
            cv2.putText(image, confi, s_point,cv2.FONT_HERSHEY_SIMPLEX,0.3,color)
            image_with_rectangle = cv2.rectangle(
                img = image,
                pt1 = s_point,
                pt2 = e_point, 
                color = color, 
                thickness = thickness
            )

        detect_fname = str(fname).split('.')[0]+'_detect.jpg'
        cv2.imwrite('./flask_project/static/img/'+ detect_fname, image_with_rectangle)

    return detect_fname


app = Flask(__name__)


# 웹 페이지 접속시 메인 화면
@app.route('/', methods=['GET'])
def main_loading():
    return render_template("home_img.html")


# 파일 업로드 및 해당 파일 화면에 출력
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # 파일 정보 받아오기
    file = request.files['file']
    fname = file.filename

    # 파일 저장하기
    file.save(os.path.join('static/img/', fname))
    time.sleep(5)
    image_path = 'static/img/'+fname
    style_transfer(image_path)

    # 저장한 파일 화면에 출력하기
    f_name = os.path.join('/img/', fname)
    return render_template("home_img.html", f_name=f_name)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=106, debug=True)
