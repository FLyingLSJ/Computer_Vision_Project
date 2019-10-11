# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
import time
import cv2
import json

http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
key = "2r3Ydxj5X8roXt2RwftJ0jYkYH0NMsVL"  #"填上你的KEY"
secret = "xabfyC5_s_7c2y86t1_qiVD6FxbVtF3i"#"填上你的SECRET"
filepath = r"./test_img/erjie.jpeg"

boundary = '----------%s' % hex(int(time.time() * 1000))
data = []
data.append('--%s' % boundary)
data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
data.append(key)
data.append('--%s' % boundary)
data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
data.append(secret)
data.append('--%s' % boundary)
fr = open(filepath, 'rb')
data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
data.append('Content-Type: %s\r\n' % 'application/octet-stream')
data.append(fr.read())
fr.close()
data.append('--%s' % boundary)
data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_landmark')
data.append('1')
data.append('--%s' % boundary)
data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
data.append(
    "gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus")
data.append('--%s--\r\n' % boundary)

for i, d in enumerate(data):
    if isinstance(d, str):
        data[i] = d.encode('utf-8')

http_body = b'\r\n'.join(data)

# build http request
req = urllib.request.Request(url=http_url, data=http_body)

# header
req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)

try:
    # post data to server
    resp = urllib.request.urlopen(req, timeout=5)
    # get response
    qrcont = resp.read()
    # if you want to load as json, you should decode first,
    # for example: json.loads(qrount.decode('utf-8'))
    #print(qrcont.decode('utf-8'))
    result = json.loads(qrcont.decode('utf-8'))
    smile_value = result['faces'][0]['attributes']['smile']['value']
    print(type(result))
    if float(smile_value) > 60:
      smile_flag = "smile"
    else: 
      smile_flag = "none_smile"
      
    img = cv2.imread(filepath)
    cv2.putText(img,
                    smile_flag,
                    (200, 200),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(0, 0, 255),
                    thickness=2)
    cv2.imwrite("erjie.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except urllib.error.HTTPError as e:
    print(e.read().decode('utf-8'))