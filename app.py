# -*- coding: utf-8 -*-
"""
@author: lywen
"""
import os
import json
import time
import numpy as np
import uuid
from PIL import Image
from model import  text_predict,crnn_handle
from flask import Flask, request, render_template

app = Flask(__name__)

filelock='file.lock'
if os.path.exists(filelock):
   os.remove(filelock)

from config import *
from apphelper.image import union_rbox,adjust_box_to_origin,base64_to_PIL
from application import trainTicket,idcard

   
# from main import TextOcrModel

billList = ['通用OCR','火车票','身份证']


@app.route('/ocr')
def ocr_get():
    post = {}
    post['postName'] = 'ocr'##请求地址
    post['height'] = 1000
    post['H'] = 1000
    post['width'] = 600
    post['W'] = 600
    post['billList'] = billList
    return render_template('ocr.html', post=post)


def run_ocr(img, billModel, textLine):
    t = time.time()
    H, W = img.shape[:2]
    res = ''
    uidJob = uuid.uuid1().__str__()

    while time.time()-t <= TIMEOUT:
        if os.path.exists(filelock):
            continue
        else:
            with open(filelock, 'w') as f:
                f.write(uidJob)

            if textLine:
                ##单行识别
                partImg = Image.fromarray(img)
                text = crnn_handle.predict(partImg)
                res = [{'text': text, 'name': '0',
                        'box': [0, 0, W, 0, W, H, 0, H]}]
                os.remove(filelock)
                break

            else:
                # detectAngle = textAngle
                result = text_predict(img)

                if billModel == '' or billModel == '通用OCR':
                    # result = union_rbox(result,0.2)
                    res = [{'text': x['text'],
                            'name':str(i),
                            'box':{'cx': x['cx'],
                                   'cy':x['cy'],
                                   'w':x['w'],
                                   'h':x['h'],
                                   'angle':x['degree']

                                   }
                            } for i, x in enumerate(result)]
                    # res = adjust_box_to_origin(img,angle, res)##修正box

                elif billModel == '火车票':
                    res = trainTicket.trainTicket(result)
                    res = res.res
                    res = [{'text': res[key], 'name':key, 'box':{}}
                           for key in res]

                elif billModel == '身份证':

                    res = idcard.idcard(result)
                    res = res.res
                    res = [{'text': res[key], 'name':key, 'box':{}}
                           for key in res]

                os.remove(filelock)
                break

    return res


@app.route('/ocr', methods=['POST'])
def ocr_post():
    data = request.json
    
    billModel = data.get('billModel','')
    # textAngle = data.get('textAngle',False)##文字检测
    textLine = data.get('textLine',False)##只进行单行识别

    imgString = data['imgString'].encode().split(b';base64,')[-1]
    img = base64_to_PIL(imgString)
    if img is not None:
        img = np.array(img)
        
    res = run_ocr(img, billModel, textLine)
        
    return json.dumps({'res': res},ensure_ascii=False)
        

if __name__ == "__main__":
    import sys
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, help='port', default=5000)
    parser.add_argument(
        '--file', type=str, help='ocr file instead of web server', default='')
    parser.add_argument(
        '--billmode', type=str, help='bill mode', choices=billList, default=billList[0])
    parser.add_argument(
        '--singleline', type=bool, default=False)
    args = parser.parse_args()

    if args.file:
        res = run_ocr(np.array(Image.open(args.file)), args.billmode, args.singleline)
        print(res)
    else:
        os.environ['FLASK_ENVIRONMENT'] = 'development'
        sys.argv = [sys.argv[0]]
        app.run(debug=True, port=args.port)
