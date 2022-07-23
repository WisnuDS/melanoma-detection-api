from cv2 import log
from flask import Flask, request, jsonify
import numpy as np
import cv2
from keras.models import load_model

app = Flask(__name__)
model = load_model('ModelResnet50.h5')

@app.route('/', methods=['POST'])
def test():
    if 'image' not in request.files:
        return jsonify({'code': 422, 'message': 'image diperlukan untuk proses diagnosa'})
    imgstr = request.files['image'].read()
    npimg = np.fromstring(imgstr, np.uint8)
    rawimg = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    raw = cv2.cvtColor(rawimg, cv2.COLOR_BGR2RGB)
    img = cv2.resize(raw, (224, 224))
    result = model.predict(np.expand_dims(img, axis=0))
    prosResult = ((1 - result[0][0]) * 100)
    strResult = format(prosResult, '.2f') 
    str = 'Diagnosis anda terkena melanoma adalah {}%. '.format(strResult)
    if prosResult > 50 :
        str += 'Dianjurkan untuk memeriksakan diri kedokter untuk diagnosis lebih akurat'

    return jsonify({'code': 200, 'data': { 'pros': strResult, 'message': str}})

