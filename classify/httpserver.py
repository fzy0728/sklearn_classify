#coding=utf-8
from flask import Flask, request, make_response, jsonify
import json
from train import *

def classify_backend(sentence):
    lr = model()
    lr.getfeature()
    return lr.parse(sentence.decode("utf-8"))

app = Flask(__name__)
@app.route('/classify',methods=['GET', 'POST'])
def classify():
    try:
        query = request.args['query']
        query = query.encode('utf-8')
    except:
        return make_response(jsonify({'status': 500, 'info': 'format error'}))

    try:
        result = classify_backend(query)
        return make_response(jsonify({'status': 200, 'query': query, 'result': result}))
    except:
        return make_response(jsonify({'status': 500, 'info': 'system error'}))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=28000, threaded=True)