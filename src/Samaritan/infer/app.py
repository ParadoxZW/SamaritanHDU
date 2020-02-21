from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, Response
from flask import render_template
from datetime import datetime
from flask import Flask, render_template, request
from werkzeug import secure_filename
# from facecog import Config, Model
# from facecog import remember, check
import json, os

app = Flask(__name__)

write_path = "./pipe.in"
read_path = "./pipe.out"

wf = os.open(write_path, os.O_SYNC | os.O_CREAT | os.O_RDWR)
rf = os.open(read_path, os.O_RDONLY)


@app.route('/key/')
def logkey():
    # stu_id = request.json['sid']
    req = request.json
    inr_set = {}
    inr_set['type'] = 'key'
    inr_set['data'] = req
    inr = json.dumps(inr_set)
    inr = bytes(inr + '\0' * (4096 - len(inr)), encoding='utf8')
    os.write(wf, inr)
    result = os.read(rf, 1024)
    result = str(result, encoding='utf8')
    return result


@app.route('/query/', methods=['GET', 'POST'])
def query():
    # students = request.json
    # students = json.loads(data.decode("utf-8"))
    req = request.json
    inr_set = {}
    inr_set['type'] = 'query'
    inr_set['data'] = req
    inr = json.dumps(inr_set)
    inr = bytes(inr + '\0' * (4096 - len(inr)), encoding='utf8')
    os.write(wf, inr)
    result = os.read(rf, 1024)
    result = str(result, encoding='utf8')

    # result, miss = check(
    #     model,
    #     ['./input/class.jpg', './identities', './output/out.jpg'],
    #     students
    # )
    # miss = json.dumps({'result': result, 'miss': miss})
    return result


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, debug=True)
