import os
import time
import json
from facecog import Config, Model
from facecog import remember, check

config = Config()
config.threshold = [0.6, 0.7, 0.8]
model = Model(config)
print('model is loaded!!!')

def key(data):
    stu_id = data['sid']
    result = remember(model, ['./input', './identities'], stu_id)
    result = json.dumps({'result': result})
    return result

def query(students):
    result, miss = check(
        model,
        ['./input/class.jpg', './identities', './output/out.jpg'],
        students
    )
    result = json.dumps({'result': result, 'miss': miss})
    return result

ops = {'key': key, 'query': query}

read_path = "./pipe.in"
write_path = "./pipe.out"

if os.path.exists(read_path):
    os.remove(read_path)
if os.path.exists(write_path):
    os.remove(write_path)

os.mkfifo(write_path)
os.mkfifo(read_path)

rf = os.open(read_path, os.O_RDONLY)
wf = os.open(write_path, os.O_SYNC | os.O_CREAT | os.O_RDWR)

while True:
    s_ = os.read(rf, 4096).strip(b'\0')
    # s = str(s, encoding='utf8')
    s = json.loads(s_, encoding='utf8')
    ret = ops[s['type']](s['data'])
    # print("received msg: %s" % s['id'])
    print(ret)
    ret = ret + '\0' * (1024 - len(ret))
    ret = bytes(ret, encoding='utf8')
    #ret = ret + b'\0' * (1024 - len(ret))
    os.write(wf, ret)
