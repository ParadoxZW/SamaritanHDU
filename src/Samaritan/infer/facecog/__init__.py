from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math, os
import mxnet as mx
import numpy as np
import cv2

import facecog.mtcnn_detector as mtcnn_detector
import facecog.face_preprocess as face_preprocess

from sklearn import preprocessing

class Config(object):
    def __init__(self, config=None):
        mp = os.path.dirname(__file__)
        if config is None:
            self.det_path = mp + '/model'
            self.ext_path = mp + '/model/model-r34-amf/model'
            self.ctx = mx.cpu()
            self.threshold = [0.6, 0.7, 0.7]
            self.num_worker = 1
            self.accurate_landmark = False
            self.chip_size = 112
            self.vthreshold = 1.24
        else:
            self.det_path = config['det_path']
            self.ext_path = config['ext_path']
            self.ctx = config['ctx']
            self.threshold = config['threshold']
            self.num_worker = config['num_worker']
            self.accurate_landmark = config['landmark']
            self.chip_size = config['chip_size']
            self.vthreshold = config['vthreshold']

class Model(object):
    def __init__(self, config):
        self.config = config
        self.threshold = config.vthreshold
        self.detector = mtcnn_detector.MtcnnDetector(
            model_folder=config.det_path,
            ctx=config.ctx,
            threshold=config.threshold,
            num_worker=config.num_worker,
            accurate_landmark=config.accurate_landmark
        )
        sym, arg_params, aux_params = mx.model.load_checkpoint(config.ext_path, 0)
        self.extractor = mx.mod.Module(symbol=sym, context=config.ctx, label_names=None)
        self.size = config.chip_size
        self.extractor.bind(data_shapes=[('data', (1, 3, self.size, self.size))])
        self.extractor.set_params(arg_params, aux_params)

    def detect(self, image):
        results = self.detector.detect_face(image)
        return results

    def reinit(self):
        config = self.config
        self.detector = mtcnn_detector.MtcnnDetector(
            model_folder=config.det_path,
            ctx=config.ctx,
            threshold=config.threshold,
            num_worker=config.num_worker,
            accurate_landmark=config.accurate_landmark
        )

    def extract(self, img, bx, pt):
        '''chip should be cv2 type'''
        size = '%d,%d' % (self.size, self.size)
        nimg = face_preprocess.preprocess(img, bx, pt, image_size=size)
        chip = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        chip = np.transpose(chip, (2,0,1))
        input_blob = np.expand_dims(chip, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.extractor.forward(db, is_train=False)
        embedding = self.extractor.get_outputs()[0].asnumpy()
        embedding = preprocessing.normalize(embedding).flatten()
        return embedding

def is_idd(threshold, emb1, emb2):
    d = np.sum((emb1 - emb2)**2)
    print("is_idd = ", d)
    if d < threshold:
        return True
    else:
        return False 

def oneface(model, img):
    '''find one face in a image of one face, then return its embedding'''
    results = model.detect(img)
    #model.reinit()
    if results is not None:
        total_boxes = results[0]
        points = results[1]
        bx = total_boxes[0,0:4]
        pt = points[0,:].reshape((2,5)).T
        emb = model.extract(img, bx, pt)
        return emb
    else:
        return None

def remember(model, path, idd, threshold=None):
    '''remember a face, i.e., save feats'''
    rpath, wpath = path
    pas = [rpath + '/' + idd + '_0.jpg', rpath + '/' + idd + '_1.jpg', rpath + '/' + idd + '_2.jpg']
    embs = []
    for p in pas:
        img = cv2.imread(p)
        if img is None:
            print(path)
            return 1
        emb = oneface(model, img)
        if emb is None:
            print(path)
            return 2
        embs.append(emb)
    #model.reinit()
    if threshold is None:
        threshold = model.threshold
    check = []
    check.append(is_idd(threshold, embs[0], embs[1]))
    check.append(is_idd(threshold, embs[0], embs[2]))
    check.append(is_idd(threshold, embs[2], embs[1]))
    if False in check:
        return 2
    pas = [wpath + '/' + idd + '_%d.npy' % i for i in [0, 1, 2]]
    for path, emb in zip(pas, embs):        
        print(path)
        np.save(path, emb)
    return 0

def check(model, path, ids, threshold=None):
    if threshold is None:
        threshold = model.threshold
    img_path, feats_path, out_path = path
    img = cv2.imread(img_path)
    if img is None:
        return (1, [])
    source = np.ones((len(ids) * 3, 512))
    i = 0
    try:
        for idt in ids:
            emb1 = np.load(feats_path + '/' + idt + '_0.npy')
            emb2 = np.load(feats_path + '/' + idt + '_1.npy')
            emb3 = np.load(feats_path + '/' + idt + '_2.npy')
            for emb in [emb1, emb2, emb3]:
                source[i] = emb
                i += 1
    except:
        return (2, [])
    # y = [[i, i, i] for i in range(len(ids))]
    # y = np.array(y).reshape(1, len(ids)*3)
    flag = [-1 for _ in range(len(ids))]
    results = model.detect(img)
    # model.reinit()
    if results is None:
        return (3, [])
    n = results[0].shape[0]
    tar = np.ones((n, 512))
    total_boxes = results[0]
    points = results[1]
    for i in range(n):
        bx = total_boxes[i,0:4]
        pt = points[i,:].reshape((2,5)).T
        tar[i] = model.extract(img, bx, pt)
    mat = -2 * np.dot(tar, source.T) + np.sum(tar**2, 1, keepdims=True) + np.sum(source**2, 1, keepdims=True).T
    np.save("./mat", mat)

    for i in range(n):
        idx = np.argmin(mat[i])
        if mat[i, idx] < threshold:
            # print(mat[i, idx])
            id_idx = idx // 3
            flag[id_idx] = i
            idx = id_idx * 3
            mat[:, idx:idx+3] = 3
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        b = total_boxes[i]
        # top, right, bottom, left = total_boxes[i,0:4].astype(np.int32)
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
    miss = []
    for i in range(len(ids)):
        if flag[i] == -1:
            miss.append(ids[i])
    cv2.imwrite(out_path, img)
    return (0, miss)
