import os
import pickle
import numpy as np
import mtcnn
from keras.models import load_model


import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import Normalizer

# from utils import get_face, get_encode, l2_normalizer, normalize

# temp
def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


l2_normalizer = Normalizer('l2')


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


def plt_show(cv_img):
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()


def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict


def save_pickle(path, obj):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def read_vc(vc, func_to_call, break_print=':(', show=False, win_name='', break_key='q', **kwargs):
    while vc.isOpened():
        ret, frame = vc.read()
        if not ret:
            print(break_print)
            break
        res = func_to_call(frame, **kwargs)
        if res is not None:
            frame = res

        if show:
            cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xff == ord(break_key):
            break



# tempclose



# hyper-parameters
def prepareData():
    encoder_model = 'facenet_keras.h5'
    people_dir = 'data/people'
    encodings_path = 'encodings/encodings.pkl'
    required_size = (160, 160)

    face_detector = mtcnn.MTCNN()
    face_encoder = load_model(encoder_model)

    encoding_dict = dict()

    for person_name in os.listdir(people_dir):
        person_dir = os.path.join(people_dir, person_name)
        encodes = []
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            print(img_path)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detector.detect_faces(img_rgb)
            if results:
                res = max(results, key=lambda b: b['box'][2] * b['box'][3])
                face, _, _ = get_face(img_rgb, res['box'])

                face = normalize(face)
                face = cv2.resize(face, required_size)
                encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
                encodes.append(encode)
        if encodes:
            encode = np.sum(encodes, axis=0)
            encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
            encoding_dict[person_name] = encode

    for key in encoding_dict.keys():
        print(key)

    with open(encodings_path, 'bw') as file:
        pickle.dump(encoding_dict, file)