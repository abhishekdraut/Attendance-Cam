from scipy.spatial.distance import cosine
import numpy as np
import cv2
import mtcnn
from keras.models import load_model
# from utils import get_face, plt_show, get_encode, load_pickle, l2_normalizer

import pickle

import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer

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


def face_reconizer_image():
    encoder_model = 'facenet_keras.h5'
    people_dir = 'data/people'
    encodings_path = 'data/encodings/encodings.pkl'
    test_img_path = 'data/test/test.jpeg'
    test_res_path = 'data/results/test.jpeg'

    recognition_t = 0.3
    required_size = (160, 160)

    encoding_dict = load_pickle(encodings_path)
    face_detector = mtcnn.MTCNN()
    face_encoder = load_model(encoder_model)

    img = cv2.imread(test_img_path)
    # plt_show(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.detect_faces(img_rgb)
    for res in results:
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(face_encoder, face, required_size)
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]

        name = 'unknown'
        distance = float("inf")

        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist
        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', pt_1, cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imwrite(test_res_path, img)
    plt_show(img)