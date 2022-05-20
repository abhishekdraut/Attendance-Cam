
import base64
from flask import Flask,jsonify, request
from flask_cors import CORS, cross_origin
import io
from PIL import Image
import prepare_data
import face_recognizer_image
import utils
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'application/json'
CORS(app, support_credentials=True)
imagebase64 = ''


@app.route('/img', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def base64I():
    if request.method == 'POST':
        base = request.get_json()

        imagebase64 = base["base64Img"]

        image = base64.b64decode(str(imagebase64))
        fileName = 'test.jpeg'
        imagePath=('data/test/' + fileName)
        # imagePath = ('./imgDir/' + "test.jpeg")
        img = Image.open(io.BytesIO(image))
        img.save(imagePath, 'jpeg')
        # prepare_data.prepareData()
        face_recognizer_image.face_reconizer_image()
        # imgdata = base64.b64decode(imagebase64)
        # filename = 'some_image.jpg'
        # detected=extract_from_images("./imgDir/")
        # print(detected)

        # with open(filename, 'wb') as f:
        #     f.write(imgdata)
        return base



    elif request.method == 'GET':
        return 'reachedImageUrl'


if __name__ == '__main__':
    app.run(host='0.0.0.0',
            debug=True,
            port=8080)
