from flask import Flask, request, jsonify
import mxnet as mx
from mxnet import nd
from mxnet.gluon.model_zoo import vision as models
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

net = models.get_model('cifar_resnet20_v1', classes=10, pretrained=True)

def transform_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((32, 32))
    img_array = np.array(img)
    img_nd = nd.array(img_array).astype('float32') / 255
    img_nd = img_nd.transpose((2, 0, 1)).expand_dims(axis=0)
    return img_nd

@app.route('/predict', methods=['POST'])
def predict():
    img_bytes = request.files['img'].read()
    img_nd = transform_image(img_bytes)
    pred = net(img_nd)
    ind = nd.argmax(pred, axis=1).astype('int')
    prob = nd.softmax(pred)[0][ind].asscalar()
    label = class_names[ind.asscalar()]
    prediction = 'The input picture is classified as [%s], with probability %.3f.' % (label, prob)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


