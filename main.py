from flask import Flask, render_template, request, session, url_for, redirect
import os.path
from PIL import Image
import cv2
import io
import base64
import numpy as np
import tensorflow
import keras
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras import optimizers

# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = './uploads'
# アップロードされる拡張子の制限
#ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

app = Flask(__name__, static_folder='.', static_url_path='')

#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send',methods = ['POST'])
def posttest():
    img_file = request.files['img_file']
    fileName = img_file.filename
    root, ext = os.path.splitext(fileName)
    ext = ext.lower()
    gazouketori = set([".jpg", ".jpeg", ".jpe", ".jp2", ".png", ".webp", ".bmp", ".pbm", ".pgm", ".ppm",
                       ".pxm", ".pnm",  ".sr",  ".ras", ".tiff", ".tif", ".exr", ".hdr", ".pic", ".dib"])
    if ext not in gazouketori:
        return render_template('index.html',massege = "対応してない拡張子です",color = "red")
    print("success")
    buf = io.BytesIO()
    image = Image.open(img_file)
    image.save(buf, 'png')
    
    #機械学習で分類する
    #モデル構築
    #転移学習 inception v3
    base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224,224,3))
    x = base_model.output
    # チャネルごとの平均をとってフラットにする
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs = base_model.input, outputs=predictions)

    for layer in base_model.layers[:249]:
        layer.trainable = False
  
        if layer.name.startswith('batch_normalization'):
            layer.trainable = True

    for layer in base_model.layers[249:]:
        layer.trainable = True

    # モデルにパラメータを読み込む。前回の学習状態を引き継げる。
    model.load_weights('param.hdf5')
    
    #モデルに入れる用の写真データ整形（judge）

    judge = np.array(image, dtype=np.uint8)
    judge = cv2.cvtColor(judge, cv2.COLOR_RGB2BGR)

    h, w, c = judge.shape

    if h <= w:
        judge = judge[:, int(w/2 - h/2):int(w/2 + h/2), :]
    else:
        judge = judge[int(h/2 - w/2):int(h/2 + w/2), :, :]
                     
    judge = cv2.resize(judge, (224, 224), cv2.INTER_LANCZOS4)  # 画像サイズを224px × 224pxにする
            
    #ヒストグラム平坦化
    judge = cv2.cvtColor(judge, cv2.COLOR_BGR2YUV)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    judge[:,:,0] = clahe.apply(judge[:,:,0])

    judge = cv2.cvtColor(judge, cv2.COLOR_YUV2RGB)

    judge = judge / 255.0

    #judge完成 judgeのラベルを予測する
    pre_label = model.predict(judge.reshape(1,224,224,3)).argmax()

    print("ok")

    qr_b64str = base64.b64encode(buf.getvalue()).decode("utf-8")
    qr_b64data = "data:image/png;base64,{}".format(qr_b64str)
    return render_template('kekka.html',img = qr_b64data, label = pre_label)

@app.route('/kekka')
def result():
    return render_template('kekka.html')

#@app.route('/uploads/<filename>')
# ファイルを表示する
#def uploaded_file(filename):
#return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

app.run()
