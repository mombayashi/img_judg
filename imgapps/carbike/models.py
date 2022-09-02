from django.db import models

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import io, base64

graph = tf.compat.v1.get_default_graph()

class Photo(models.Model):
    image = models.ImageField(upload_to='photos')

    IMAGE_SIZE = 224 # 画像サイズ
    MODEL_FILE_PATH = './carbike/ml_models/vgg16_transfer.h5'   # モデルファイル
    classes = ["car", "motorbike"]  # クラス分類
    num_classes = len(classes)      # クラス数

    # 引数から画像ファイルを参照して読込む
    def predict(self):
        model = None
        global graph
        with graph.as_default():
            model = load_model(self.MODEL_FILE_PATH)
            
            img_data = self.image.read()    # 画像データをイメージデータで格納する
            img_bin = io.BytesIO(img_data)  # バイナリデータに変換

            image = Image.open(img_bin) # イメージを開く
            image = image.convert("RGB")    # 色の設定
            image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))   # 画像サイズの設定
            data = np.asarray(image) / 255.0     # イメージデータの変換
            X = []  # リスト作成
            X.append(data)
            X = np.array(X) # リストから配列に変換

            result = model.predict([X])[0]  # 推定結果を格納 先頭の値を取得
            predicted = result.argmax() # 大きい値と取り出す
            percentage = int(result[predicted] * 100)   # 推定確立の取得

            # print(self.classes[predicted], percentage)   # 推定結果・確率を出力
            return self.classes[predicted], percentage

    def image_src(self):
        with self.image.open() as img:
            base64_img = base64.b64encode(img.read()).decode()
            return 'data:' + img.file.content_type + ';base64,' + base64_img