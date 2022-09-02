import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from PIL import Image
import sys

# パラメータの初期化
classes = ["car", "motorbike"]  # クラス分類
num_classes = len(classes)      # クラス数
image_size = 224                # 画像サイズ（ピクセル）

# 引数から画像ファイルを参照して読込む
image = Image.open(sys.argv[1]) # イメージを開く
image = image.convert("RGB")    # 色の設定
image = image.resize((image_size,image_size))   # 画像サイズの設定
data = np.asarray(image) / 255.0     # イメージデータの変換
X = []  # リスト作成
X.append(data)
X = np.array(X) # リストから配列に変換

# モデルのロード
model = load_model('./vgg16_transfer.h5')   # ファイル指定

result = model.predict([X])[0]  # 推定結果を格納 先頭の値を取得
predicted = result.argmax() # 大きい値と取り出す
percentage = int(result[predicted] * 100)   # 推定確立の取得

print(classes[predicted], percentage)   # 推定結果・確率を出力