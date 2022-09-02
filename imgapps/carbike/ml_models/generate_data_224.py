# モジュールのインポート
from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

# パラメータの初期化
classes = ["car", "motorbike"]  # クラス分類
num_classes = len(classes)      # クラス数
image_size = 224                # 画像サイズ（ピクセル）

# 画像の読み込み、　NumPy配列への変換
X = []  # 画像ファイルのリスト
Y = []  # 正解ラベルのリスト car or motobike(0 or 1)

for index, classlabel in enumerate(classes):    # enumerate = インデックス番号の付与
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")    # jpgファイルを代入
    for i , file in enumerate(files):
        image = Image.open(file)        # ファイルを開く
        image = image.convert("RGB")    # 色の設定
        image = image.resize((image_size,image_size))   # 画像サイズの設定
        data = np.asarray(image)     # イメージデータの変換
        X.append(data)  # リストに追加
        Y.append(index)

X = np.array(X) # リストから配列に変換
Y = np.array(Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)   # 分割してトレーニング用とテスト用に分ける
xy = (X_train, X_test, y_train, y_test) # １つにまとめる
np.save("./imagefiles_224.npy", xy) # 指定フォルダに出力