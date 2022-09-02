import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.applications import VGG16

# パラメータの初期化
classes = ["car", "motorbike"]  # クラス分類
num_classes = len(classes)      # クラス数
image_size = 224                # 画像サイズ（ピクセル）

# データの読込み
X_train, X_test, y_train, y_test = np.load("./imagefiles_224.npy", allow_pickle=True)  # 指定ファイルからの読み込み
y_train = np_utils.to_categorical(y_train, num_classes)         # one hot表現に変換
y_test = np_utils.to_categorical(y_test, num_classes)
X_train = X_train.astype("float") / 255.0   # 正規化の処理をするために整数からfloat型に変更して255.0で割る
X_test = X_test.astype("float") / 225.0

# モデルの定義
model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))  # ImageNetを使って学習した重み、　全結合層の解除、　入力画像の形状を指定
# print('Model loaded')   # モデルの読込み確認
# model.summary() # モデル構造の可視化

# 全結合層の構築
top_model = Sequential()    # Sequentialモデルの宣言
top_model.add(Flatten(input_shape=model.output_shape[1:]))  # Flattenで入力されたデータを直列で並べる、　modelのアウトプットの形状で１番目以降をshapeを捕捉
top_model.add(Dense(256, activation='relu'))    # 全結合　出力:256
top_model.add(Dropout(0.5))  # ドロップアウト層（過学習予防）
top_model.add(Dense(num_classes, activation='softmax'))     # 全結合　出力:num_classes、  softmax関数で総和が１になるようにする

model = Model(inputs=model.input, outputs=top_model(model.output))  #top_modelとVGG16を結合

# model.summary()

# モデル重みの固定
for layer in model.layers[:15]:
    layer.trainable = False     # パラメータ更新の設定

#opt = SGD(lr=0.01)  #学習率の設定
opt = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])   #訓練プロセス

model.fit(X_train, y_train, batch_size=32, epochs=5)   #訓練開始

score = model.evaluate(X_test, y_test, batch_size=32)   #評価

# モデルの保存
model.save("./vgg16_transfer.h5")