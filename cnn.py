import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.utils import np_utils

# パラメータの初期化
classes = ["car", "motorbike"]  # クラス分類
num_classes = len(classes)      # クラス数
image_size = 150                # 画像サイズ（ピクセル）

X_train, X_test, y_train, y_test = np.load("./imagefiles.npy", allow_pickle=True)  # 指定ファイルからの読み込み
y_train = np_utils.to_categorical(y_train, num_classes)         # one hot表現に変換
y_test = np_utils.to_categorical(y_test, num_classes)

# モデルの定義
model = Sequential()    # Sequentialモデルの宣言
model.add(Conv2D(32,(3,3), activation='relu', input_shape=(image_size, image_size, 3)))  # 畳み込み層、　活性化関数（ランプ関数）
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))    # プーリング層(最大値)
model.add(Dropout(0.25))    # ドロップアウト層（過学習予防）

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())    # 平坦化
model.add(Dense(256, activation='relu'))    # 全結合　出力:256
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))     # 全結合　出力:num_classes、  softmax関数で総和が１になるようにする

# opt = SGD(lr=0.01)  # 学習率の設定
opt = Adam()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])   # 訓練プロセス

model.fit(X_train, y_train, batch_size=32, epochs=5)   # 訓練開始

score = model.evaluate(X_test, y_test, batch_size=32)   # 評価

