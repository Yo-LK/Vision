import tensorflow as tf # TensorFlow 라이브러리를 불러옵니다
from tensorflow.keras import layers, models # Keras의 레이어와 모델 클래스를 불러옵니다
import numpy as np # NumPy 라이브러리를 불러옵니다
import os # 운영체제 관련 기능을 사용하기 위한 os 모듈을 불러옵니다
from PIL import Image # 이미지 처리를 위한 PIL 라이브러리를 불러옵니다

# 1. 데이터셋 로드 및 정규화
print("CIFAR-10 데이터를 로드 중입니다...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 0~255 값을 0~1 사이 소수점으로 변환
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck'] 

# 2. 고성능 CNN 모델 설계 
# 레이어를 더 깊게 쌓고 BatchNormalization과 Dropout을 추가해 정확도를 높였습니다.
model = models.Sequential([
    # 첫 번째 블록
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    # 두 번째 블록
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # 출력 블록
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 3. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. 모델 훈련 (Epoch 15 설정)
print("\n[학습 시작] Epochs=15로 설정을 확인하세요.")
model.fit(x_train, y_train, epochs=15, batch_size=64, validation_data=(x_test, y_test))

# 5. 최종 성능 평가
print("\n--- 모델 최종 성능 평가 ---")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"최종 테스트 정확도: {test_acc*100:.2f}%")

# 6. dog.jpg 파일을 사용한 예측 결과 출력
img_name = '0402/dog.jpg'
img_path = os.path.join(os.getcwd(), img_name)

if os.path.exists(img_path):
    print(f"\n[성공] '{img_name}' 파일을 찾았습니다! 예측을 시작합니다.")
    # 이미지 전처리
    img = Image.open(img_path).resize((32, 32))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0) # (1, 32, 32, 3) 형태로 변환

    # 예측
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    
    # 클래스별 확률 계산
    confidence = np.max(predictions) * 100
    
    print("-" * 30)
    print(f"결과: 이 이미지는 [{predicted_class}]입니다!")
    print(f"신뢰도: {confidence:.2f}%")
    print("-" * 30)
else:
    print(f"\n[알림] 현재 폴더({os.getcwd()})에 '{img_name}' 파일이 없습니다.")
    print("파일이 파이썬 코드와 같은 폴더에 있는지 확인해주세요!")