import tensorflow as tf # TensorFlow 라이브러리를 불러옵니다
from tensorflow.keras import layers, models # Keras의 레이어와 모델 클래스를 불러옵니다

# 1. MNIST 데이터셋 로드 및 훈련/테스트 세트로 분할
print("데이터를 로드 중입니다...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

# 2. 데이터 전처리 (0~255 사이의 픽셀 값을 0~1 사이로 정규화)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Sequential 모델을 사용하여 신경망 구축
# Dense 레이어 활용
model = models.Sequential([
    # 28x28 2차원 배열을 1차원(784)으로 펼침
    layers.Flatten(input_shape=(28, 28)), 
    # 은닉층: 128개 노드, ReLU 활성화 함수
    layers.Dense(128, activation='relu'),
    # 과적합 방지를 위한 드롭아웃(선택 사항이나 권장됨)
    layers.Dropout(0.2),
    # 출력층: 10개 클래스(숫자 0~9), Softmax로 확률 출력
    layers.Dense(10, activation='softmax')
])

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련
print("\n모델 훈련을 시작합니다...")
model.fit(x_train, y_train, epochs=5)

# 6. 정확도 평가
print("\n모델 성능 평가 결과:")
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\n최종 테스트 정확도: {accuracy*100:.2f}%")