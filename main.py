# Importar as bibliotecas necessárias
import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Inicializar o detector de mãos do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Função para detectar mãos em uma imagem
def detect_hands(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    return image, results.multi_hand_landmarks

# Carregar imagens de referência
known_hands = []
known_labels = []
labels = ['A', 'B', 'C', 'D', 'E']

for label in labels:
    dir_path = os.path.join("imagens", label)
    if os.path.isdir(dir_path):
        for image in os.listdir(dir_path):
            image_path = os.path.join(dir_path, image)
            hand_image = cv2.imread(image_path)
            hand_image, _ = detect_hands(hand_image)  # Ignorar os marcos das mãos
            known_hands.append(hand_image)
            known_labels.append(label)

# Pré-processar imagens de referência
known_hands_encoded = []
for hand in known_hands:
    hand = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)
    hand = cv2.resize(hand, (160, 160))
    known_hands_encoded.append(hand)

known_hands_encoded = np.array(known_hands_encoded) / 255.0

# Codificar rótulos
label_encoder = tf.keras.utils.to_categorical(np.array([labels.index(label) for label in known_labels]), num_classes=len(labels))

# Criar modelo de reconhecimento de mãos
inputs = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(len(labels), activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(known_hands_encoded, label_encoder, epochs=50, batch_size=32)  # Aumentado para 50 epochs

# Inicializar a câmera
video = cv2.VideoCapture(0)

previous_label = None  # Variável para armazenar a letra anterior

while True:
    ret, img = video.read()
    if not ret:
        print("Erro ao capturar o vídeo.")
        break

    img, hand_landmarks = detect_hands(img)

    # Inverter a imagem da câmera
    img = cv2.flip(img, 1)

    if hand_landmarks:  # Verificar se há mãos detectadas
        # Pré-processar a imagem da câmera
        img_encoded = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_encoded = cv2.resize(img_encoded, (160, 160))
        img_encoded = np.expand_dims(img_encoded, axis=0) / 255.0

        # Fazer a classificação
        prediction = model.predict(img_encoded)
        predicted_label = labels[np.argmax(prediction)]

        # Verificar se a letra prevista é diferente da anterior
        if predicted_label != previous_label:
            previous_label = predicted_label  # Atualizar a letra anterior

        # Exibir a letra prevista
        cv2.putText(img, f"Letra: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Exibir mensagem quando nenhuma mão for detectada
        cv2.putText(img, "Nenhuma mao detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Exibir mensagem de saída
    cv2.putText(img, "Pressione 'Esc' para sair", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Resultado", img)
    if cv2.waitKey(1) == 27:  # Pressione 'Esc' para sair
        break

video.release()
cv2.destroyAllWindows()
