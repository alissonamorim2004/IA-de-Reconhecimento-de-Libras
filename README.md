Desenvolvi este projeto em Python, utilizando bibliotecas como OpenCV, NumPy, TensorFlow e MediaPipe. Ele consiste em um sistema de reconhecimento de letras do alfabeto por meio da detecção de gestos de mãos em imagens, representando uma leitura das mãos para reconhecimento de Libras com inteligência artificial.

Iniciei o projeto carregando imagens de referência das letras A, B e C e as pré-processando para uso no treinamento do modelo. Utilizei TensorFlow para criar e treinar uma rede neural convolucional capaz de reconhecer essas letras. A detecção de mãos foi realizada com a biblioteca MediaPipe, que oferece um detector de mãos eficiente.

Durante a execução do script, a câmera captura vídeo e, em cada quadro, o sistema detecta mãos, pré-processa a imagem e classifica a letra correspondente ao gesto da mão utilizando o modelo treinado. 
