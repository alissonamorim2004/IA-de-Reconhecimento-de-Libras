Este projeto foi desenvolvido em Python e utiliza as bibliotecas OpenCV, NumPy, TensorFlow e MediaPipe para criar um sistema de reconhecimento de letras do alfabeto por meio da detecção de gestos das mãos. O objetivo é facilitar a leitura das mãos e promover o reconhecimento da Língua Brasileira de Sinais (Libras) com o auxílio da inteligência artificial.

Funcionalidades Principais
-
- Reconhecimento de Gestos: O sistema é capaz de identificar gestos de mão correspondentes às letras A, B, C, D e E, fornecendo uma interface intuitiva para usuários interessados em aprender Libras.

- Pré-processamento de Imagens: As imagens de referência são carregadas e pré-processadas, incluindo redimensionamento e normalização, para garantir que sejam adequadas para o treinamento do modelo.

- Modelo de Aprendizado Profundo: Foi criada uma rede neural convolucional utilizando TensorFlow, que foi treinada com as imagens pré-processadas para reconhecer as letras com precisão.

- Detecção de Mãos em Tempo Real: Utilizando a biblioteca MediaPipe, o sistema detecta mãos em tempo real, permitindo uma interação dinâmica. O vídeo da câmera é processado quadro a quadro, onde cada gesto é analisado e classificado.

- Interface Amigável: Durante a execução, o sistema exibe na tela a letra correspondente ao gesto detectado, tornando o aprendizado mais acessível e divertido.

Tecnologias Utilizadas
-
- Python: Linguagem de programação utilizada para o desenvolvimento do projeto.
- OpenCV: Biblioteca para processamento de imagens e captura de vídeo.
- NumPy: Biblioteca para operações matemáticas e manipulação de arrays.
- TensorFlow: Framework de aprendizado de máquina usado para construir e treinar a rede neural.
- MediaPipe: Biblioteca que facilita a detecção de mãos e outros recursos em tempo real.
