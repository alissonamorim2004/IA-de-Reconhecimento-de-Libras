# IA-de-Reconhecimento-de-Libras
Este projeto utiliza um modelo de rede neural convolucional (CNN) para reconhecer sinais de mão em tempo real, com base em imagens capturadas pela câmera. O objetivo é identificar letras específicas da língua de sinais através da análise das imagens das mãos.

Funcionalidades:

- Reconhecimento em Tempo Real: O modelo é capaz de identificar sinais de mão em tempo real usando a câmera do computador.
- Suporte a Várias Letras: O sistema foi treinado para reconhecer as letras A, B, C, D e E.
- Visualização de Resultados: As letras reconhecidas são exibidas na tela junto com a imagem capturada.


Tecnologias Utilizadas:
- TensorFlow: Para a criação e treinamento do modelo de rede neural convolucional.
- OpenCV: Para captura e processamento de imagens em tempo real.
- MediaPipe: Para detecção e rastreamento das mãos nas imagens.
- Keras: Para a construção e treinamento do modelo de rede neural.

  
Funcionalidade
- Captura de Imagens: Utiliza a câmera para capturar imagens das mãos.
- Detecção de Mãos: MediaPipe é usado para detectar e desenhar as mãos nas imagens.
- Pré-processamento: As imagens das mãos são pré-processadas para adequação ao modelo.
- Predição: O modelo prevê a letra correspondente ao sinal da mão detectada.
- Exibição dos Resultados: A letra reconhecida é exibida na tela.
  
Estrutura do Projeto
- Código de Treinamento: Scripts para treinamento do modelo com imagens de referência.
- Código de Inferência: Scripts para realizar a detecção de sinais em tempo real usando a câmera.
- Imagens de Referência: Imagens usadas para treinar o modelo.

