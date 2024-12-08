# Mario Action Detection

Este projeto utiliza visão computacional para detectar as ações do Mario em um vídeo, identificando se ele está:
- Parado (Standing)
- Correndo (Running)
- Pulando (Jumping)

## Requisitos

- Python 3.7+
- OpenCV (cv2)
- NumPy

## Como usar

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Coloque seu vídeo do Mario (em formato MP4) na pasta do projeto com o nome `mario_video.mp4`

3. Execute o script:
```bash
python mario_detector.py
```

4. Para sair do programa, pressione 'q'

## Como funciona

O detector utiliza processamento de imagem para:
1. Detectar o Mario usando suas cores características (vermelho do chapéu e azul do macacão)
2. Rastrear a posição do Mario entre frames
3. Analisar o movimento para determinar a ação:
   - se está Pulando
   - o quao alto foi o pulo

## Ajustes

Você pode ajustar os limiares de detecção no código:
- `jump_threshold`: Limiar para detectar pulos (em pixels)
- `movement_threshold`: Limiar para detectar corrida (em pixels)
