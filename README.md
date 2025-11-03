# LLM Multi Robots

Simulação de robôs móveis com capacidades de compreensão visual através de modelos de linguagem visual integrados ao Webots.

## Visão Geral

Este repositório implementa controladores para robôs autônomos que utilizam visão computacional com aprendizado de máquina para compreender e navegar em seu ambiente. O projeto integra:

- Ambiente de simulação Webots com física realista para múltiplos agentes robóticos
- Modelo BLIP (Vision Language Model) para análise de imagens e compreensão visual
- Desenvolvimento em Python 3.9+ com controllers parametrizáveis
- Prototipagem customizável de plataformas robóticas baseada em BB-8

## Objetivo

O trabalho visa desenvolver um sistema robótico capaz de:

- Capturar imagens em tempo real a partir de câmeras embarcadas
- Processar informações visuais e gerar descrições semânticas de cenas
- Responder perguntas sobre características e objetos presentes no ambiente
- Executar navegação autônoma com detecção de obstáculos baseada em visão
- Escalar para múltiplos robôs com coordenação distribuída

## Estrutura do Projeto

```
llm_multi_robots/
├── controllers/
│   └── droid_llm/
│       └── droid_llm.py           # Controller integrado com modelo BLIP
├── protos/
│   ├── BallDroid.proto            # Extensão do BB-8 com câmera embarcada
│   ├── BB-8.proto                 # Protótipo base do robô Sphero BB-8
│   └── icons/                     # Recursos gráficos para interface Webots
├── worlds/
│   └── LLM_ORCHESTRATOR.wbt       # Cenário de simulação
└── README.md                       # Esta documentação
```

## Componentes Principais

### droid_llm.py - Controller Inteligente

O controller central que executa em cada robô simulado. Implementa a classe `BLIPRobotController` com as seguintes funcionalidades:

- Captura de imagens da câmera embarcada (`head_camera`)
- Carregamento do modelo BLIP pré-treinado (aproximadamente 990 MB)
- Detecção automática e utilização de aceleração por GPU (CUDA ou Metal Performance Shaders)
- Geração de descrições de cenas (image captioning)
- Resposta a perguntas sobre o conteúdo visual (visual question answering)
- Lógica de controle reativa baseada em compreensão visual

O controller executa consultas ao modelo com intervalo configurável (padrão: 100 ms) e processa as respostas para tomar decisões de navegação.

### BallDroid.proto - Robô Customizado

Extensão do protótipo BB-8 que incorpora uma câmera embarcada na estrutura do robô:

- Campo de visão: 1.2 radianos
- Resolução de imagem: 640x480 pixels
- Posicionamento: câmera integrada na cabeça do robô
- Sensores adicionais: acelerômetro e giroscópio para estimativa de estado

### BB-8.proto - Base do Robô

Protótipo oficial da plataforma robótica Sphero BB-8 com os seguintes atributos:

- Corpo esférico com raio configurável (padrão: 0.25 m)
- Cabeça articulada com dois graus de liberdade (pitch e yaw)
- Motores rotativos para controle de movimento
- Dinâmica realista com parâmetros de massa e fricção

### LLM_ORCHESTRATOR.wbt - Ambiente de Simulação

Define o cenário de simulação Webots contendo:

- Arena retangular de 30x30 metros
- Piso com textura Parquetry para realismo visual
- Posicionamento inicial do robô BallDroid
- Configurações de iluminação e câmeras de visualização

## Tecnologias Utilizadas

| Componente | Versão | Função |
|-----------|--------|--------|
| Webots | R2025a | Simulador robótico com motor de física |
| Python | 3.9+ | Linguagem de implementação |
| PyTorch | Recente | Framework de deep learning |
| Transformers | Recente | Biblioteca para acesso a modelos pré-treinados |
| BLIP (Salesforce) | Base/Large | Modelo de visão e linguagem |
| PIL/Pillow | Recente | Processamento de imagens |
| NumPy | Recente | Operações numéricas |

## Dependências

Instale os pacotes Python necessários através do gerenciador de pacotes:

```bash
pip install torch torchvision
pip install transformers
pip install pillow
pip install numpy
```

## Como Usar

### Requisitos do Sistema

- Webots R2025a instalado
- Python 3.9 ou superior
- Dependências Python conforme listadas acima

### Executando a Simulação

1. Abra o Webots
2. Carregue o arquivo de mundo: `worlds/LLM_ORCHESTRATOR.wbt`
3. Inicie a simulação pressionando o botão de play
4. O controller será carregado automaticamente
5. Acompanhe a saída no console do Webots

Saída esperada do console:

```
Loading BLIP model...
Using device: cuda
Robot controller started. Press Ctrl+C to stop.

Generating scene description...
Caption (2.34s): a robot rolling on a wooden floor
Obstacle check: no
Action: Path appears clear
```

## Funcionalidades do Modelo de Linguagem Visual

### Geração de Descrições (Image Captioning)

Produz uma descrição textual do conteúdo visual capturado:

```python
caption = controller.caption_image(image)
# Resultado: "A wooden floor with shadows"
```

### Resposta a Perguntas Visuais (VQA)

Responde questões específicas sobre o cenário observado:

```python
answer = controller.answer_question(image, "Is there an obstacle in front?")
# Resultado: "no" ou "yes"
```

### Integração com Lógica de Navegação

As respostas do modelo informam decisões de controle:

- Detecção de obstáculos: ativa comportamento de desvio
- Ausência de obstáculos: mantém trajetória

## Fluxo de Execução

O ciclo de controle segue a seguinte sequência:

1. Simulação Webots é iniciada
2. Controller é carregado
3. Modelo BLIP é inicializado em GPU/CPU
4. Em cada iteração de tempo:
   - Captura imagem da câmera
   - Gera descrição semântica
   - Responde pergunta de navegação
   - Processa lógica de decisão
   - Envia comandos aos motores

## Configurações Parametrizáveis

No arquivo `droid_llm.py` é possível ajustar:

```python
# Modelo de linguagem visual
model_name = "Salesforce/blip-image-captioning-base"
# Alternativa: "Salesforce/blip-image-captioning-large" para maior precisão

# Frequência de consulta ao modelo (milissegundos)
self.query_interval = 100

# Pergunta customizável para análise de obstáculos
question = "Is there an obstacle in front?"
```

## Extensões Planejadas

As seguintes melhorias estão planejadas para versões futuras:

- Suporte completo para múltiplos robôs com coordenação
- Comunicação inter-robôs utilizando gRPC
- Integração com modelos VLM mais avançados (LLaVA, GPT-4V)
- Navegação com SLAM para mapeamento ambiental
- Integração com planejadores de trajetória
- Dashboard para monitoramento em tempo real
- Fine-tuning de modelos para tarefas específicas

## Modelos de Linguagem Visual Disponíveis

### BLIP (Configuração Padrão)

- Vantagens: modelo leve (990 MB), velocidade de inferência reduzida, bom desempenho geral
- Desvantagens: precisão inferior comparado com variantes maiores

### BLIP-Large (Alternativa)

- Vantagens: melhor qualidade nas respostas e descrições
- Desvantagens: maior consumo de memória, requer mais VRAM para GPU

Para utilizar a variante maior:

```python
model_name = "Salesforce/blip-image-captioning-large"
```

## Licença

Componentes do Webots são utilizados sob licença de Cyberbotics. Demais componentes do projeto estão disponibilizados sob termos de software livre.

## Autor

Projeto desenvolvido para explorar integração de modelos de linguagem visual com simulação robótica.

Equipe: Marcos, Nicolas Fonteneles, Ravaneli, Rodrigo Viana

Repositório: https://github.com/Nielo-47/llm_multi_robots
Branch: main

---

Última atualização: Novembro de 2025
Status: Em desenvolvimento ativo
