# LLM Multi Robots

Simulação de robôs móveis com capacidades de compreensão visual através de modelos de linguagem visual integrados ao Webots.

## Visão Geral

Este repositório implementa controladores para robôs autônomos que utilizam visão computacional com aprendizado de máquina para compreender e navegar em seu ambiente. O projeto integra:

- Ambiente de simulação Webots com física realista para múltiplos agentes robóticos
- Modelo de visão acoplado a um modelo de linguagem para geração de comandos

## Objetivo

O trabalho visa desenvolver um sistema robótico capaz de:

- Capturar imagens em tempo real a partir de câmeras embarcadas
- Processar informações visuais e gerar descrições semânticas de cenas
- Executar navegação autônoma com detecção de obstáculos baseada em visão
- Escalar para múltiplos robôs com coordenação distribuída

### Requisitos do Sistema

- Webots R2025a instalado
- Python 3.12 ou superior
- Ollama instalado e executando

## Dependências

Instale os pacotes Python necessários através do gerenciador de pacotes:

```bash
pip install -r requirements.txt
```

Instale o modelo de linguagem utilizado. Abra o terminal e execute:

```bash
ollama pull gemma3:1b-it-qat
```

## Como Usar

### Executando a Simulação

1. Abra o Webots
2. Carregue o arquivo de mundo: `worlds/LLM_ORCHESTRATOR.wbt`
3. Inicie a simulação pressionando o botão de play
4. O controller será carregado automaticamente
5. Acompanhe a saída no console do Webots

## Extensões Planejadas

As seguintes melhorias estão planejadas para versões futuras:

- Suporte completo para múltiplos robôs com coordenação
- Comunicação inter-robôs utilizando gRPC
- Navegação com SLAM para mapeamento ambiental
- Integração com planejadores de trajetória
- Fine-tuning de modelos para tarefas específicas

## Licença

Componentes do Webots são utilizados sob licença de Cyberbotics. Demais componentes do projeto estão disponibilizados sob termos de software livre.

## Autor

Projeto desenvolvido para explorar integração de modelos de linguagem visual com simulação robótica.

Equipe: Marcos, Nícolas Fonteles, Ravanelli, Rodrigo Viana

Repositório: https://github.com/Nielo-47/llm_multi_robots
Branch: main

---

Última atualização: Novembro de 2025
Status: Em desenvolvimento ativo
