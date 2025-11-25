"""
Cliente para Google Gemini API - Controlador de navegação do robô
"""
import google.generativeai as genai
import json
import base64
from typing import Optional
from pydantic import BaseModel, Field
from typing import Literal


class RobotDecision(BaseModel):
    """Modelo estruturado para decisões de navegação do robô"""
    direction: Literal["N", "S", "E", "W", "NE", "NW", "SE", "SW", "STOP"] = Field(
        description="Direção de navegação"
    )
    speed: Literal["SLOW", "MEDIUM", "FAST"] = Field(description="Velocidade de movimento")
    reason: str = Field(description="Explicação curta da decisão")


class GeminiNavigator:
    """Controlador de navegação usando Google Gemini API"""
    
    def __init__(self, api_key: str):
        """
        Inicializa o cliente Gemini.
        
        Args:
            api_key: Chave da API do Google AI Studio
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        print("✅ Google Gemini API configurada com sucesso!")
        
    def decide_navigation(
        self, 
        scene_analysis: dict,
        spatial_analysis: dict,
        previous_commands: list = None
    ) -> RobotDecision:
        """
        Decide a próxima ação de navegação baseado nas análises visuais.
        
        Args:
            scene_analysis: Resultado da análise semântica (objetos detectados)
            spatial_analysis: Resultado da análise espacial (distâncias, direções)
            previous_commands: Lista de comandos anteriores
            
        Returns:
            RobotDecision com direção, velocidade e razão
        """
        previous_str = "Nenhum comando anterior"
        if previous_commands:
            recent = previous_commands[-3:]
            previous_str = "\n".join([
                f"- {cmd.direction} ({cmd.speed}): {cmd.reason}" 
                for cmd in recent
            ])
        
        prompt = f"""Você controla um robô que deve IR ATÉ A ÁRVORE desviando das PEDRAS.

ANÁLISE DA CENA:
{json.dumps(scene_analysis, indent=2, ensure_ascii=False)}

ANÁLISE ESPACIAL:
{json.dumps(spatial_analysis, indent=2, ensure_ascii=False)}

COMANDOS ANTERIORES:
{previous_str}

=== REGRAS OBRIGATÓRIAS ===

1. EVITAR COLISÃO (prioridade máxima):
   - Se "center" está BLOCKED com distância CLOSE: DESVIE IMEDIATAMENTE (E ou W)
   - Se "center" está BLOCKED com distância FAR: pode ir NE ou NW
   
2. DESVIAR PEDRAS:
   - Pedra à esquerda (left blocked): vá para DIREITA (E, NE)
   - Pedra à direita (right blocked): vá para ESQUERDA (W, NW)
   - Pedra no centro: escolha o lado livre (E se right clear, W se left clear)

3. IR PARA ÁRVORE:
   - Árvore à esquerda: NW (se left não bloqueado)
   - Árvore à direita: NE (se right não bloqueado)  
   - Árvore ao centro: N (se center não bloqueado)

4. CHEGOU:
   - Se tree distance = "close": STOP

5. VELOCIDADE:
   - SLOW: perto de obstáculos ou árvore próxima
   - MEDIUM: caminho livre

=== RESPOSTA ===
Responda APENAS com JSON:
{{
    "direction": "N/S/E/W/NE/NW/SE/SW/STOP",
    "speed": "SLOW/MEDIUM",
    "reason": "explicação curta"
}}

JSON:"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Limpa o texto para extrair apenas o JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Tenta encontrar o JSON na resposta
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                response_text = response_text[start_idx:end_idx]
            
            data = json.loads(response_text)
            return RobotDecision(**data)
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Erro ao parsear JSON do Gemini: {e}")
            print(f"   Resposta raw: {response_text[:200]}...")
            return RobotDecision(
                direction="STOP",
                speed="SLOW",
                reason="Erro ao processar resposta do Gemini"
            )
        except Exception as e:
            print(f"❌ Erro na API Gemini: {e}")
            return RobotDecision(
                direction="STOP",
                speed="SLOW",
                reason=f"Erro na API: {str(e)}"
            )


# Teste rápido
if __name__ == "__main__":
    API_KEY = "AIzaSyAzY9Bit9DZR1PSX_gjOJUVxcT0Q3ELBX8"
    
    navigator = GeminiNavigator(API_KEY)
    
    # Teste com dados simulados
    scene = {
        "objects": ["stone", "stone", "stone", "tree"],
        "description": "Três pedras cinzas no centro e uma árvore verde ao fundo à direita"
    }
    
    spatial = {
        "obstacles": {
            "center": {"type": "stone", "distance": "close"},
            "left": {"type": "stone", "distance": "medium"},
            "right": {"type": "clear", "distance": "far"}
        },
        "target": {
            "tree": {"position": "right", "distance": "far"}
        }
    }
    
    decision = navigator.decide_navigation(scene, spatial)
    print(f"\n🎯 Decisão: {decision.direction} ({decision.speed})")
    print(f"   Razão: {decision.reason}")
