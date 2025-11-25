"""
VLM B - Análise Espacial usando Ollama
Analisa posições, distâncias e direções dos objetos PRÓXIMOS
"""
import ollama
import base64
from typing import Dict, Any


class SpatialVLM:
    """VLM para análise espacial - posições e distâncias de objetos no chão"""
    
    def __init__(self, model_name: str = "llama3.2-vision"):
        """
        Inicializa o modelo de visão para análise espacial.
        
        Args:
            model_name: Nome do modelo Ollama (padrão: llama3.2-vision)
        """
        self.model_name = model_name
        print(f"🔧 VLM B (Espacial) inicializado: {model_name}")
        
    def analyze_spatial(self, image_base64: str) -> Dict[str, Any]:
        """
        Analisa posições espaciais dos objetos na cena.
        
        Args:
            image_base64: Imagem em formato base64
            
        Returns:
            Dict com análise espacial (posições, distâncias, obstáculos)
        """
        # Prompt específico para navegação - IGNORA FUNDO
        prompt = """You are helping a robot navigate. Look at this image from the robot's camera.

CRITICAL: Only consider objects ON THE FLOOR in the FOREGROUND. 
IGNORE: Sky, mountains, distant bushes, background scenery, walls.

Divide the image into 3 vertical zones: LEFT (0-33%), CENTER (33-66%), RIGHT (66-100%)

Questions:
1. LEFT ZONE: Is there a ROCK? If yes, is it CLOSE (big, >20% of image height) or FAR (small)? Answer: no/close/far
2. CENTER ZONE: Is there a ROCK? If yes, is it CLOSE (big, >20% of image height) or FAR (small)? Answer: no/close/far
3. RIGHT ZONE: Is there a ROCK? If yes, is it CLOSE (big, >20% of image height) or FAR (small)? Answer: no/close/far
4. Do you see a POTTED TREE (tree in a pot/vase)? Where? (left/center/right/not visible)
5. Is the potted tree CLOSE (big in image) or FAR (small in image)?

Answer format:
1. LEFT: [no/close/far]
2. CENTER: [no/close/far]
3. RIGHT: [no/close/far]
4. TREE: [left/center/right/not visible]
5. TREE DISTANCE: [close/far]"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_base64]
                }],
                options={
                    'temperature': 0.3,
                    'num_ctx': 1024
                }
            )
            
            description = response['message']['content'].strip()
            print(f"📍 VLM B: {description[:250]}...")
            
            # Parse da resposta para extrair informações espaciais
            desc_lower = description.lower()
            lines = desc_lower.split('\n')
            
            # Análise de obstáculos por zona
            # blocked=True apenas se pedra está CLOSE (perto)
            obstacles = {
                "left": {"blocked": False, "distance": None},
                "center": {"blocked": False, "distance": None},
                "right": {"blocked": False, "distance": None}
            }
            
            # Parser mais robusto - procura linhas específicas
            for line in lines:
                line = line.strip()
                
                # LEFT ZONE - só bloqueia se CLOSE
                if 'left' in line and ('zone' in line or '1.' in line or 'left:' in line):
                    if 'close' in line:
                        obstacles["left"]["blocked"] = True
                        obstacles["left"]["distance"] = "close"
                    elif 'far' in line:
                        obstacles["left"]["blocked"] = False  # Pedra longe, não bloqueia
                        obstacles["left"]["distance"] = "far"
                    elif 'no' in line:
                        obstacles["left"]["blocked"] = False
                
                # CENTER ZONE - só bloqueia se CLOSE
                if ('center' in line or 'middle' in line) and ('zone' in line or '2.' in line or 'center:' in line):
                    if 'close' in line:
                        obstacles["center"]["blocked"] = True
                        obstacles["center"]["distance"] = "close"
                    elif 'far' in line:
                        obstacles["center"]["blocked"] = False  # Pedra longe, não bloqueia
                        obstacles["center"]["distance"] = "far"
                    elif 'no' in line:
                        obstacles["center"]["blocked"] = False
                
                # RIGHT ZONE - só bloqueia se CLOSE
                if 'right' in line and ('zone' in line or '3.' in line or 'right:' in line):
                    if 'close' in line:
                        obstacles["right"]["blocked"] = True
                        obstacles["right"]["distance"] = "close"
                    elif 'far' in line:
                        obstacles["right"]["blocked"] = False  # Pedra longe, não bloqueia
                        obstacles["right"]["distance"] = "far"
                    elif 'no' in line:
                        obstacles["right"]["blocked"] = False
            
            # Fallback: se não encontrou nas linhas, usa busca geral
            # Só marca como bloqueado se mencionar CLOSE
            if not any([obstacles["left"]["distance"], obstacles["center"]["distance"], obstacles["right"]["distance"]]):
                # Procura padrões como "rock on the left close"
                if 'rock' in desc_lower:
                    # Só bloqueia se pedra estiver perto (close/near/big)
                    has_close = 'close' in desc_lower or 'near' in desc_lower or 'big' in desc_lower
                    if has_close:
                        if 'left' in desc_lower:
                            obstacles["left"]["blocked"] = True
                            obstacles["left"]["distance"] = "close"
                        if 'center' in desc_lower or 'middle' in desc_lower:
                            obstacles["center"]["blocked"] = True
                            obstacles["center"]["distance"] = "close"
                        if 'right' in desc_lower:
                            obstacles["right"]["blocked"] = True
                            obstacles["right"]["distance"] = "close"
                    else:
                        # Pedra visível mas longe - registra mas não bloqueia
                        if 'left' in desc_lower:
                            obstacles["left"]["distance"] = "far"
                        if 'center' in desc_lower or 'middle' in desc_lower:
                            obstacles["center"]["distance"] = "far"
                        if 'right' in desc_lower:
                            obstacles["right"]["distance"] = "far"
            
            # Detecta posição da árvore
            tree_position = None
            tree_distance = "medium"
            
            for line in lines:
                line = line.strip()
                # Procura linha sobre a árvore (questão 4)
                if ('tree' in line or 'potted' in line) and ('4.' in line or 'where' in line or 'location' in line):
                    if 'left' in line and 'not' not in line:
                        tree_position = "left"
                    elif 'right' in line and 'not' not in line:
                        tree_position = "right"
                    elif 'center' in line and 'not' not in line:
                        tree_position = "center"
                    elif 'not visible' in line or 'no' in line:
                        tree_position = None
                        
                # Procura linha sobre distância (questão 5)
                if ('close' in line or 'far' in line or 'near' in line) and ('5.' in line or 'distance' in line):
                    if 'close' in line or 'near' in line:
                        tree_distance = "close"
                    elif 'far' in line:
                        tree_distance = "far"
            
            # Fallback para posição da árvore
            if tree_position is None and 'tree' in desc_lower:
                if 'tree' in desc_lower:
                    parts = desc_lower.split('tree')
                    for part in parts[1:] if len(parts) > 1 else []:
                        context = part[:60]
                        if 'left' in context and 'not' not in context:
                            tree_position = "left"
                            break
                        elif 'right' in context and 'not' not in context:
                            tree_position = "right"
                            break
                        elif ('center' in context or 'middle' in context) and 'not' not in context:
                            tree_position = "center"
                            break
                            
                # Distância fallback
                if 'close' in desc_lower or 'near' in desc_lower or 'big' in desc_lower:
                    tree_distance = "close"
                elif 'far' in desc_lower or 'small' in desc_lower or 'distant' in desc_lower:
                    tree_distance = "far"
            
            # Determina caminho mais livre
            clear_path = None
            if not obstacles["right"]["blocked"]:
                clear_path = "right"
            elif not obstacles["left"]["blocked"]:
                clear_path = "left"
            elif not obstacles["center"]["blocked"]:
                clear_path = "center"
            
            return {
                "obstacles": obstacles,
                "tree": {
                    "position": tree_position,
                    "distance": tree_distance
                },
                "clear_path": clear_path,
                "navigation_suggestion": self._suggest_direction(obstacles, tree_position),
                "description": description
            }
            
        except Exception as e:
            print(f"❌ Erro VLM B: {e}")
            return {
                "obstacles": {
                    "left": {"blocked": False, "distance": None},
                    "center": {"blocked": False, "distance": None},
                    "right": {"blocked": False, "distance": None}
                },
                "tree": {"position": None, "distance": None},
                "clear_path": None,
                "navigation_suggestion": "N",
                "description": f"Erro: {e}"
            }
    
    def _suggest_direction(self, obstacles: dict, tree_position: str) -> str:
        """Sugere direção baseado nos obstáculos e posição da árvore"""
        
        # Se há obstáculo no centro, desviar
        if obstacles["center"]["blocked"]:
            if not obstacles["right"]["blocked"]:
                return "E"  # Desviar para direita
            elif not obstacles["left"]["blocked"]:
                return "W"  # Desviar para esquerda
            else:
                return "S"  # Voltar (todos bloqueados)
        
        # Se caminho livre, ir em direção à árvore
        if tree_position == "left":
            if not obstacles["left"]["blocked"]:
                return "NW"
            return "N"
        elif tree_position == "right":
            if not obstacles["right"]["blocked"]:
                return "NE"
            return "N"
        else:
            return "N"


if __name__ == "__main__":
    vlm = SpatialVLM()
    print("VLM B pronto")
