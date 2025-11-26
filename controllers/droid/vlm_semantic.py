"""
VLM A - Análise Semântica usando Ollama
Detecta e descreve objetos na cena - FOCO em objetos PRÓXIMOS
"""
import ollama
import base64
from typing import Dict, Any


class SemanticVLM:
    """VLM para análise semântica - detecção de objetos PRÓXIMOS"""
    
    def __init__(self, model_name: str = "llama3.2-vision"):
        """
        Inicializa o modelo de visão para análise semântica.
        
        Args:
            model_name: Nome do modelo Ollama (padrão: llama3.2-vision)
        """
        self.model_name = model_name
        print(f"🔧 VLM A (Semântico) inicializado: {model_name}")
        
    def analyze_scene(self, image_base64: str) -> Dict[str, Any]:
        """
        Analisa a cena e identifica objetos.
        
        Args:
            image_base64: Imagem em formato base64
            
        Returns:
            Dict com objetos detectados e descrição
        """
        # Prompt MUITO específico para diferenciar objetos próximos de fundo
        prompt = """You are a robot navigation assistant. Analyze this indoor/arena scene.

IMPORTANT: Only describe objects that are CLOSE to the camera (in the foreground).
IGNORE: Mountains, sky, distant bushes, background scenery.

Look for these specific objects:
1. ROCKS/STONES: Large gray/brown rocks on the FLOOR. How many? (0, 1, 2, or 3)
2. POTTED TREE: A tree in a POT/VASE on the floor. NOT bushes in the background.

Answer format:
- ROCKS: [number] rocks visible. Location: [left/center/right of image]
- POTTED TREE: [yes/no]. If yes, location: [left/center/right] and distance: [close/medium/far]
- PATH: Which side has NO obstacles? [left/center/right/none]

Be precise. Only count objects on the FLOOR, not in the background."""

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
            print(f"📝 VLM A (Semântico): {description[:150]}...")
            
            # Parse da resposta para extrair objetos
            desc_lower = description.lower()
            
            objects = []
            stone_count = 0
            has_tree = False
            tree_position = None
            tree_distance = None
            clear_path = None
            
            # Conta pedras
            if 'three' in desc_lower or '3 rock' in desc_lower or '3 stone' in desc_lower:
                stone_count = 3
            elif 'two' in desc_lower or '2 rock' in desc_lower or '2 stone' in desc_lower:
                stone_count = 2
            elif 'one' in desc_lower or '1 rock' in desc_lower or '1 stone' in desc_lower:
                stone_count = 1
            elif 'rock' in desc_lower or 'stone' in desc_lower:
                stone_count = max(desc_lower.count('rock'), desc_lower.count('stone'), 1)
            
            # Adiciona pedras detectadas
            for i in range(min(stone_count, 3)):
                objects.append(f"stone_{i+1}")
                
            # Verifica árvore - DEVE ter "yes" na resposta
            # "POTTED TREE: No" = não encontrou
            # "POTTED TREE: yes" ou "POTTED TREE: Yes" = encontrou
            has_tree = False
            tree_position = None
            tree_distance = None
            
            # Procura a linha específica sobre árvore
            lines = description.split('\n')
            for line in lines:
                line_lower = line.lower()
                # Procura "POTTED TREE:" seguido de sim/não
                if 'potted tree' in line_lower:
                    # Se a linha contém "yes" (considerando variações)
                    if 'yes' in line_lower:
                        has_tree = True
                        objects.append("tree")
                        
                        # Extrai posição e distância
                        if 'left' in line_lower:
                            tree_position = 'left'
                        elif 'right' in line_lower:
                            tree_position = 'right'
                        elif 'center' in line_lower or 'middle' in line_lower:
                            tree_position = 'center'
                        
                        if 'close' in line_lower:
                            tree_distance = 'close'
                        elif 'far' in line_lower:
                            tree_distance = 'far'
                        else:
                            tree_distance = 'medium'
                    # Se contém "no" ou "not visible"
                    elif 'no' in line_lower or 'not visible' in line_lower or 'not found' in line_lower:
                        has_tree = False
                        tree_position = None
                        tree_distance = None
                    break  # Já encontrou a linha sobre árvore
            
            # Caminho livre
            if 'clear_path' in desc_lower:
                clear_part = desc_lower.split('clear_path')[1][:30]
                if 'left' in clear_part:
                    clear_path = 'left'
                elif 'right' in clear_part:
                    clear_path = 'right'
                elif 'center' in clear_part:
                    clear_path = 'center'
            
            return {
                "objects": objects,
                "stone_count": stone_count,
                "has_tree": has_tree,
                "tree_position": tree_position,
                "tree_distance": tree_distance,
                "clear_path": clear_path,
                "description": description,
                "raw_response": description
            }
            
        except Exception as e:
            print(f"❌ Erro VLM A: {e}")
            return {
                "objects": [],
                "stone_count": 0,
                "tree_found": False,
                "description": f"Erro na análise: {str(e)}",
                "raw_response": ""
            }


# Teste
if __name__ == "__main__":
    vlm = SemanticVLM()
    print("VLM A pronto para análise semântica")
