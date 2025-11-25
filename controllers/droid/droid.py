from controller import Supervisor
import warnings
import threading
from devices import Wheels, Camera, image_to_base64

# Nova arquitetura: 2 VLMs + Gemini
from vlm_semantic import SemanticVLM
from vlm_spatial import SpatialVLM
from gemini_client import GeminiNavigator, RobotDecision

# Sistema de logging de coordenadas
from coordinate_logger import CoordinateLogger

warnings.filterwarnings("ignore")


class NavigationPhase:
    """Estados de navegação do droide"""
    SCANNING = "SCANNING"           # Analisando cena
    NAVIGATING = "NAVIGATING"       # Movendo-se em direção ao objetivo
    AVOIDING = "AVOIDING"           # Contornando obstáculo
    REPOSITIONING = "REPOSITIONING" # Reposicionando após desvio
    SEARCHING = "SEARCHING"         # Procurando pela árvore (sem nada visível)
    ARRIVED = "ARRIVED"             # Chegou ao destino


class Droid:
    def __init__(self):
        # Usa Supervisor ao invés de Robot para ter acesso às coordenadas
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.wheels = Wheels(self.robot, max_speed=6.28)
        self.camera = Camera(self.robot, self.timestep)
        
        # Sistema de logging de coordenadas
        self.coord_logger = CoordinateLogger(self.robot)
        
        # Nova arquitetura: 2 VLMs locais + Gemini externo
        print("\n🔧 Inicializando sistema de visão e navegação...")
        self.vlm_semantic = SemanticVLM()   # VLM A - Análise semântica
        self.vlm_spatial = SpatialVLM()     # VLM B - Análise espacial
        
        # API Key do Google Gemini
        GEMINI_API_KEY = "AIzaSyAzY9Bit9DZR1PSX_gjOJUVxcT0Q3ELBX8"
        self.gemini = GeminiNavigator(GEMINI_API_KEY)  # LLM externo - Decisões

        # Navigation state
        self.phase = NavigationPhase.NAVIGATING  # Começa navegando, não scanning
        # Direção BASE: Robô em X=-12, Árvore em X=9 → Começando com NW para contornar obstáculos
        self.current_command = RobotDecision(direction="NW", speed="MEDIUM", reason="Direção inicial NW para contornar")
        
        # Object tracking
        self.tree_found = False
        self.tree_position = None  # left/center/right
        self.stones_found = 0
        self.approach_counter = 0
        self.first_vlm_done = False  # Flag para saber se já processou primeira imagem
        
        # Obstacle avoidance
        self.avoid_direction = None  # Direção escolhida para desvio
        self.avoid_steps = 0         # Passos no modo de desvio
        self.avoid_timeout = 0       # Timeout para não ficar preso em AVOIDING
        self.oscillation_count = 0   # Contador de oscilações E/W
        self.last_directions = []    # Últimas direções para detectar oscilação
        
        # Estado atual dos obstáculos (para logging)
        self.current_left_blocked = False
        self.current_center_blocked = False
        self.current_right_blocked = False
        
        # Thread management
        self.command_lock = threading.Lock()
        self.is_processing = False
        self.vlm_thread = None

        print("\n🤖 Droide Inteligente Iniciado!")
        print("📸 Camera: {}x{}".format(
            self.camera.camera.getWidth(), 
            self.camera.camera.getHeight()
        ))
        print("🎯 Objetivo: Encontrar 3 pedras e 1 árvore, navegar até a árvore")
        print("🧭 Direção base: NE (árvore está à direita-frente)")
        print("=" * 60 + "\n")

    def process_vlm_pipeline(self, image_base64: str):
        """
        Pipeline principal: VLM A + VLM B → Gemini → Decisão
        Com lógica de contorno de obstáculos
        """
        try:
            self.is_processing = True
            
            print("\n" + "=" * 60)
            print(f"🔄 PROCESSANDO NOVA IMAGEM | Fase: {self.phase}")
            print("=" * 60)
            
            # ═══════════════════════════════════════════════════════════
            # PASSO 1: VLM A - Análise Semântica (objetos)
            # ═══════════════════════════════════════════════════════════
            print("\n📊 VLM A - Análise Semântica...")
            semantic_result = self.vlm_semantic.analyze_scene(image_base64)
            
            self.stones_found = semantic_result.get("stone_count", 0)
            self.tree_found = semantic_result.get("has_tree", False)
            
            print(f"   └─ Pedras: {self.stones_found} | Árvore: {'✅' if self.tree_found else '❌'}")
            
            # ═══════════════════════════════════════════════════════════
            # PASSO 2: VLM B - Análise Espacial (posições)
            # ═══════════════════════════════════════════════════════════
            print("\n📍 VLM B - Análise Espacial...")
            spatial_result = self.vlm_spatial.analyze_spatial(image_base64)
            
            tree_pos = spatial_result.get("tree", {}).get("position")
            tree_dist = spatial_result.get("tree", {}).get("distance", "medium")
            clear_path = spatial_result.get("clear_path")
            
            # Atualiza posição da árvore se detectada
            if tree_pos:
                self.tree_position = tree_pos
            
            # Status dos obstáculos
            obs = spatial_result.get("obstacles", {})
            left_blocked = obs.get("left", {}).get("blocked", False)
            center_blocked = obs.get("center", {}).get("blocked", False)
            right_blocked = obs.get("right", {}).get("blocked", False)
            
            left_status = "🚫" if left_blocked else "✅"
            center_status = "🚫" if center_blocked else "✅"
            right_status = "🚫" if right_blocked else "✅"
            
            # Armazena estado dos obstáculos para logging
            self.current_left_blocked = left_blocked
            self.current_center_blocked = center_blocked
            self.current_right_blocked = right_blocked
            
            print(f"   └─ Zonas: L{left_status} C{center_status} R{right_status}")
            print(f"   └─ Árvore: {tree_pos} ({tree_dist}) | Caminho livre: {clear_path}")
            
            # ═══════════════════════════════════════════════════════════
            # PASSO 3: Lógica de Navegação Inteligente
            # ═══════════════════════════════════════════════════════════
            decision = self._smart_navigation(
                semantic_result, spatial_result,
                left_blocked, center_blocked, right_blocked,
                tree_pos, tree_dist, clear_path
            )
            
            print(f"\n🎯 Decisão Final: {decision.direction} ({decision.speed})")
            print(f"   └─ Razão: {decision.reason}")
            
            # Atualiza comando
            with self.command_lock:
                self.current_command = decision
            
            # Rastreia direções para detectar oscilação
            self._track_oscillation(decision.direction)
            
        except Exception as e:
            print(f"\n❌ Erro no pipeline: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_processing = False
    
    def _smart_navigation(self, semantic, spatial, left_blocked, center_blocked, right_blocked, tree_pos, tree_dist, clear_path):
        """
        Navegação inteligente com contorno de obstáculos
        
        MAPA DO MUNDO:
        - Robô começa em X=-12, Y=4
        - Árvore está em X=9.12, Y=0
        - Pedras estão em X≈0, Y≈0 (centro do mapa)
        - Portanto: árvore está à DIREITA (E) e um pouco à FRENTE (N)
        - Direção ideal: NE (nordeste)
        
        Sistema de coordenadas do robô:
        - N = Frente (direção que está olhando)
        - S = Trás
        - E = Direita
        - W = Esquerda
        """
        
        # Direção base para a árvore (calculada pelo mapa)
        BASE_DIRECTION = "NE"  # Árvore está à direita-frente
        
        # Posição da árvore no mundo
        TREE_X = 9.12
        TREE_Y = 0.0
        
        # ═══════════════════════════════════════════════════════════
        # OBTÉM POSIÇÃO REAL DO ROBÔ (GPS)
        # ═══════════════════════════════════════════════════════════
        robot_x, robot_y = None, None
        try:
            pos = self.coord_logger.get_robot_position()
            if pos[0] is not None:
                robot_x, robot_y = pos[0], pos[1]
                
                # Calcula distância até a árvore
                import math
                dist_to_tree = math.sqrt((TREE_X - robot_x)**2 + (TREE_Y - robot_y)**2)
                print(f"   📍 Posição: ({robot_x:.1f}, {robot_y:.1f}) | Distância árvore: {dist_to_tree:.1f}m")
        except:
            pass
        
        # ═══════════════════════════════════════════════════════════
        # CASO 1: Chegou à árvore (verifica por GPS, não por VLM!)
        # ═══════════════════════════════════════════════════════════
        if robot_x is not None:
            # Só considera "chegou" se estiver a menos de 2m da árvore
            if robot_x > 7.0 and abs(robot_y) < 2.0:
                self.phase = NavigationPhase.ARRIVED
                return RobotDecision(direction="STOP", speed="SLOW", reason="🎉 Chegou na árvore! (GPS confirmado)")
        
        # Fallback: se VLM diz que árvore está perto E robô está em X > 5
        if tree_dist == "close" and self.tree_found:
            if robot_x is not None and robot_x > 5.0:
                self.phase = NavigationPhase.ARRIVED
                return RobotDecision(direction="STOP", speed="SLOW", reason="🎉 Chegou na árvore!")
            else:
                # VLM errou - provavelmente está vendo pedra de perto
                print("   ⚠️ VLM diz árvore perto, mas GPS diz que não! Ignorando...")
        
        # ═══════════════════════════════════════════════════════════
        # CASO 2: Timeout de AVOIDING - Não ficar preso
        # ═══════════════════════════════════════════════════════════
        if self.phase == NavigationPhase.AVOIDING:
            self.avoid_timeout += 1
            
            # Timeout: se ficou muito tempo evitando, força avançar
            if self.avoid_timeout > 15:
                print("   ⏰ Timeout AVOIDING! Forçando avanço...")
                self.phase = NavigationPhase.NAVIGATING
                self.avoid_timeout = 0
                self.avoid_steps = 0
                return RobotDecision(direction=BASE_DIRECTION, speed="MEDIUM", reason="Timeout - voltando rota base")
        
        # ═══════════════════════════════════════════════════════════
        # CASO 3: Modo AVOIDING - Contornando lateralmente
        # ═══════════════════════════════════════════════════════════
        if self.phase == NavigationPhase.AVOIDING:
            self.avoid_steps += 1
            print(f"   🔄 Contornando (passo {self.avoid_steps}/8)")
            
            # Contorna por 8 passos para garantir que passou completamente
            if self.avoid_steps >= 8:
                self.phase = NavigationPhase.REPOSITIONING
                self.avoid_steps = 0
                return RobotDecision(direction="N", speed="MEDIUM", reason="Contorno completo, avançando")
            
            # Move em diagonal para contornar E avançar ao mesmo tempo
            # Como começamos NW, contornamos SW para passar por baixo das pedras
            if self.avoid_direction == "W":
                return RobotDecision(direction="SW", speed="MEDIUM", reason=f"Contornando SW (passo {self.avoid_steps}/8)")
            else:
                return RobotDecision(direction="SE", speed="MEDIUM", reason=f"Contornando SE (passo {self.avoid_steps}/8)")
        
        # ═══════════════════════════════════════════════════════════
        # CASO 4: Modo REPOSITIONING - Avançando após contorno
        # ═══════════════════════════════════════════════════════════
        if self.phase == NavigationPhase.REPOSITIONING:
            self.avoid_steps += 1
            
            # Avança por 5 passos
            if self.avoid_steps >= 5:
                self.phase = NavigationPhase.NAVIGATING
                self.avoid_steps = 0
                self.avoid_timeout = 0
            
            # Se encontrar obstáculo novamente, volta a contornar
            if center_blocked:
                self.phase = NavigationPhase.AVOIDING
                self.avoid_steps = 0
                return RobotDecision(direction=self.avoid_direction, speed="MEDIUM", reason="Obstáculo encontrado, continuando contorno")
            
            return RobotDecision(direction=BASE_DIRECTION, speed="MEDIUM", reason=f"Avançando após contorno (passo {self.avoid_steps}/5)")
        
        # ═══════════════════════════════════════════════════════════
        # CASO 5: Detectou oscilação - Força contorno longo
        # ═══════════════════════════════════════════════════════════
        if self.oscillation_count >= 2:
            print("   ⚠️ Oscilação detectada! Contorno forçado...")
            self.oscillation_count = 0
            self.phase = NavigationPhase.AVOIDING
            self.avoid_steps = 0
            self.avoid_timeout = 0
            # Prefere desviar pela ESQUERDA (SW) - começamos com NW
            self.avoid_direction = "W"
            return RobotDecision(direction="SW", speed="MEDIUM", reason="Contorno forçado (oscilação) - lado SW")
        
        # ═══════════════════════════════════════════════════════════
        # NOVO COMPORTAMENTO INTELIGENTE
        # ═══════════════════════════════════════════════════════════
        
        # COMPORTAMENTO 1: Nenhuma pedra E nenhuma árvore visível → Busca girando
        if self.stones_found == 0 and not self.tree_found:
            print("   🔍 Árvore não encontrada! Fazendo curvas para procurar...")
            self.phase = NavigationPhase.SEARCHING
            # Faz curvas em espiral (NW → N → NE → N → NW)
            search_pattern = ["NW", "N", "NE", "N", "NW"]
            pattern_index = len(self.last_directions) % len(search_pattern)
            direction = search_pattern[pattern_index]
            return RobotDecision(direction=direction, speed="MEDIUM", reason=f"Buscando árvore (padrão: {direction})")
        
        # COMPORTAMENTO 2: Árvore visível E pedras visíveis mas NÃO no centro → Segue em frente
        if self.tree_found and self.stones_found > 0 and not center_blocked:
            self.phase = NavigationPhase.NAVIGATING
            print("   ✅ Árvore visível, pedras nos lados (não vão bater) → Avançando!")
            
            # Se árvore está visível, vai direto pra ela
            if tree_pos == "left":
                return RobotDecision(direction="NW", speed="MEDIUM", reason="Árvore à esquerda, avançando NW")
            elif tree_pos == "right":
                return RobotDecision(direction="NE", speed="MEDIUM", reason="Árvore à direita, avançando NE")
            else:
                return RobotDecision(direction="N", speed="MEDIUM", reason="Árvore no centro, avançando N")
        
        # COMPORTAMENTO 3: Árvore visível E nenhuma pedra visível → Segue em frente
        if self.tree_found and self.stones_found == 0:
            self.phase = NavigationPhase.NAVIGATING
            print("   🌳 Árvore visível, nenhuma pedra → Caminho livre!")
            
            if tree_pos == "left":
                return RobotDecision(direction="NW", speed="MEDIUM", reason="Árvore à esquerda, avançando NW")
            elif tree_pos == "right":
                return RobotDecision(direction="NE", speed="MEDIUM", reason="Árvore à direita, avançando NE")
            else:
                return RobotDecision(direction="N", speed="MEDIUM", reason="Árvore no centro, avançando N")
        
        # ═══════════════════════════════════════════════════════════
        # CASO 6: PEDRAS VISÍVEIS E PRÓXIMAS NO CENTRO - Inicia desvio
        # Só desvia se a pedra estiver CLOSE (grande na imagem)
        # ═══════════════════════════════════════════════════════════
        if self.stones_found > 0 and center_blocked:
            self.phase = NavigationPhase.AVOIDING
            self.avoid_steps = 0
            self.avoid_timeout = 0
            
            # Prefere desviar pelo lado oposto: se começou NW, desvia por SW/W/NW
            # Se right_blocked também, vai pra esquerda
            if not left_blocked:
                self.avoid_direction = "W"
                return RobotDecision(direction="SW", speed="MEDIUM", reason="Pedra PERTO no centro! Desviando SW")
            elif not right_blocked:
                self.avoid_direction = "E"
                return RobotDecision(direction="SE", speed="MEDIUM", reason="Esquerda bloqueada, desviando SE")
            else:
                # Tudo bloqueado - vai reto tentando passar
                return RobotDecision(direction="N", speed="SLOW", reason="Lados bloqueados, tentando passar pelo centro")
        
        # ═══════════════════════════════════════════════════════════
        # CASO 7: Pedras nos lados mas centro livre - Avança
        # ═══════════════════════════════════════════════════════════
        if self.stones_found > 0 and not center_blocked:
            self.phase = NavigationPhase.NAVIGATING
            return RobotDecision(direction="N", speed="MEDIUM", reason="Centro livre, avançando entre pedras")
        
        # ═══════════════════════════════════════════════════════════
        # CASO 8: Caminho livre - Avança em direção à árvore
        # ═══════════════════════════════════════════════════════════
        self.phase = NavigationPhase.NAVIGATING
        
        # Se vê a árvore, vai na direção dela
        if self.tree_found and tree_pos:
            if tree_pos == "left" and not left_blocked:
                return RobotDecision(direction="NW", speed="MEDIUM", reason="Árvore visível à esquerda, indo NW")
            elif tree_pos == "right" and not right_blocked:
                return RobotDecision(direction="NE", speed="MEDIUM", reason="Árvore visível à direita, indo NE")
            elif tree_pos == "center" and not center_blocked:
                return RobotDecision(direction="N", speed="MEDIUM", reason="Árvore visível no centro, indo N")
        
        # Direção padrão: NE (direção da árvore no mapa)
        if not right_blocked:
            return RobotDecision(direction=BASE_DIRECTION, speed="MEDIUM", reason="Rota base para árvore (NE)")
        elif not center_blocked:
            return RobotDecision(direction="N", speed="MEDIUM", reason="Direita bloqueada, indo N")
        else:
            return RobotDecision(direction="NW", speed="MEDIUM", reason="Centro bloqueado, indo NW")
    
    def _track_oscillation(self, direction: str):
        """Detecta padrão de oscilação E-W-E-W"""
        self.last_directions.append(direction)
        
        # Mantém só as últimas 4 direções
        if len(self.last_directions) > 4:
            self.last_directions.pop(0)
        
        # Verifica oscilação
        if len(self.last_directions) >= 4:
            dirs = self.last_directions[-4:]
            # Padrão E-W-E-W ou W-E-W-E
            if (dirs == ["E", "W", "E", "W"] or 
                dirs == ["W", "E", "W", "E"] or
                dirs == ["NE", "W", "NE", "W"] or
                dirs == ["NW", "E", "NW", "E"]):
                self.oscillation_count += 1
                print(f"   ⚠️ Oscilação detectada! ({self.oscillation_count})")
            else:
                self.oscillation_count = 0

    def run(self):
        """Loop principal de controle"""
        print("🚀 Sistema de navegação inteligente iniciado!")
        print("   Pipeline: VLM A (semântico) + VLM B (espacial) → Gemini → Ação")
        print("=" * 60)
        
        frame_count = 0
        
        try:
            while self.robot.step(self.timestep) != -1:
                frame_count += 1
                
                # Para se chegou ao destino
                if self.phase == NavigationPhase.ARRIVED:
                    self.wheels.execute_command(
                        RobotDecision(direction="STOP", speed="SLOW", reason="Destino alcançado"),
                        force_print=True
                    )
                    # Log do evento de chegada
                    self.coord_logger.log_event(frame_count, "ARRIVED", "Robô chegou ao destino (árvore)")
                    continue

                # Captura imagem a cada 20 frames (dar tempo para processamento)
                if not self.is_processing and frame_count % 20 == 0:
                    new_image = self.camera.get_camera_image()
                    
                    if new_image:
                        # Converte para base64 para os VLMs
                        image_b64 = image_to_base64(new_image)
                        
                        # Inicia thread de processamento
                        self.vlm_thread = threading.Thread(
                            target=self.process_vlm_pipeline,
                            args=(image_b64,),
                            daemon=True
                        )
                        self.vlm_thread.start()

                # Executa comando atual
                with self.command_lock:
                    command_to_execute = self.current_command

                self.wheels.execute_command(command_to_execute)
                
                # ══════════════════════════════════════════════════════════
                # LOG DE COORDENADAS - a cada 10 frames
                # ══════════════════════════════════════════════════════════
                if frame_count % 10 == 0:
                    self.coord_logger.log(
                        frame=frame_count,
                        phase=self.phase,
                        direction=command_to_execute.direction,
                        speed=command_to_execute.speed,
                        stones_detected=self.stones_found,
                        tree_visible=self.tree_found,
                        tree_position=self.tree_position,
                        left_blocked=self.current_left_blocked,
                        center_blocked=self.current_center_blocked,
                        right_blocked=self.current_right_blocked,
                        reason=command_to_execute.reason
                    )
        
        except KeyboardInterrupt:
            print("\n⚠️ Simulação interrompida pelo usuário")
        finally:
            # Finaliza o logger e mostra resumo
            self.coord_logger.close()


if __name__ == "__main__":
    controller = Droid()
    controller.run()