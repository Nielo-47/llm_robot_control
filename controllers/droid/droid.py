from controller import Robot
import threading
from vlm import GeminiRobotController, RobotDecision
from devices import Wheels, Camera


class NavigationPhase:
    """Estados de navegação do droide"""

    SCANNING = "SCANNING"  # Analisando cena
    NAVIGATING = "NAVIGATING"  # Movendo-se em direção ao objetivo
    AVOIDING = "AVOIDING"  # Contornando obstáculo
    REPOSITIONING = "REPOSITIONING"  # Reposicionando após desvio
    SEARCHING = "SEARCHING"  # Procurando pela árvore (sem nada visível)
    ARRIVED = "ARRIVED"  # Chegou ao destino


class Droid:
    def __init__(self):
        # Usa Supervisor ao invés de Robot para ter acesso às coordenadas
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.wheels = Wheels(self.robot, max_speed=6.28)
        self.camera = Camera(self.robot, self.timestep)
        self.controller = GeminiRobotController()

        self.current_command = RobotDecision(
            direction="STOP", speed="SLOW", reason="Initializing"
        )
        self.is_processing = False

    def process_frame_async(self, image_b64, frame_num):
        """Background thread for VLM processing"""
        try:
            self.is_processing = True

            result = self.controller.generate_command(image_b64)

            if result:
                self.current_command = result
                print(
                    f"✅ [{frame_num}] {result.direction} {result.speed} - {result.reason}"
                )

        except Exception as e:
            print(f"❌ [{frame_num}] {e}")
            self.current_command = RobotDecision(
                direction="STOP", speed="SLOW", reason="Error"
            )
        finally:
            self.is_processing = False

    def _smart_navigation(
        self,
        semantic,
        spatial,
        left_blocked,
        center_blocked,
        right_blocked,
        tree_pos,
        tree_dist,
        clear_path,
    ):
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

                dist_to_tree = math.sqrt(
                    (TREE_X - robot_x) ** 2 + (TREE_Y - robot_y) ** 2
                )
                print(
                    f"   📍 Posição: ({robot_x:.1f}, {robot_y:.1f}) | Distância árvore: {dist_to_tree:.1f}m"
                )
        except:
            pass

        # ═══════════════════════════════════════════════════════════
        # CASO 1: Chegou à árvore (verifica por GPS, não por VLM!)
        # ═══════════════════════════════════════════════════════════
        if robot_x is not None:
            # Só considera "chegou" se estiver a menos de 2m da árvore
            if robot_x > 7.0 and abs(robot_y) < 2.0:
                self.phase = NavigationPhase.ARRIVED
                return RobotDecision(
                    direction="STOP",
                    speed="SLOW",
                    reason="🎉 Chegou na árvore! (GPS confirmado)",
                )

        # Fallback: se VLM diz que árvore está perto E robô está em X > 5
        if tree_dist == "close" and self.tree_found:
            if robot_x is not None and robot_x > 5.0:
                self.phase = NavigationPhase.ARRIVED
                return RobotDecision(
                    direction="STOP", speed="SLOW", reason="🎉 Chegou na árvore!"
                )
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
                return RobotDecision(
                    direction=BASE_DIRECTION,
                    speed="MEDIUM",
                    reason="Timeout - voltando rota base",
                )

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
                return RobotDecision(
                    direction="N", speed="MEDIUM", reason="Contorno completo, avançando"
                )

            # Move em diagonal para contornar E avançar ao mesmo tempo
            # Como começamos NW, contornamos SW para passar por baixo das pedras
            if self.avoid_direction == "W":
                return RobotDecision(
                    direction="SW",
                    speed="MEDIUM",
                    reason=f"Contornando SW (passo {self.avoid_steps}/8)",
                )
            else:
                return RobotDecision(
                    direction="SE",
                    speed="MEDIUM",
                    reason=f"Contornando SE (passo {self.avoid_steps}/8)",
                )

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
                return RobotDecision(
                    direction=self.avoid_direction,
                    speed="MEDIUM",
                    reason="Obstáculo encontrado, continuando contorno",
                )

            return RobotDecision(
                direction=BASE_DIRECTION,
                speed="MEDIUM",
                reason=f"Avançando após contorno (passo {self.avoid_steps}/5)",
            )

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
            return RobotDecision(
                direction="SW",
                speed="MEDIUM",
                reason="Contorno forçado (oscilação) - lado SW",
            )

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
            return RobotDecision(
                direction=direction,
                speed="MEDIUM",
                reason=f"Buscando árvore (padrão: {direction})",
            )

        # COMPORTAMENTO 2: Árvore visível E pedras visíveis mas NÃO no centro → Segue em frente
        if self.tree_found and self.stones_found > 0 and not center_blocked:
            self.phase = NavigationPhase.NAVIGATING
            print("   ✅ Árvore visível, pedras nos lados (não vão bater) → Avançando!")

            # Se árvore está visível, vai direto pra ela
            if tree_pos == "left":
                return RobotDecision(
                    direction="NW",
                    speed="MEDIUM",
                    reason="Árvore à esquerda, avançando NW",
                )
            elif tree_pos == "right":
                return RobotDecision(
                    direction="NE",
                    speed="MEDIUM",
                    reason="Árvore à direita, avançando NE",
                )
            else:
                return RobotDecision(
                    direction="N",
                    speed="MEDIUM",
                    reason="Árvore no centro, avançando N",
                )

        # COMPORTAMENTO 3: Árvore visível E nenhuma pedra visível → Segue em frente
        if self.tree_found and self.stones_found == 0:
            self.phase = NavigationPhase.NAVIGATING
            print("   🌳 Árvore visível, nenhuma pedra → Caminho livre!")

            if tree_pos == "left":
                return RobotDecision(
                    direction="NW",
                    speed="MEDIUM",
                    reason="Árvore à esquerda, avançando NW",
                )
            elif tree_pos == "right":
                return RobotDecision(
                    direction="NE",
                    speed="MEDIUM",
                    reason="Árvore à direita, avançando NE",
                )
            else:
                return RobotDecision(
                    direction="N",
                    speed="MEDIUM",
                    reason="Árvore no centro, avançando N",
                )

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
                return RobotDecision(
                    direction="SW",
                    speed="MEDIUM",
                    reason="Pedra PERTO no centro! Desviando SW",
                )
            elif not right_blocked:
                self.avoid_direction = "E"
                return RobotDecision(
                    direction="SE",
                    speed="MEDIUM",
                    reason="Esquerda bloqueada, desviando SE",
                )
            else:
                # Tudo bloqueado - vai reto tentando passar
                return RobotDecision(
                    direction="N",
                    speed="SLOW",
                    reason="Lados bloqueados, tentando passar pelo centro",
                )

        # ═══════════════════════════════════════════════════════════
        # CASO 7: Pedras nos lados mas centro livre - Avança
        # ═══════════════════════════════════════════════════════════
        if self.stones_found > 0 and not center_blocked:
            self.phase = NavigationPhase.NAVIGATING
            return RobotDecision(
                direction="N",
                speed="MEDIUM",
                reason="Centro livre, avançando entre pedras",
            )

        # ═══════════════════════════════════════════════════════════
        # CASO 8: Caminho livre - Avança em direção à árvore
        # ═══════════════════════════════════════════════════════════
        self.phase = NavigationPhase.NAVIGATING

        # Se vê a árvore, vai na direção dela
        if self.tree_found and tree_pos:
            if tree_pos == "left" and not left_blocked:
                return RobotDecision(
                    direction="NW",
                    speed="MEDIUM",
                    reason="Árvore visível à esquerda, indo NW",
                )
            elif tree_pos == "right" and not right_blocked:
                return RobotDecision(
                    direction="NE",
                    speed="MEDIUM",
                    reason="Árvore visível à direita, indo NE",
                )
            elif tree_pos == "center" and not center_blocked:
                return RobotDecision(
                    direction="N",
                    speed="MEDIUM",
                    reason="Árvore visível no centro, indo N",
                )

        # Direção padrão: NE (direção da árvore no mapa)
        if not right_blocked:
            return RobotDecision(
                direction=BASE_DIRECTION,
                speed="MEDIUM",
                reason="Rota base para árvore (NE)",
            )
        elif not center_blocked:
            return RobotDecision(
                direction="N", speed="MEDIUM", reason="Direita bloqueada, indo N"
            )
        else:
            return RobotDecision(
                direction="NW", speed="MEDIUM", reason="Centro bloqueado, indo NW"
            )

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
            if (
                dirs == ["E", "W", "E", "W"]
                or dirs == ["W", "E", "W", "E"]
                or dirs == ["NE", "W", "NE", "W"]
                or dirs == ["NW", "E", "NW", "E"]
            ):
                self.oscillation_count += 1
                print(f"   ⚠️ Oscilação detectada! ({self.oscillation_count})")
            else:
                self.oscillation_count = 0

    def run(self):
        print("🤖 Starting navigation...\n")

        # Warmup
        for _ in range(300):
            self.robot.step(self.timestep)

        frame_count = 0

        try:
            while self.robot.step(self.timestep) != -1:
                frame_count += 1

                # Process every 10th frame
                if not self.is_processing and frame_count % 10 == 0:
                    try:
                        image_b64 = self.camera.get_camera_image()
                        if image_b64:
                            thread = threading.Thread(
                                target=self.process_frame_async,
                                args=(image_b64, frame_count),
                                daemon=True,
                            )
                            thread.start()
                    except Exception as e:
                        print(f"⚠️ [{frame_count}] Camera error: {e}")

                # Execute current command
                self.wheels.execute_command(self.current_command)

                # Status update every 500 frames
                if frame_count % 500 == 0:
                    status = "Processing..." if self.is_processing else "Ready"
                    print(f"[{frame_count}] {status}")

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        finally:
            self.wheels.stop()
            print("👋 Shutdown complete")


if __name__ == "__main__":
    Droid().run()
