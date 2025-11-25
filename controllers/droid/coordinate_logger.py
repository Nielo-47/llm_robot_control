"""
Sistema de Logging de Coordenadas
Rastreia o caminho do robô no mapa e salva em arquivo CSV
"""
import os
import csv
from datetime import datetime
from typing import Optional


class CoordinateLogger:
    """
    Logger de coordenadas do robô para análise de trajetória
    
    Salva em CSV:
    - Timestamp
    - Frame
    - Posição X, Y, Z
    - Rotação (ângulo)
    - Fase de navegação
    - Direção comandada
    - Velocidade
    - Pedras detectadas
    - Árvore visível
    - Razão da decisão
    """
    
    def __init__(self, robot, log_dir: str = None):
        """
        Inicializa o logger
        
        Args:
            robot: Instância do Webots Robot (deve ser Supervisor)
            log_dir: Diretório para salvar logs (padrão: controllers/droid/logs)
        """
        self.robot = robot
        
        # Configura diretório de logs
        if log_dir is None:
            # Diretório padrão: junto ao controller
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(script_dir, "logs")
        
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Nome do arquivo com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"trajectory_{timestamp}.csv")
        
        # Tenta obter referência ao próprio robô (precisa ser Supervisor)
        self.robot_node = None
        self.translation_field = None
        self.rotation_field = None
        
        try:
            # Tenta obter o nó do robô pelo DEF ou procurando
            self.robot_node = robot.getSelf()
            if self.robot_node:
                self.translation_field = self.robot_node.getField("translation")
                self.rotation_field = self.robot_node.getField("rotation")
                print(f"📍 Coordinate Logger: Supervisor mode ativo")
            else:
                print(f"⚠️ Coordinate Logger: Robô não é Supervisor - posição estimada")
        except Exception as e:
            print(f"⚠️ Coordinate Logger: Erro ao obter nó do robô: {e}")
        
        # Inicializa arquivo CSV
        self._init_csv()
        
        # Contador de entradas
        self.entry_count = 0
        self.last_position = None
        
        print(f"📝 Coordinate Logger iniciado: {self.log_file}")
    
    def _init_csv(self):
        """Cria arquivo CSV com cabeçalho"""
        headers = [
            "timestamp",
            "frame",
            "x",
            "y", 
            "z",
            "rotation_angle",
            "phase",
            "direction",
            "speed",
            "stones_detected",
            "tree_visible",
            "tree_position",
            "left_blocked",
            "center_blocked",
            "right_blocked",
            "reason"
        ]
        
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def get_robot_position(self) -> tuple:
        """
        Obtém posição atual do robô
        
        Returns:
            (x, y, z, rotation_angle) ou (None, None, None, None) se não disponível
        """
        if self.translation_field and self.rotation_field:
            try:
                pos = self.translation_field.getSFVec3f()
                rot = self.rotation_field.getSFRotation()
                # rot é [axis_x, axis_y, axis_z, angle]
                return (pos[0], pos[1], pos[2], rot[3])
            except Exception as e:
                print(f"⚠️ Erro ao obter posição: {e}")
        
        return (None, None, None, None)
    
    def log(self, frame: int, phase: str, direction: str, speed: str,
            stones_detected: int = 0, tree_visible: bool = False,
            tree_position: str = None, left_blocked: bool = False,
            center_blocked: bool = False, right_blocked: bool = False,
            reason: str = ""):
        """
        Registra uma entrada no log
        
        Args:
            frame: Número do frame atual
            phase: Fase de navegação (SCANNING, NAVIGATING, etc.)
            direction: Direção comandada (N, S, E, W, NE, etc.)
            speed: Velocidade (SLOW, MEDIUM, FAST)
            stones_detected: Número de pedras detectadas
            tree_visible: Se a árvore está visível
            tree_position: Posição da árvore (left, center, right)
            left_blocked: Se zona esquerda está bloqueada
            center_blocked: Se zona central está bloqueada
            right_blocked: Se zona direita está bloqueada
            reason: Razão da decisão
        """
        x, y, z, rotation = self.get_robot_position()
        
        # Timestamp formatado
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        row = [
            timestamp,
            frame,
            f"{x:.4f}" if x is not None else "N/A",
            f"{y:.4f}" if y is not None else "N/A",
            f"{z:.4f}" if z is not None else "N/A",
            f"{rotation:.4f}" if rotation is not None else "N/A",
            phase,
            direction,
            speed,
            stones_detected,
            tree_visible,
            tree_position or "N/A",
            left_blocked,
            center_blocked,
            right_blocked,
            reason
        ]
        
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        self.entry_count += 1
        
        # Atualiza última posição conhecida
        if x is not None:
            self.last_position = (x, y, z)
    
    def log_event(self, frame: int, event_type: str, details: str):
        """
        Registra um evento especial (chegada, colisão, etc.)
        """
        x, y, z, rotation = self.get_robot_position()
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        row = [
            timestamp,
            frame,
            f"{x:.4f}" if x is not None else "N/A",
            f"{y:.4f}" if y is not None else "N/A",
            f"{z:.4f}" if z is not None else "N/A",
            f"{rotation:.4f}" if rotation is not None else "N/A",
            f"EVENT:{event_type}",
            "N/A",
            "N/A",
            0,
            False,
            "N/A",
            False,
            False,
            False,
            details
        ]
        
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def get_summary(self) -> dict:
        """
        Retorna resumo da trajetória
        """
        return {
            "total_entries": self.entry_count,
            "log_file": self.log_file,
            "last_position": self.last_position
        }
    
    def close(self):
        """
        Finaliza o logger e imprime resumo
        """
        summary = self.get_summary()
        print(f"\n📊 Coordinate Logger - Resumo:")
        print(f"   └─ Total de entradas: {summary['total_entries']}")
        print(f"   └─ Arquivo: {summary['log_file']}")
        if summary['last_position']:
            x, y, z = summary['last_position']
            print(f"   └─ Última posição: ({x:.2f}, {y:.2f}, {z:.2f})")


def generate_trajectory_visualization(csv_file: str, output_file: str = None):
    """
    Gera uma visualização ASCII da trajetória a partir do CSV
    
    Args:
        csv_file: Caminho para o arquivo CSV
        output_file: Arquivo de saída (opcional, imprime no console se None)
    """
    import csv
    
    positions = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row['x']) if row['x'] != 'N/A' else None
                y = float(row['y']) if row['y'] != 'N/A' else None
                phase = row['phase']
                if x is not None and y is not None:
                    positions.append((x, y, phase))
            except (ValueError, KeyError):
                continue
    
    if not positions:
        print("Nenhuma posição válida encontrada no arquivo")
        return
    
    # Encontra limites
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Cria grade ASCII (50x30 caracteres)
    width, height = 60, 30
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Bordas
    for i in range(width):
        grid[0][i] = '#'
        grid[height-1][i] = '#'
    for i in range(height):
        grid[i][0] = '#'
        grid[i][width-1] = '#'
    
    # Normaliza e plota posições
    def normalize(x, y):
        nx = int((x - min_x) / (max_x - min_x + 0.001) * (width - 4)) + 2
        ny = int((y - min_y) / (max_y - min_y + 0.001) * (height - 4)) + 2
        return min(max(nx, 1), width-2), min(max(ny, 1), height-2)
    
    # Plota trajetória
    phase_chars = {
        "SCANNING": '.',
        "NAVIGATING": 'o',
        "AVOIDING": 'x',
        "REPOSITIONING": '+',
        "ARRIVED": '*'
    }
    
    for i, (x, y, phase) in enumerate(positions):
        nx, ny = normalize(x, y)
        char = phase_chars.get(phase, '?')
        
        # Marca início e fim especialmente
        if i == 0:
            char = 'S'  # Start
        elif i == len(positions) - 1:
            char = 'E'  # End
        
        grid[ny][nx] = char
    
    # Gera string
    result = []
    result.append(f"\n📍 Visualização da Trajetória")
    result.append(f"   Arquivo: {csv_file}")
    result.append(f"   Posições: {len(positions)}")
    result.append(f"   X: [{min_x:.1f}, {max_x:.1f}] | Y: [{min_y:.1f}, {max_y:.1f}]")
    result.append("")
    result.append("Legenda: S=Início E=Fim .=Scanning o=Navigating x=Avoiding +=Repositioning *=Arrived")
    result.append("")
    
    for row in reversed(grid):  # Inverte Y para visualização correta
        result.append(''.join(row))
    
    output = '\n'.join(result)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Visualização salva em: {output_file}")
    else:
        print(output)
    
    return output


if __name__ == "__main__":
    # Teste: gera visualização de um CSV existente
    import sys
    
    if len(sys.argv) > 1:
        generate_trajectory_visualization(sys.argv[1])
    else:
        print("Uso: python coordinate_logger.py <arquivo_csv>")
        print("Exemplo: python coordinate_logger.py logs/trajectory_20250101_120000.csv")
