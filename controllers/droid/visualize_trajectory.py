"""
Visualizador de Trajetória do Robô
Gera mapas e gráficos a partir dos logs de coordenadas

Uso:
    python visualize_trajectory.py                    # Usa o log mais recente
    python visualize_trajectory.py trajectory_xxx.csv # Usa um log específico
"""
import os
import sys
import csv
import math
from datetime import datetime


def find_latest_log(log_dir: str = "logs") -> str:
    """Encontra o arquivo de log mais recente"""
    if not os.path.exists(log_dir):
        return None
    
    csv_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
    if not csv_files:
        return None
    
    # Ordena por data de modificação
    csv_files.sort(key=lambda f: os.path.getmtime(os.path.join(log_dir, f)), reverse=True)
    return os.path.join(log_dir, csv_files[0])


def load_trajectory(csv_file: str) -> list:
    """Carrega dados da trajetória"""
    data = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                entry = {
                    'timestamp': row['timestamp'],
                    'frame': int(row['frame']),
                    'x': float(row['x']) if row['x'] != 'N/A' else None,
                    'y': float(row['y']) if row['y'] != 'N/A' else None,
                    'z': float(row['z']) if row['z'] != 'N/A' else None,
                    'rotation': float(row['rotation_angle']) if row['rotation_angle'] != 'N/A' else None,
                    'phase': row['phase'],
                    'direction': row['direction'],
                    'speed': row['speed'],
                    'stones': int(row['stones_detected']),
                    'tree_visible': row['tree_visible'] == 'True',
                    'tree_pos': row['tree_position'],
                    'left_blocked': row['left_blocked'] == 'True',
                    'center_blocked': row['center_blocked'] == 'True',
                    'right_blocked': row['right_blocked'] == 'True',
                    'reason': row['reason']
                }
                data.append(entry)
            except (ValueError, KeyError) as e:
                continue
    
    return data


def calculate_stats(data: list) -> dict:
    """Calcula estatísticas da trajetória"""
    positions = [(d['x'], d['y']) for d in data if d['x'] is not None]
    
    if not positions:
        return {}
    
    # Distância total percorrida
    total_distance = 0
    for i in range(1, len(positions)):
        x1, y1 = positions[i-1]
        x2, y2 = positions[i]
        total_distance += math.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    # Distância em linha reta (início -> fim)
    if len(positions) >= 2:
        straight_distance = math.sqrt(
            (positions[-1][0] - positions[0][0])**2 +
            (positions[-1][1] - positions[0][1])**2
        )
    else:
        straight_distance = 0
    
    # Contagem de fases
    phases = {}
    for d in data:
        phase = d['phase']
        phases[phase] = phases.get(phase, 0) + 1
    
    # Contagem de direções
    directions = {}
    for d in data:
        direction = d['direction']
        directions[direction] = directions.get(direction, 0) + 1
    
    return {
        'total_entries': len(data),
        'total_distance': total_distance,
        'straight_distance': straight_distance,
        'efficiency': (straight_distance / total_distance * 100) if total_distance > 0 else 0,
        'start_pos': positions[0] if positions else None,
        'end_pos': positions[-1] if positions else None,
        'phases': phases,
        'directions': directions,
        'frames': data[-1]['frame'] if data else 0
    }


def generate_ascii_map(data: list, width: int = 80, height: int = 35) -> str:
    """Gera mapa ASCII da trajetória"""
    positions = [(d['x'], d['y'], d['phase']) for d in data if d['x'] is not None]
    
    if not positions:
        return "Nenhuma posição válida encontrada"
    
    # Encontra limites
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Adiciona margem
    margin = 0.5
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin
    
    # Cria grade
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Bordas
    for i in range(width):
        grid[0][i] = '═'
        grid[height-1][i] = '═'
    for i in range(height):
        grid[i][0] = '║'
        grid[i][width-1] = '║'
    grid[0][0] = '╔'
    grid[0][width-1] = '╗'
    grid[height-1][0] = '╚'
    grid[height-1][width-1] = '╝'
    
    # Função para normalizar coordenadas
    def normalize(x, y):
        range_x = max_x - min_x if max_x != min_x else 1
        range_y = max_y - min_y if max_y != min_y else 1
        nx = int((x - min_x) / range_x * (width - 4)) + 2
        ny = int((y - min_y) / range_y * (height - 4)) + 2
        return min(max(nx, 1), width-2), min(max(ny, 1), height-2)
    
    # Caracteres por fase
    phase_chars = {
        "SCANNING": '·',
        "NAVIGATING": '○',
        "AVOIDING": '×',
        "REPOSITIONING": '+',
        "ARRIVED": '★'
    }
    
    # Plota trajetória
    for i, (x, y, phase) in enumerate(positions):
        nx, ny = normalize(x, y)
        
        # Determina caractere
        if i == 0:
            char = 'S'  # Start
        elif i == len(positions) - 1:
            char = 'E'  # End
        elif 'EVENT:' in phase:
            char = '!'
        else:
            char = phase_chars.get(phase, '?')
        
        grid[height - 1 - ny][nx] = char  # Inverte Y
    
    # Gera string
    lines = [''.join(row) for row in grid]
    
    return '\n'.join(lines)


def generate_report(csv_file: str) -> str:
    """Gera relatório completo da trajetória"""
    data = load_trajectory(csv_file)
    stats = calculate_stats(data)
    ascii_map = generate_ascii_map(data)
    
    report = []
    report.append("=" * 80)
    report.append("📊 RELATÓRIO DE TRAJETÓRIA DO ROBÔ")
    report.append("=" * 80)
    report.append(f"📁 Arquivo: {csv_file}")
    report.append(f"📅 Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("─" * 80)
    report.append("📈 ESTATÍSTICAS GERAIS")
    report.append("─" * 80)
    report.append(f"  Total de entradas:     {stats.get('total_entries', 0)}")
    report.append(f"  Total de frames:       {stats.get('frames', 0)}")
    report.append(f"  Distância percorrida:  {stats.get('total_distance', 0):.2f} m")
    report.append(f"  Distância em linha reta: {stats.get('straight_distance', 0):.2f} m")
    report.append(f"  Eficiência do caminho: {stats.get('efficiency', 0):.1f}%")
    report.append("")
    
    if stats.get('start_pos'):
        x, y = stats['start_pos']
        report.append(f"  Posição inicial:       ({x:.2f}, {y:.2f})")
    if stats.get('end_pos'):
        x, y = stats['end_pos']
        report.append(f"  Posição final:         ({x:.2f}, {y:.2f})")
    report.append("")
    
    report.append("─" * 80)
    report.append("🔄 FASES DE NAVEGAÇÃO")
    report.append("─" * 80)
    phases = stats.get('phases', {})
    phase_symbols = {
        "SCANNING": "🔍",
        "NAVIGATING": "🚀",
        "AVOIDING": "🔀",
        "REPOSITIONING": "📍",
        "ARRIVED": "🎯"
    }
    for phase, count in sorted(phases.items(), key=lambda x: -x[1]):
        symbol = phase_symbols.get(phase, "  ")
        report.append(f"  {symbol} {phase:20s}: {count:5d} ({count/stats['total_entries']*100:.1f}%)")
    report.append("")
    
    report.append("─" * 80)
    report.append("🧭 DIREÇÕES COMANDADAS")
    report.append("─" * 80)
    directions = stats.get('directions', {})
    dir_symbols = {
        "N": "↑", "S": "↓", "E": "→", "W": "←",
        "NE": "↗", "NW": "↖", "SE": "↘", "SW": "↙",
        "STOP": "⏸"
    }
    for direction, count in sorted(directions.items(), key=lambda x: -x[1]):
        symbol = dir_symbols.get(direction, " ")
        report.append(f"  {symbol} {direction:5s}: {count:5d} ({count/stats['total_entries']*100:.1f}%)")
    report.append("")
    
    report.append("─" * 80)
    report.append("🗺️  MAPA DA TRAJETÓRIA")
    report.append("─" * 80)
    report.append("")
    report.append("Legenda: S=Início  E=Fim  ·=Scanning  ○=Navigating  ×=Avoiding  +=Repositioning  ★=Arrived")
    report.append("")
    report.append(ascii_map)
    report.append("")
    
    report.append("=" * 80)
    report.append("FIM DO RELATÓRIO")
    report.append("=" * 80)
    
    return '\n'.join(report)


def main():
    # Diretório do script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    
    # Determina arquivo a usar
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        if not os.path.isabs(csv_file):
            csv_file = os.path.join(log_dir, csv_file)
    else:
        csv_file = find_latest_log(log_dir)
    
    if not csv_file or not os.path.exists(csv_file):
        print("❌ Nenhum arquivo de log encontrado!")
        print(f"   Procurado em: {log_dir}")
        print("\nUso:")
        print("  python visualize_trajectory.py                    # Usa o log mais recente")
        print("  python visualize_trajectory.py trajectory_xxx.csv # Usa um log específico")
        return
    
    # Gera relatório
    report = generate_report(csv_file)
    print(report)
    
    # Salva relatório
    report_file = csv_file.replace('.csv', '_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n💾 Relatório salvo em: {report_file}")


if __name__ == "__main__":
    main()
