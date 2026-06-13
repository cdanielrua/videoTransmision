import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import *

def build_fiducial_marker(size_cells: int) -> np.ndarray:
    s = size_cells * CELL_SIZE
    marker = np.zeros((s, s), dtype=np.uint8)
    c = CELL_SIZE
    marker[c:s-c, c:s-c] = 255      # anillo blanco
    c2 = 2 * CELL_SIZE
    marker[c2:s-c2, c2:s-c2] = 0    # centro negro
    return marker

def get_data_cells():
    """Retorna lista de (row, col) disponibles para datos."""
    fid = FIDUCIAL_SIZE  # 5 celdas
    cells = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            in_tl = row < fid and col < fid
            in_tr = row < fid and col >= GRID_COLS - fid
            in_bl = row >= GRID_ROWS - fid and col < fid
            in_br = row >= GRID_ROWS - fid and col >= GRID_COLS - fid
            is_pilot = (row == PILOT_ROW)
            
            if not any([in_tl, in_tr, in_bl, in_br, is_pilot]):
                cells.append((row, col))
    return cells

def build_frame(symbols: np.ndarray, frame_number: int = 0,
                total_frames: int = 1) -> np.ndarray:
    # Canvas negro en BGR
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    
    # Fondo gris medio solo en zona de datos
    frame[:, :] = (64, 64, 64)

    # --- PRIMERO los datos ---
    data_cells = get_data_cells()
    n = min(len(symbols), len(data_cells))
    for i in range(n):
        row, col = data_cells[i]
        y, x = row * CELL_SIZE, col * CELL_SIZE
        val = int(symbols[i])
        frame[y:y+CELL_SIZE, x:x+CELL_SIZE] = (val, val, val)

    # --- Fila de pilotos blanco/negro ---
    for col in range(GRID_COLS):
        val = 255 if col % 2 == 0 else 0
        y = PILOT_ROW * CELL_SIZE
        x = col * CELL_SIZE
        frame[y:y+CELL_SIZE, x:x+CELL_SIZE] = (val, val, val)

    # --- ÚLTIMO: fiduciales en verde brillante ---
    fid_px = FIDUCIAL_SIZE * CELL_SIZE
    fid    = _build_fiducial_color(FIDUCIAL_SIZE)
    frame[0:fid_px,            0:fid_px           ] = fid  # TL
    frame[0:fid_px,            FRAME_W-fid_px:FRAME_W] = fid  # TR
    frame[FRAME_H-fid_px:FRAME_H, 0:fid_px        ] = fid  # BL
    frame[FRAME_H-fid_px:FRAME_H, FRAME_W-fid_px:FRAME_W] = fid  # BR

    return frame


def _build_fiducial_color(size_cells: int) -> np.ndarray:
    """
    Fiducial en color BGR:
    - Borde exterior: negro
    - Anillo medio:   verde brillante (0, 255, 0)
    - Centro:         negro
    Inconfundible con los datos en escala de grises.
    """
    s  = size_cells * CELL_SIZE
    c  = CELL_SIZE
    c2 = 2 * CELL_SIZE

    marker = np.zeros((s, s, 3), dtype=np.uint8)        # todo negro
    marker[c:s-c,   c:s-c  ] = (0, 255, 0)              # anillo verde
    marker[c2:s-c2, c2:s-c2] = (0,   0, 0)              # centro negro
    return marker

def cells_capacity():
    return len(get_data_cells())

if __name__ == '__main__':
    from transmisor.encoder import text_to_bits, manchester_encode, bits_to_ook_symbols
    
    texto = "Hola mundo! Este es el primer frame del modem optico."
    bits = manchester_encode(text_to_bits(texto))
    symbols = bits_to_ook_symbols(bits)
    
    cap = cells_capacity()
    print(f"Capacidad del frame: {cap} celdas")
    print(f"Símbolos generados: {len(symbols)}")
    print(f"Celdas usadas: {min(len(symbols), cap)}/{cap}")
    
    frame = build_frame(symbols, 1, 1)
    cv2.imshow("Frame corregido", frame)
    cv2.imwrite("frame_v2.png", frame)
    print("Guardado: frame_v2.png")
    cv2.waitKey(0)
    cv2.destroyAllWindows()