GRID_COLS = 40
GRID_ROWS = 30
CELL_SIZE = 20
# --- Multi-frame ---
PREAMBLE_FRAMES = 2      # frames de sincronización al inicio
FRAME_DURATION  = 0.200   # segundos por frame (150ms)
END_FRAMES      = 6       # repeticiones del frame de fin

# --- Receptor ---
TIMEOUT_SECONDS = 3.0     # tiempo de gracia sin frames nuevos antes de abortar


FRAME_W = GRID_COLS * CELL_SIZE   # 800px
FRAME_H = GRID_ROWS * CELL_SIZE   # 600px

MODULATION = 'OOK'
FIDUCIAL_SIZE = 5   # 5×5 celdas por marcador
PILOT_ROW = 3       # fila 3 reservada para pilotos (después de fiduciales)

