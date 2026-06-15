"""
Lógica de transmisión reutilizable: construcción de tramas especiales y
una máquina de estados no bloqueante para transmisión continua en loop.

Usado por main_tx.py y bidireccional.py.
"""
import time
import numpy as np
from transmisor.encoder import text_to_bits
from transmisor.frame_builder import build_frame, get_data_cells
from common.protocol import build_packet, payload_capacity
from common.config import *


def build_preamble_frame() -> np.ndarray:
    data_cells = get_data_cells()
    symbols = np.array(
        [255 if i % 2 == 0 else 0 for i in range(len(data_cells))],
        dtype=np.uint8
    )
    return build_frame(symbols, frame_number=-1, total_frames=-1)


def build_end_frame() -> np.ndarray:
    data_cells = get_data_cells()
    symbols = np.full(len(data_cells), 128, dtype=np.uint8)
    return build_frame(symbols, frame_number=-2, total_frames=-2)


def text_to_frames(text: str):
    data_cells             = get_data_cells()
    total_cells            = len(data_cells)
    payload_bits_per_frame = payload_capacity(total_cells)

    # Terminador nulo (0x00): marca el fin real del mensaje para que el
    # receptor pueda distinguir datos de relleno en el último frame.
    all_bits = np.concatenate([text_to_bits(text),
                                np.zeros(8, dtype=np.uint8)])
    print(f"Texto           : {len(text)} caracteres → {len(all_bits)} bits")
    print(f"Capacidad/frame : {payload_bits_per_frame} bits")

    chunks = []
    for i in range(0, len(all_bits), payload_bits_per_frame):
        chunks.append(all_bits[i:i + payload_bits_per_frame])

    total = len(chunks)
    print(f"Frames necesarios: {total}")
    print(f"Tiempo estimado  : {total * FRAME_DURATION:.2f}s")

    frames = []
    for seq, chunk in enumerate(chunks):
        if len(chunk) < payload_bits_per_frame:
            chunk = np.pad(chunk, (0, payload_bits_per_frame - len(chunk)),
                           constant_values=0)
        symbols = build_packet(seq, total, chunk)
        if len(symbols) < total_cells:
            symbols = np.pad(symbols, (0, total_cells - len(symbols)),
                             constant_values=128)
        frames.append(build_frame(symbols[:total_cells], seq, total))

    return frames


class Transmitter:
    """
    Máquina de estados no bloqueante para transmisión continua en loop.

    `current_frame()` debe llamarse una vez por iteración del loop
    principal; internamente avanza al siguiente frame según el tiempo
    transcurrido (FRAME_DURATION por frame). Útil cuando el mismo loop
    también necesita atender la recepción (bidireccional.py).
    """
    PREVIEW  = 'PREVIEW'   # mostrando preámbulo, esperando inicio (SPACE)
    PREAMBLE = 'PREAMBLE'
    DATA     = 'DATA'
    END      = 'END'

    def __init__(self, text: str):
        self.text        = text
        self.data_frames = text_to_frames(text)
        self.preamble    = build_preamble_frame()
        self.end_frame   = build_end_frame()
        self.state       = self.PREVIEW
        self.index       = 0
        self.last_switch = None
        self.started     = False

    def start(self):
        """Inicia el ciclo de transmisión continua (tecla SPACE)."""
        if self.started:
            return
        self.started    = True
        self.state       = self.PREAMBLE
        self.index       = 0
        self.last_switch = time.time()

    def current_frame(self) -> np.ndarray:
        """Retorna el frame BGR a mostrar y avanza el estado según el tiempo."""
        if self.state == self.PREVIEW:
            return self.preamble

        now = time.time()
        if self.last_switch is None:
            self.last_switch = now

        if now - self.last_switch >= FRAME_DURATION:
            self.last_switch = now
            self._advance()

        if self.state == self.PREAMBLE:
            return self.preamble
        elif self.state == self.DATA:
            return self.data_frames[self.index]
        else:
            return self.end_frame

    def _advance(self):
        if self.state == self.PREAMBLE:
            self.index += 1
            if self.index >= PREAMBLE_FRAMES:
                self.state = self.DATA
                self.index = 0
        elif self.state == self.DATA:
            self.index += 1
            if self.index >= len(self.data_frames):
                self.state = self.END
                self.index = 0
        elif self.state == self.END:
            self.index += 1
            if self.index >= END_FRAMES:
                self.state = self.PREAMBLE
                self.index = 0
