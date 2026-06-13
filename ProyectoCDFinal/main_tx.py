"""
Transmisor multi-frame.
Ejecutar: python main_tx.py
"""
import cv2
import numpy as np
import time
from transmisor.encoder import text_to_bits
from transmisor.frame_builder import build_frame, get_data_cells
from common.protocol import build_packet
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
    header_syms            = 80
    payload_syms           = total_cells - header_syms
    payload_bits_per_frame = payload_syms // 2

    all_bits = text_to_bits(text)
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
        symbols = build_packet(seq, total, chunk)
        if len(symbols) < total_cells:
            symbols = np.pad(symbols, (0, total_cells - len(symbols)),
                             constant_values=128)
        frames.append(build_frame(symbols[:total_cells], seq, total))

    return frames


def show_frame(frame_bgr, screen_w, screen_h):
    scale  = min(screen_w / FRAME_W, screen_h / FRAME_H)
    new_w  = int(FRAME_W * scale)
    new_h  = int(FRAME_H * scale)
    scaled = cv2.resize(frame_bgr, (new_w, new_h),
                        interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x_off  = (screen_w - new_w) // 2
    y_off  = (screen_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = scaled
    cv2.imshow("Transmisor", canvas)
    cv2.waitKey(1)


def transmit(text: str):
    screen_w, screen_h = 1920, 1080

    cv2.namedWindow("Transmisor", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Transmisor", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    print("\nPreparando frames...")
    data_frames = text_to_frames(text)
    preamble    = build_preamble_frame()
    end_frame   = build_end_frame()

    print("\nPreámbulo activo. Alinea la cámara del receptor.")
    print("[ESPACIO] iniciar  |  [Q] cancelar\n")

    while True:
        show_frame(preamble, screen_w, screen_h)
        key = cv2.waitKey(100) & 0xFF
        if key == ord(' '):
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return

    # Preámbulo de sincronización
    print("Enviando preámbulo...")
    for i in range(PREAMBLE_FRAMES):
        show_frame(preamble, screen_w, screen_h)
        time.sleep(FRAME_DURATION)
        print(f"  Preámbulo {i+1}/{PREAMBLE_FRAMES}")

    # ── Tiempo inicia aquí ──
    start = time.time()

    # Frames de datos
    print("Transmitiendo datos...")
    for seq, frame in enumerate(data_frames):
        show_frame(frame, screen_w, screen_h)
        time.sleep(FRAME_DURATION)
        print(f"  ▶ Frame {seq+1}/{len(data_frames)}")

    # Repetir últimos 2 frames 3 veces para asegurar recepción
    print("  Repitiendo últimos frames...")
    for frame in data_frames[-2:]:
        for _ in range(3):
            show_frame(frame, screen_w, screen_h)
            time.sleep(FRAME_DURATION)

    # Pausa antes del fin
    time.sleep(FRAME_DURATION * 2)

    # Frame de fin — muchas repeticiones
    print("  Enviando fin...")
    for _ in range(6):
        show_frame(end_frame, screen_w, screen_h)
        time.sleep(FRAME_DURATION)

    elapsed = time.time() - start
    print(f"\n{'='*40}")
    print(f"Transmisión completada")
    print(f"Frames          : {len(data_frames)}")
    print(f"Tiempo (datos)  : {elapsed:.2f}s")
    print(f"Throughput      : {len(text)*8/elapsed:.0f} bps")
    print(f"{'='*40}")
    print("[Q] para cerrar")

    while True:
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    TEXTO = (
        "Universidad de Antioquia - Comunicaciones Digitales 2026. "
        "Este es el proyecto final del semestre. El objetivo es disenar "
        "e implementar un modem optico espacio-temporal half-duplex capaz "
        "de transmitir un archivo de texto de 500 caracteres de la pantalla "
        "de un computador a la camara web de otro computador en 10 segundos "
        "o menos, sin conexion de red entre los dos equipos. El canal "
        "introduce distorsiones geometricas, variaciones de iluminacion, "
        "desenfoque y ruido del sensor."
    )
    print(f"Caracteres: {len(TEXTO)}")
    transmit(TEXTO)