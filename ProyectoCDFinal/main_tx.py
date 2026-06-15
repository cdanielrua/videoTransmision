"""
Transmisor multi-frame.
Ejecutar: python main_tx.py
"""
import cv2
import numpy as np
import time
from transmisor.tx_state import (build_preamble_frame, build_end_frame,
                                  text_to_frames)
from common.config import *


def display(frame_bgr, screen_w, screen_h, wait_ms):
    """Muestra el frame y retorna la tecla presionada (o 255 si ninguna)."""
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
    return cv2.waitKey(wait_ms) & 0xFF


def _control(key):
    """Traduce una tecla a una señal de control ('quit'/'restart'/None)."""
    if key == ord('q'):
        return 'quit'
    if key == ord('r'):
        return 'restart'
    return None


def send_cycle(data_frames, preamble, end_frame, screen_w, screen_h, text):
    """Envía un ciclo completo: preámbulo + datos + fin.

    Retorna 'quit', 'restart' o 'done'.
    """
    frame_ms = int(FRAME_DURATION * 1000)

    # Preámbulo de sincronización
    print("Enviando preámbulo...")
    for i in range(PREAMBLE_FRAMES):
        ctrl = _control(display(preamble, screen_w, screen_h, frame_ms))
        if ctrl:
            return ctrl
        print(f"  Preámbulo {i+1}/{PREAMBLE_FRAMES}")

    # ── Tiempo inicia aquí ──
    start = time.time()

    # Frames de datos
    print("Transmitiendo datos...")
    for seq, frame in enumerate(data_frames):
        ctrl = _control(display(frame, screen_w, screen_h, frame_ms))
        if ctrl:
            return ctrl
        print(f"  ▶ Frame {seq+1}/{len(data_frames)}")

    # Frame de fin — muchas repeticiones
    print("  Enviando fin...")
    for _ in range(END_FRAMES):
        ctrl = _control(display(end_frame, screen_w, screen_h, frame_ms))
        if ctrl:
            return ctrl

    elapsed = time.time() - start
    print(f"\n{'='*40}")
    print(f"Transmisión completada")
    print(f"Frames          : {len(data_frames)}")
    print(f"Tiempo (datos)  : {elapsed:.2f}s")
    print(f"Throughput      : {len(text)*8/elapsed:.0f} bps")
    print(f"{'='*40}")
    return 'done'


def transmit(text: str):
    screen_w, screen_h = 1920, 1080

    cv2.namedWindow("Transmisor", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Transmisor", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    with open("mensaje_enviado.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("\nPreparando frames...")
    data_frames = text_to_frames(text)
    preamble    = build_preamble_frame()
    end_frame   = build_end_frame()

    print("\nPreámbulo activo. Alinea la cámara del receptor.")
    print("[ESPACIO] iniciar  |  [Q] cancelar\n")

    while True:
        key = display(preamble, screen_w, screen_h, 100)
        if key == ord(' '):
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return

    print("\nTransmisión continua iniciada. [R] reiniciar  |  [Q] salir\n")

    # ── Loop continuo: reinicia automáticamente al terminar cada ciclo ──
    while True:
        result = send_cycle(data_frames, preamble, end_frame,
                            screen_w, screen_h, text)
        if result == 'quit':
            break
        if result == 'restart':
            print("\n↺ Reinicio solicitado — reenviando desde el preámbulo\n")

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