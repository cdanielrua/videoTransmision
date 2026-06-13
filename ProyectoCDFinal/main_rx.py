"""
Receptor multi-frame: captura secuencia de frames y reconstruye el texto.
Ejecutar: python main_rx.py
"""
import cv2
import numpy as np
import time
from receptor.capture import find_fiducials_robust, rectify_frame, extract_symbols
from receptor.decoder import calibrate_threshold, symbols_to_bits_ook, \
                             manchester_decode, bits_to_text
from common.protocol import parse_packet
from common.config import *


def decode_frame(gray, corners):
    """Rectifica y decodifica un frame. Retorna símbolos crudos."""
    rectified = rectify_frame(gray, corners)
    if rectified is None:
        return None, None
    threshold = calibrate_threshold(rectified)
    symbols   = extract_symbols(rectified)
    return symbols, threshold


def is_preamble(symbols, threshold):
    """
    Detecta si el frame actual es un preámbulo
    (patrón alternado 255/0 en las celdas de datos).
    """
    bits = (symbols >= threshold).astype(np.uint8)
    expected = np.array([1 if i % 2 == 0 else 0
                         for i in range(len(bits))], dtype=np.uint8)
    match = np.mean(bits == expected)
    return match > 0.85  # 85% de coincidencia


def is_end_frame(symbols, threshold):
    """Detecta frame de fin (todas las celdas ~128 gris)."""
    return np.mean(np.abs(symbols.astype(int) - 128) < 40) > 0.80


def receive():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    print("Receptor multi-frame iniciado.")
    print("Esperando preámbulo del transmisor...")
    print("[Q] salir en cualquier momento\n")

    state         = "WAITING"   # WAITING → SYNCED → RECEIVING → DONE
    received      = {}          # seq → payload_bits
    total_frames  = None
    last_seq      = -1
    crc_errors    = 0
    start_time    = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, method = find_fiducials_robust(gray)
        display = frame.copy()

        # --- HUD ---
        status_color = {
            "WAITING":   (0, 0, 255),
            "SYNCED":    (0, 165, 255),
            "RECEIVING": (0, 255, 0),
            "DONE":      (255, 255, 0),
        }.get(state, (255,255,255))

        cv2.putText(display, f"Estado: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        if corners is not None:
            # Polígono verde sobre la pantalla detectada
            pts = np.array([corners[0], corners[1],
                            corners[3], corners[2]], dtype=np.int32)
            cv2.polylines(display, [pts], True, (0,255,0), 2)
            cv2.putText(display, f"[{method}]", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if total_frames is not None:
            recv_count = len(received)
            cv2.putText(display,
                        f"Frames: {recv_count}/{total_frames}  "
                        f"CRC errors: {crc_errors}",
                        (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

        cv2.imshow("Receptor", display)

        # --- Máquina de estados ---
        if corners is not None and state != "DONE":
            symbols, threshold = decode_frame(gray, corners)

            if symbols is None:
                pass

            elif state == "WAITING":
                if is_preamble(symbols, threshold):
                    print("✓ Preámbulo detectado — sincronizado")
                    state = "SYNCED"

            elif state == "SYNCED":
                if not is_preamble(symbols, threshold):
                    # Primer frame de datos
                    state     = "RECEIVING"
                    
                    print("✓ Inicio de datos detectado")

# En receive(), dentro de la máquina de estados, cambia esto:

            elif state == "SYNCED":
                if not is_preamble(symbols, threshold):
                    state = "RECEIVING"
                    print("✓ Inicio de datos detectado")
                    # start_time NO va aquí

            if state == "RECEIVING":
                if is_end_frame(symbols, threshold):
                    state   = "DONE"
                    elapsed = time.time() - start_time
                    print(f"\n✓ Frame de fin detectado ({elapsed:.2f}s)")
                    _reconstruct(received, total_frames, elapsed)
                else:
                    result = parse_packet(symbols, threshold)
                    if result is not None:
                        seq, total, payload, crc_ok = result
                        total_frames = total

                        # NUEVO: iniciar tiempo con el primer frame válido
                        if start_time is None:
                            start_time = time.time()
                            print("  ⏱ Tiempo iniciado")

                        if not crc_ok:
                            crc_errors += 1
                            print(f"  ✗ Frame {seq}: CRC error")
                        elif seq not in received:
                            received[seq] = payload
                            print(f"  ✓ Frame {seq+1}/{total}  "
                                  f"({len(payload)} bits)")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def _reconstruct(received: dict, total_frames: int, elapsed: float):
    """Reconstruye el texto a partir de los frames recibidos."""
    if total_frames is None:
        print("No se recibió ningún frame.")
        return

    print(f"\n{'='*50}")
    print(f"Frames recibidos: {len(received)}/{total_frames}")
    print(f"Frames perdidos:  {total_frames - len(received)}")

    # Concatenar payloads en orden
    all_bits = []
    for seq in range(total_frames):
        if seq in received:
            all_bits.extend(received[seq])
        else:
            print(f"  ⚠ Frame {seq} faltante — rellenando con ceros")
            all_bits.extend([0] * 8)  # placeholder

    all_bits = np.array(all_bits, dtype=np.uint8)
    text     = bits_to_text(all_bits)

    print(f"\nTexto reconstruido ({len(text)} caracteres):")
    print(f"{'='*50}")
    print(text)
    print(f"{'='*50}")
    print(f"Tiempo de transmisión: {elapsed:.2f}s")

    # Guardar a archivo
    with open("mensaje_recibido.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Guardado en: mensaje_recibido.txt")


if __name__ == '__main__':
    receive()