"""
Comunicación bidireccional (full-duplex sobre el enlace óptico half-duplex):
cada computador transmite su propio mensaje en loop continuo mientras recibe
el del otro, de forma simultánea.

Ejecutar en cada computador (con su propio TEXTO):
    python bidireccional.py

Controles:
    [ESPACIO]            inicia el envío del propio mensaje
    [R] / click derecho  reinicia el receptor para recibir otro mensaje
                         (solo funciona una vez que la recepción terminó)
    [D]                  alterna la vista de depuración de fiduciales
    [Q]                  salir
"""
import cv2
import numpy as np
from receptor.capture import find_fiducials_robust
from receptor.rx_state import Receiver, decode_frame
from transmisor.tx_state import Transmitter
from common.config import *


def display_tx(frame_bgr, screen_w, screen_h):
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


def run(text: str):
    screen_w, screen_h = 1920, 1080

    cv2.namedWindow("Transmisor", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Transmisor", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow("Receptor")

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    with open("mensaje_enviado.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("\nPreparando transmisor...")
    tx = Transmitter(text)
    rx = Receiver()

    debug_mode = False
    restart_requested = [False]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            restart_requested[0] = True

    cv2.setMouseCallback("Receptor", on_mouse)

    print("\nTransmisor: preámbulo activo. Alinea la cámara del receptor remoto.")
    print("Receptor  : esperando preámbulo del transmisor remoto.")
    print("[ESPACIO] iniciar envío  |  [D] debug  |  "
          "[R]/click derecho reiniciar receptor  |  [Q] salir\n")

    while True:
        # ── TX: mostrar el frame correspondiente al instante actual ──
        display_tx(tx.current_frame(), screen_w, screen_h)

        # ── RX: leer y procesar un frame de la cámara ────────────────
        ret, frame = cap.read()
        if ret:
            gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, method = find_fiducials_robust(gray, bgr=frame,
                                                    debug=debug_mode)
            disp = frame.copy()

            status_color = {
                "WAITING":   (0, 0,   255),
                "SYNCED":    (0, 165, 255),
                "RECEIVING": (0, 255,   0),
                "DONE":      (255, 255, 0),
            }.get(rx.state, (255, 255, 255))

            cv2.putText(disp, f"Estado: {rx.state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            if corners is not None:
                pts = np.array([corners[0], corners[1],
                                corners[3], corners[2]], dtype=np.int32)
                cv2.polylines(disp, [pts], True, (0, 255, 0), 2)
                cv2.putText(disp, f"[{method}]", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if rx.total_frames is not None:
                cv2.putText(disp,
                            f"Frames: {len(rx.received)}/{rx.total_frames}  "
                            f"CRC errors: {rx.crc_errors}",
                            (10, disp.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            if rx.state == "DONE":
                cv2.putText(disp,
                            "Recepcion completa - [R]/click derecho para recibir otro",
                            (10, disp.shape[0] - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 1)

            cv2.imshow("Receptor", disp)

            if rx.state != "DONE":
                if rx.check_timeout():
                    print(f"\n⚠ Timeout ({TIMEOUT_SECONDS}s sin frames nuevos)")
                    print(f"  Frames recibidos: {len(rx.received)}/{rx.total_frames}")
                elif corners is not None:
                    symbols, threshold = decode_frame(gray, corners, bgr=frame)
                    if symbols is not None:
                        rx.process(symbols, threshold)

        # ── Teclado ────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            tx.start()
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug: {'ON' if debug_mode else 'OFF'}")

        if (key == ord('r') or restart_requested[0]) and rx.state == "DONE":
            print("\n↺ Reinicio manual del receptor")
            rx.restart()
        restart_requested[0] = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    TEXTO = (
        "Universidad de Antioquia - Comunicaciones Digitales 2026. "
        "Prueba de comunicacion bidireccional entre dos equipos."
    )
    print(f"Caracteres a enviar: {len(TEXTO)}")
    run(TEXTO)
