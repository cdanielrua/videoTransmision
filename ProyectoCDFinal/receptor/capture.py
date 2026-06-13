import cv2
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import *


# ─────────────────────────────────────────────
#  DETECCIÓN PRINCIPAL
# ─────────────────────────────────────────────

def find_fiducials_robust(gray: np.ndarray,
                          bgr:  np.ndarray = None,
                          debug: bool = False):
    """
    Intenta detectar los 4 fiduciales usando color verde (si se pasa bgr)
    o por contornos en escala de grises como fallback.
    Retorna (corners, method) o (None, 'no_encontrado').
    """
    # Método 1: detección por color verde (más robusto)
    if bgr is not None:
        corners = _find_by_green(bgr, debug)
        if corners is not None:
            return corners, "color_verde"

    # Método 2: contornos en escala de grises
    candidates = _get_candidates_gray(gray)

    if len(candidates) >= 4:
        candidates.sort(key=lambda x: x[2], reverse=True)
        top4    = [(x, y) for x, y, _ in candidates[:4]]
        corners = sort_corners(top4)
        if _validate_quad(corners, gray.shape):
            return corners, "contornos_4"

    if len(candidates) == 3:
        pts3   = [(x, y) for x, y, _ in candidates]
        fourth = _infer_fourth(pts3, gray.shape)
        if fourth is not None:
            corners = sort_corners(pts3 + [fourth])
            return corners, "geometria_3+1"

    return None, "no_encontrado"


# ─────────────────────────────────────────────
#  MÉTODO 1: DETECCIÓN POR COLOR VERDE
# ─────────────────────────────────────────────

def _find_by_green(bgr: np.ndarray, debug: bool = False):
    """
    Detecta los 4 fiduciales verdes en la imagen BGR.
    Filtra por color HSV, busca contornos cuadrados
    solo en las 4 esquinas de la imagen.
    """
    h, w = bgr.shape[:2]
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Rango de verde en HSV
    lower_green = np.array([40,  80,  80])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morfología para limpiar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    if debug:
        cv2.imshow("Mascara Verde", mask)

    # Zonas de búsqueda: 40% de cada esquina
    f = 0.40
    zones = [
        (0,       0,       int(w*f), int(h*f)),
        (int(w*(1-f)), 0,  w,        int(h*f)),
        (0,       int(h*(1-f)), int(w*f), h  ),
        (int(w*(1-f)), int(h*(1-f)), w,   h  ),
    ]

    centers = []
    for (x1, y1, x2, y2) in zones:
        roi_mask = mask[y1:y2, x1:x2]
        cnts, _  = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        # Tomar el contorno verde más grande en esta zona
        best = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < 100:
            continue

        M = cv2.moments(best)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00']) + x1
        cy = int(M['m01'] / M['m00']) + y1
        centers.append((cx, cy))

    if len(centers) == 4:
        corners = sort_corners(centers)
        if _validate_quad(corners, bgr.shape[:2]):
            return corners

    # Si solo encontramos 3, inferir el 4to
    if len(centers) == 3:
        fourth = _infer_fourth(centers, bgr.shape[:2])
        if fourth is not None:
            corners = sort_corners(centers + [fourth])
            return corners

    return None


# ─────────────────────────────────────────────
#  MÉTODO 2: CONTORNOS EN GRIS
# ─────────────────────────────────────────────

def _get_candidates_gray(gray: np.ndarray):
    """Busca candidatos por estructura concéntrica solo en esquinas."""
    h, w   = gray.shape
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    f      = 0.40
    zones  = [
        (0,       0,       int(w*f), int(h*f)),
        (int(w*(1-f)), 0,  w,        int(h*f)),
        (0,       int(h*(1-f)), int(w*f), h  ),
        (int(w*(1-f)), int(h*(1-f)), w,   h  ),
    ]
    candidates = []

    for block_size in [11, 21]:
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 2
        )
        cnts, hier = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if hier is None:
            continue
        hier = hier[0]

        for i, cnt in enumerate(cnts):
            area = cv2.contourArea(cnt)
            if area < 150 or area > w * h * 0.06:
                continue

            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            if len(approx) != 4:
                continue

            if hier[i][2] == -1:   # sin hijo → no es concéntrico
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            if not (0.6 < bw/bh < 1.6):
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Solo en zonas de esquina
            in_zone = any(
                zx1 <= cx <= zx2 and zy1 <= cy <= zy2
                for zx1, zy1, zx2, zy2 in zones
            )
            if not in_zone:
                continue

            is_dup = any(
                abs(cx-ex) < 20 and abs(cy-ey) < 20
                for ex, ey, _ in candidates
            )
            if not is_dup:
                candidates.append((cx, cy, area))

    return candidates


# ─────────────────────────────────────────────
#  GEOMETRÍA: INFERIR 4TO PUNTO
# ─────────────────────────────────────────────

def _infer_fourth(pts3, image_shape):
    """
    Dado 3 esquinas de un rectángulo, calcula la 4ta
    usando la propiedad del paralelogramo: D = A + C - B
    donde B es la esquina opuesta a D.
    Valida que el resultado sea geométricamente coherente.
    """
    img_h, img_w = image_shape[:2]
    diag         = np.sqrt(img_w**2 + img_h**2)
    best         = None
    best_score   = float('inf')

    for i in range(3):
        p_opp = np.array(pts3[i],   dtype=float)
        p_a   = np.array(pts3[(i+1) % 3], dtype=float)
        p_b   = np.array(pts3[(i+2) % 3], dtype=float)

        p4 = p_a + p_b - p_opp
        p4 = (int(p4[0]), int(p4[1]))

        # ── Validaciones ──────────────────────────────────────
        # 1. Dentro de la imagen
        margin = 5
        if not (margin < p4[0] < img_w - margin and
                margin < p4[1] < img_h - margin):
            continue

        # 2. No demasiado cerca de ningún punto existente
        dists = [np.linalg.norm(np.array(p4) - np.array(p))
                 for p in pts3]
        if min(dists) < 0.08 * diag:
            continue

        # 3. Lados opuestos similares (rectángulo)
        all4 = sort_corners(pts3 + [p4])
        tl, tr, bl, br = [np.array(p) for p in all4]
        top    = np.linalg.norm(tr - tl)
        bottom = np.linalg.norm(br - bl)
        left   = np.linalg.norm(bl - tl)
        right  = np.linalg.norm(br - tr)

        if 0 in [top, bottom, left, right]:
            continue
        if (max(top,   bottom) / min(top,   bottom) > 1.5 or
            max(left,  right)  / min(left,  right)  > 1.5):
            continue

        # 4. Área mínima razonable
        area = _quad_area(all4)
        if area < 0.04 * img_w * img_h:
            continue

        # 5. Rectangularidad
        score = _rect_score(all4)
        if score < best_score:
            best_score = score
            best       = p4

    # Solo aceptar si es suficientemente rectangular
    return best if best_score < 25.0 else None


def _validate_quad(corners, image_shape):
    """Valida que los 4 corners formen un cuadrilátero razonable."""
    img_h, img_w = image_shape[:2]
    tl, tr, bl, br = [np.array(p) for p in corners]

    top    = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left   = np.linalg.norm(bl - tl)
    right  = np.linalg.norm(br - tr)

    if 0 in [top, bottom, left, right]:
        return False

    # Lados opuestos similares
    if (max(top,  bottom) / min(top,  bottom) > 1.6 or
        max(left, right)  / min(left, right)  > 1.6):
        return False

    # Área mínima
    if _quad_area(corners) < 0.04 * img_w * img_h:
        return False

    return True


def _quad_area(pts4):
    """Área por fórmula del zapato."""
    pts  = np.array(pts4, dtype=np.float32)
    n    = len(pts)
    area = 0.0
    for i in range(n):
        j     = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def _rect_score(pts4):
    """Score de rectangularidad — menor es mejor."""
    corners = sort_corners(pts4)
    tl, tr, bl, br = corners

    def ang(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 90.0
        c = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        return abs(np.degrees(np.arccos(c)) - 90.0)

    tl, tr, bl, br = [np.array(p) for p in corners]
    return sum([
        ang(tr - tl, bl - tl),
        ang(tl - tr, br - tr),
        ang(br - bl, tl - bl),
        ang(bl - br, tr - br),
    ])


# ─────────────────────────────────────────────
#  UTILIDADES
# ─────────────────────────────────────────────

def sort_corners(pts):
    """Ordena puntos como [TL, TR, BL, BR]."""
    pts = sorted(pts, key=lambda p: p[1])
    top = sorted(pts[:2], key=lambda p: p[0])
    bot = sorted(pts[2:], key=lambda p: p[0])
    return [top[0], top[1], bot[0], bot[1]]


def rectify_frame(image: np.ndarray, corners):
    """Rectifica perspectiva usando los 4 corners."""
    fid_half = (FIDUCIAL_SIZE * CELL_SIZE) // 2
    dst_pts  = np.float32([
        [fid_half,           fid_half          ],
        [FRAME_W - fid_half, fid_half          ],
        [fid_half,           FRAME_H - fid_half],
        [FRAME_W - fid_half, FRAME_H - fid_half],
    ])
    src_pts = np.float32(corners)
    H, _    = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None
    return cv2.warpPerspective(image, H, (FRAME_W, FRAME_H))


def extract_symbols(rectified_gray: np.ndarray) -> np.ndarray:
    """Lee el valor de cada celda con margen interior."""
    from transmisor.frame_builder import get_data_cells
    data_cells = get_data_cells()
    symbols    = []
    margin     = CELL_SIZE // 4

    for row, col in data_cells:
        y   = row * CELL_SIZE + margin
        x   = col * CELL_SIZE + margin
        roi = rectified_gray[y:y+CELL_SIZE-2*margin,
                              x:x+CELL_SIZE-2*margin]
        symbols.append(int(np.median(roi)))

    return np.array(symbols, dtype=np.uint8)