"""
Referencia em Python da deteccao de texto PP-OCRv6 (DBNet).

Transcreve o pre e o pos-processamento do PaddleX
(paddlex/inference/models/text_detection/processors.py) sem depender do
pacote paddleocr, para servir de gabarito ao decoder Delphi:

  DetResizeForTest  lado menor >= 736, lado maior <= 4000, multiplo de 32
  NormalizeImage    BGR, (x/255 - media ImageNet) / desvio, NA ORDEM BGR
  DBPostProcess     thresh 0.2, box_thresh 0.45, unclip 1.4, quad, fast

Os valores vem do inference.yml que acompanha o .onnx no Hugging Face.

O reconhecimento (PP-OCRv6_medium_rec) segue o pipeline OCR do PaddleX
(pipelines/ocr/pipeline.py, sem o classificador de orientacao de linha):

  CropByPolys       retangulo minimo da caixa -> warpPerspective INTER_CUBIC,
                    girado 90 graus se altura/largura >= 1.5
  OCRReisizeNormImg altura 48, largura proporcional (minimo 320, padding 0),
                    BGR, (x/255 - 0.5) / 0.5
  CTCLabelDecode    argmax por coluna, junta repeticoes, remove o branco;
                    score = media das probabilidades dos caracteres mantidos

Uso:
  .venv/Scripts/python ocr_ref.py ../delphi/bin/imagem/ocr_exemplo.png
"""
import math
import sys

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper

MODEL = "../delphi/bin/ocr/PP-OCRv6_medium_det.onnx"
REC_MODEL = "../delphi/bin/ocr/PP-OCRv6_medium_rec.onnx"
REC_DICT = "../delphi/bin/ocr/PP-OCRv6_medium_rec.yml"
LIMIT_SIDE, LIMIT_TYPE, MAX_SIDE = 736, "min", 4000
THRESH, BOX_THRESH, UNCLIP, MAX_CAND, MIN_SIZE = 0.2, 0.45, 1.4, 3000, 3


def resize(img):
    h, w = img.shape[:2]
    if LIMIT_TYPE == "min":
        ratio = LIMIT_SIDE / min(h, w) if min(h, w) < LIMIT_SIDE else 1.0
    else:
        ratio = LIMIT_SIDE / max(h, w) if max(h, w) > LIMIT_SIDE else 1.0
    rh, rw = int(h * ratio), int(w * ratio)
    if max(rh, rw) > MAX_SIDE:
        ratio = MAX_SIDE / max(rh, rw)
        rh, rw = int(rh * ratio), int(rw * ratio)
    rh = max(int(round(rh / 32) * 32), 32)
    rw = max(int(round(rw / 32) * 32), 32)
    if (rh, rw) == (h, w):
        return img
    return cv2.resize(img, (rw, rh))


def normalize(bgr):
    # A media/desvio sao aplicados na ordem dos canais da imagem, que e BGR:
    # o canal B recebe 0.485/0.229. Quirk herdado do PaddleOCR.
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = (bgr.astype(np.float32) / 255.0 - mean) / std
    return x.transpose(2, 0, 1)[None]


def mini_box(contour):
    rect = cv2.minAreaRect(contour)
    pts = sorted(list(cv2.boxPoints(rect)), key=lambda p: p[0])
    i1, i4 = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
    i2, i3 = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
    return np.array([pts[i1], pts[i2], pts[i3], pts[i4]]), min(rect[1])


def score_fast(pred, box):
    h, w = pred.shape
    xmin = max(0, min(math.floor(box[:, 0].min()), w - 1))
    xmax = max(0, min(math.ceil(box[:, 0].max()), w - 1))
    ymin = max(0, min(math.floor(box[:, 1].min()), h - 1))
    ymax = max(0, min(math.ceil(box[:, 1].max()), h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
    b = box.copy()
    b[:, 0] -= xmin
    b[:, 1] -= ymin
    cv2.fillPoly(mask, b.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(pred[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box):
    poly = box.astype(np.float32)
    distance = cv2.contourArea(poly) * UNCLIP / cv2.arcLength(poly, True)
    off = pyclipper.PyclipperOffset()
    off.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    return np.array(off.Execute(distance)[0])


def detect(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    src_h, src_w = bgr.shape[:2]
    net = resize(bgr)
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    pred = sess.run(None, {sess.get_inputs()[0].name: normalize(net)})[0][0, 0]
    h, w = pred.shape
    print(f"imagem {src_w}x{src_h} -> rede {net.shape[1]}x{net.shape[0]}"
          f"  mapa {w}x{h}  prob min/max {pred.min():.3f}/{pred.max():.3f}")

    bitmap = (pred > THRESH).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours[:MAX_CAND]:
        pts, sside = mini_box(contour)
        if sside < MIN_SIZE:
            continue
        score = score_fast(pred, pts)
        if score < BOX_THRESH:
            continue
        box, sside = mini_box(unclip(pts).reshape(-1, 1, 2))
        if sside < MIN_SIZE + 2:
            continue
        box[:, 0] = np.clip(np.round(box[:, 0] * src_w / w), 0, src_w)
        box[:, 1] = np.clip(np.round(box[:, 1] * src_h / h), 0, src_h)
        out.append((box.astype(int), score))

    # Ordem de leitura: SortQuadBoxes do pipeline OCR.
    out.sort(key=lambda b: (b[0][0][1], b[0][0][0]))
    for i in range(len(out) - 1):
        for j in range(i, -1, -1):
            if (abs(out[j + 1][0][0][1] - out[j][0][0][1]) < 10
                    and out[j + 1][0][0][0] < out[j][0][0][0]):
                out[j], out[j + 1] = out[j + 1], out[j]
            else:
                break
    return bgr, out


def load_dict(path):
    import yaml
    chars = yaml.safe_load(open(path, encoding="utf-8"))["PostProcess"]["character_dict"]
    return ["blank"] + [str(c) for c in chars] + [" "]


def crop_line(img, poly):
    rect = cv2.minAreaRect(np.array(poly).astype(np.int32))
    pts = sorted(list(cv2.boxPoints(rect)), key=lambda p: p[0])
    a, d = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
    b, c = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
    pts = np.array([pts[a], pts[b], pts[c], pts[d]], np.float32)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    std = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, std), (w, h),
                              borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if out.shape[0] / out.shape[1] >= 1.5:
        out = np.rot90(out)
    return out


def rec_input(crop):
    h, w = crop.shape[:2]
    ratio = w / float(h)
    img_w = int(48 * max(320 / 48, ratio))
    if img_w > 3200:
        resized, rw, img_w = cv2.resize(crop, (3200, 48)), 3200, 3200
    else:
        rw = img_w if math.ceil(48 * ratio) > img_w else int(math.ceil(48 * ratio))
        resized = cv2.resize(crop, (rw, 48))
    x = (resized.astype(np.float32).transpose(2, 0, 1) / 255 - 0.5) / 0.5
    blob = np.zeros((1, 3, 48, img_w), np.float32)
    blob[0, :, :, :rw] = x
    return blob


def ctc_decode(probs, chars):
    idx = probs.argmax(-1)
    prob = probs.max(-1)
    keep = np.ones(len(idx), bool)
    keep[1:] = idx[1:] != idx[:-1]
    keep &= idx != 0
    text = "".join(chars[i] for i in idx[keep])
    return text, float(prob[keep].mean()) if keep.any() else 0.0


def recognize(bgr, boxes):
    chars = load_dict(REC_DICT)
    sess = ort.InferenceSession(REC_MODEL, providers=["CPUExecutionProvider"])
    out = []
    for box, _ in boxes:
        crop = crop_line(bgr, box)
        if crop.size == 0:
            continue
        probs = sess.run(None, {"x": rec_input(crop)})[0][0]
        out.append(ctc_decode(probs, chars))
    return out


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "../delphi/bin/imagem/ocr_exemplo.png"
    bgr, boxes = detect(image)
    texts = recognize(bgr, boxes)
    print(f"{len(boxes)} linha(s) de texto")
    for i, ((box, score), (text, rec_score)) in enumerate(zip(boxes, texts)):
        if i < 15 or "--all" in sys.argv:
            print(f"  #{i + 1:<3d} det {score * 100:6.2f}%  rec {rec_score * 100:6.2f}%  {text}")
    for box, _ in boxes:
        cv2.polylines(bgr, [box.reshape(-1, 1, 2)], True, (0, 0, 255), 2)
    cv2.imwrite("../delphi/bin/saida/ocr_ref.png", bgr)
