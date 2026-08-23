"""
Referencia em Python do pipeline de reconhecimento facial.

Serve para derivar empiricamente todos os formatos ANTES de escrever o
Delphi: layout das 9 saidas do SCRFD, unidades do bbox/kps, se o score ja
vem com sigmoid, a normalizacao de cada modelo e a transformada de 5 pontos.
"""
import numpy as np, cv2, onnxruntime as ort

DET = "models/buffalo_l/det_10g.onnx"
REC = "models/buffalo_l/w600k_r50.onnx"
DET_SIZE = 640
STRIDES = [8, 16, 32]
NUM_ANCHORS = 2

# Template canonico do ArcFace para 112x112 (olho esq, olho dir, nariz,
# canto esq da boca, canto dir da boca).
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                        [56.0252, 71.7366], [41.5493, 92.3655],
                        [70.7299, 92.2041]], dtype=np.float32)

_det = ort.InferenceSession(DET, providers=["CPUExecutionProvider"])
_rec = ort.InferenceSession(REC, providers=["CPUExecutionProvider"])


def detect(bgr, thresh=0.5, nms_thresh=0.4, verbose=False):
    h, w = bgr.shape[:2]
    scale = min(DET_SIZE / w, DET_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    # InsightFace cola no canto superior esquerdo, nao centralizado.
    canvas = np.zeros((DET_SIZE, DET_SIZE, 3), dtype=np.uint8)
    canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh))
    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (DET_SIZE, DET_SIZE),
                                 (127.5, 127.5, 127.5), swapRB=True)
    outs = _det.run(None, {_det.get_inputs()[0].name: blob})

    if verbose:
        print(f"  score min/max: {outs[0].min():.4f} / {outs[0].max():.4f}"
              f"   -> {'ja tem sigmoid' if 0 <= outs[0].min() and outs[0].max() <= 1 else 'logit cru'}")
        print(f"  bbox  min/max: {outs[3].min():.3f} / {outs[3].max():.3f}")
        print(f"  kps   min/max: {outs[6].min():.3f} / {outs[6].max():.3f}")

    boxes, kpss, scores = [], [], []
    for i, stride in enumerate(STRIDES):
        sc = outs[i].reshape(-1)
        bb = outs[i + 3].reshape(-1, 4) * stride
        kp = outs[i + 6].reshape(-1, 10) * stride
        g = DET_SIZE // stride
        cy, cx = np.mgrid[:g, :g]
        centers = np.stack([cx, cy], -1).astype(np.float32).reshape(-1, 2) * stride
        centers = np.repeat(centers, NUM_ANCHORS, axis=0)

        keep = sc >= thresh
        if not keep.any():
            continue
        c, b, k, s = centers[keep], bb[keep], kp[keep], sc[keep]
        # bbox: distancias l,t,r,b a partir do centro da ancora
        boxes.append(np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1],
                               c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], -1))
        # kps: deslocamentos x,y a partir do centro
        kpss.append(k.reshape(-1, 5, 2) + c[:, None, :])
        scores.append(s)

    if not boxes:
        return np.zeros((0, 4)), np.zeros((0, 5, 2)), np.zeros(0)

    boxes = np.concatenate(boxes) / scale
    kpss = np.concatenate(kpss) / scale
    scores = np.concatenate(scores)

    order = scores.argsort()[::-1]
    boxes, kpss, scores = boxes[order], kpss[order], scores[order]

    keep, sup = [], np.zeros(len(scores), bool)
    for i in range(len(scores)):
        if sup[i]:
            continue
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[i + 1:, 0]); yy1 = np.maximum(boxes[i, 1], boxes[i + 1:, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[i + 1:, 2]); yy2 = np.minimum(boxes[i, 3], boxes[i + 1:, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (a[i] + a[i + 1:] - inter + 1e-9)
        sup[i + 1:][iou > nms_thresh] = True
    return boxes[keep], kpss[keep], scores[keep]


def umeyama(src, dst):
    """Transformada de similaridade (escala+rotacao+translacao) por minimos quadrados."""
    n = src.shape[0]
    src_m, dst_m = src.mean(0), dst.mean(0)
    sd, dd = src - src_m, dst - dst_m
    A = dd.T @ sd / n
    d = np.ones(2)
    if np.linalg.det(A) < 0:
        d[1] = -1
    U, S, Vt = np.linalg.svd(A)
    R = U @ np.diag(d) @ Vt
    scale = (S @ d) / sd.var(0).sum()
    M = np.zeros((2, 3), np.float32)
    M[:2, :2] = scale * R
    M[:, 2] = dst_m - scale * R @ src_m
    return M


def embed(bgr, kps, std=127.5):
    M = umeyama(kps.astype(np.float32), ARCFACE_DST)
    aligned = cv2.warpAffine(bgr, M, (112, 112), borderValue=0.0)
    blob = cv2.dnn.blobFromImage(aligned, 1.0 / std, (112, 112),
                                 (127.5, 127.5, 127.5), swapRB=True)
    v = _rec.run(None, {_rec.get_inputs()[0].name: blob})[0][0]
    return v / np.linalg.norm(v), aligned
