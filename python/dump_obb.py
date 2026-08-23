import numpy as np, onnxruntime as ort
from PIL import Image

MODEL = r"..\delphi\bin\yolo\yolo26x-obb.onnx"
IMG   = r"..\delphi\bin\imagem\boats.jpg"
N = 1024

im = Image.open(IMG).convert("RGB")
w, h = im.size
s = min(N/w, N/h)
nw, nh = round(w*s), round(h*s)
canvas = Image.new("RGB", (N, N), (114,114,114))
ox, oy = (N-nw)//2, (N-nh)//2
canvas.paste(im.resize((nw,nh), Image.BILINEAR), (ox,oy))
x = np.asarray(canvas, dtype=np.float32).transpose(2,0,1)[None]/255.0

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
print("shape:", out.shape, " letterbox: scale=%.4f ox=%d oy=%d" % (s,ox,oy))
print()
print("por coluna:  min / max / media")
for c in range(out.shape[2]):
    col = out[0,:,c]
    print(f"  col{c}:  {col.min():10.4f}  {col.max():10.4f}  {col.mean():10.4f}")
print()
print("primeiras 8 linhas:")
np.set_printoptions(suppress=True, precision=3, linewidth=200)
print(out[0,:8])
print()
print("ultimas 3 linhas:")
print(out[0,-3:])
