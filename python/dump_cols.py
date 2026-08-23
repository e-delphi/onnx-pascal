import numpy as np, onnxruntime as ort, sys
from PIL import Image
MODEL, IMG, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
im = Image.open(IMG).convert("RGB"); w,h = im.size
s = min(N/w, N/h); nw,nh = round(w*s), round(h*s)
c = Image.new("RGB",(N,N),(114,114,114)); ox,oy=(N-nw)//2,(N-nh)//2
c.paste(im.resize((nw,nh), Image.BILINEAR),(ox,oy))
x = np.asarray(c,dtype=np.float32).transpose(2,0,1)[None]/255.0
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
print(f"{MODEL.split(chr(92))[-1]}  shape={out.shape}  scale={s:.4f} ox={ox} oy={oy}")
np.set_printoptions(suppress=True, precision=2, linewidth=250)
print("  6 primeiras colunas, 4 melhores linhas:")
print(out[0,:4,:6])
c0,c1,c2,c3 = (out[0,:,i] for i in range(4))
print(f"  col2>col0 em {100*(c2>c0).mean():.0f}% das linhas, col3>col1 em {100*(c3>c1).mean():.0f}%  "
      f"-> {'CANTOS (x1y1x2y2)' if (c2>c0).mean()>0.9 else 'CENTRO (cxcywh)'}")
print()
