import numpy as np, cv2, face_ref as F

print("=== 1. Semantica das saidas do SCRFD (obama.jpg) ===")
img = cv2.imread("faces/obama.jpg")
b, k, s = F.detect(img, verbose=True)
print(f"  {len(b)} rosto(s); score do melhor: {s[0]:.4f}")
print(f"  caixa : {b[0].round(1)}")
print("  5 pontos (olho-e, olho-d, nariz, boca-e, boca-d):")
for i, p in enumerate(k[0]):
    print(f"      {i}: ({p[0]:7.1f}, {p[1]:7.1f})")

print("\n=== 2. Sanidade geometrica dos landmarks ===")
kp = k[0]
print(f"  olho esq  x={kp[0][0]:.0f} < olho dir x={kp[1][0]:.0f}  -> {'OK' if kp[0][0]<kp[1][0] else 'FALHOU'}")
print(f"  olhos y={kp[0][1]:.0f} acima do nariz y={kp[2][1]:.0f}  -> {'OK' if kp[0][1]<kp[2][1] else 'FALHOU'}")
print(f"  nariz y={kp[2][1]:.0f} acima da boca y={kp[3][1]:.0f}   -> {'OK' if kp[2][1]<kp[3][1] else 'FALHOU'}")
print(f"  landmarks dentro da caixa                         -> "
      f"{'OK' if (kp[:,0]>=b[0][0]-5).all() and (kp[:,0]<=b[0][2]+5).all() else 'FALHOU'}")

print("\n=== 3. Normalizacao do ArcFace: /127.5 vs /128 ===")
imgs = {n: cv2.imread(f"faces/{n}.jpg") for n in ["obama","obama2","biden"]}
det  = {n: F.detect(v)[1][0] for n,v in imgs.items()}
for std in (127.5, 128.0):
    e = {n: F.embed(imgs[n], det[n], std=std)[0] for n in imgs}
    mesmo = float(e["obama"] @ e["obama2"]); dif1 = float(e["obama"] @ e["biden"])
    dif2  = float(e["obama2"] @ e["biden"])
    print(f"  std={std}: mesma pessoa={mesmo:.4f}  diferentes={dif1:.4f}/{dif2:.4f}"
          f"  margem={mesmo-max(dif1,dif2):.4f}")

print("\n=== 4. Duas pessoas na mesma foto (zidane.jpg) ===")
z = cv2.imread("../delphi/bin/imagem/zidane.jpg")
zb, zk, zs = F.detect(z)
print(f"  {len(zb)} rostos, scores {[round(float(x),3) for x in zs]}")
if len(zb) >= 2:
    a,_ = F.embed(z, zk[0]); c,_ = F.embed(z, zk[1])
    print(f"  similaridade entre os dois rostos: {float(a@c):.4f}  (esperado: baixo)")

print("\n=== 5. Recorte alinhado 112x112 salvo para inspecao ===")
for n in imgs:
    _, al = F.embed(imgs[n], det[n])
    cv2.imwrite(f"faces/aligned_{n}.png", al)
print("  faces/aligned_*.png")
