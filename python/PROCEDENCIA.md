# Procedência dos modelos

Os `.onnx` que estavam em `delphi/bin/yolo/` vieram de terceiros. Um deles
(`yolo26x-obb.onnx`) declarava `end2end=True` nos metadados mas emitia a cabeça
crua em 9 tensores NHWC — inconsistente com o próprio metadado, e inutilizável
também pelo predictor Python da Ultralytics. Os originais foram preservados em
`delphi/bin/yolo/_terceiros/`.

Estes aqui foram baixados e exportados por `export_models.py` neste venv.

## Pesos oficiais

Baixados pela própria `ultralytics` a partir de
`github.com/ultralytics/assets` release **v8.4.0** (a Ultralytics distribui
apenas `.pt`; não existe `.onnx` oficial).

| arquivo | sha256 |
|---|---|
| `yolo26x.pt` | `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92` |
| `yolo26x-seg.pt` | `92b3de0065766a17180d6219858717dc9d03cdce8a3ca9576c97fd75aabb64f3` |
| `yolo26x-pose.pt` | `08ed9e01d22a6f248b04f2f9992016aca9a32250b9ab57057d886a09d026700d` |
| `yolo26x-obb.pt` | `d7b1d805b0ec149890efde89119ef9d60481566ec83166052c04b5d89d9e3cb7` |
| `yolo26l-cls.pt` | `7a3aebb7de6f5d79f0153da627a5f68bd3d7df30e648c0fdb7f9217986ee129e` |

> `.pt` é pickle do Python: **executa código ao carregar**. Baixe só das fontes
> oficiais (release acima ou `huggingface.co/Ultralytics/YOLO26`). `.onnx` é
> protobuf, só dado — risco muito menor.

## Ambiente do export

`ultralytics 8.4.126` · `torch 2.13.0+cpu` · `onnx 1.22.0` · `onnxslim 0.1.96`
· `opset=12` · `simplify=True`

## Saídas resultantes

| modelo | entrada | saídas | end2end |
|---|---|---|---|
| `yolo26x.onnx` | `[1,3,640,640]` | `[1,300,6]` | True |
| `yolo26x-seg.onnx` | `[1,3,640,640]` | `[1,300,38]` + `[1,32,160,160]` | True |
| `yolo26x-pose.onnx` | `[1,3,640,640]` | `[1,300,57]` | True |
| `yolo26x-obb.onnx` | `[1,3,1024,1024]` | `[1,300,7]` | True |
| `yolo26l-cls.onnx` | `[1,3,224,224]` | `[1,1000]` | False |

Diferenças relevantes contra os arquivos de terceiros: o `-cls` deles fora
exportado a **640x640** (o oficial é 224x224), e o `-obb` deles não era
decodificado.

## Reproduzir

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install ultralytics onnx onnxslim onnxruntime
.venv/Scripts/python export_models.py
```

`dump_cols.py` e `dump_obb.py` inspecionam o tensor de saída cru — foram eles
que revelaram que a cabeça OBB usa `cx,cy,w,h` enquanto detect/segment/pose
usam `x1,y1,x2,y2`.
