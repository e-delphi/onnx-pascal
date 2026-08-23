# Procedência dos modelos

Registro dos pesos usados e do ambiente que os exportou, para que os `.onnx`
possam ser reproduzidos bit a bit em outra máquina.

## Pesos oficiais

Baixados pela própria `ultralytics` a partir de `github.com/ultralytics/assets`
release **v8.4.0**. A Ultralytics distribui apenas `.pt`; não existe `.onnx`
oficial.

| arquivo | sha256 |
|---|---|
| `yolo26x.pt` | `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92` |
| `yolo26x-seg.pt` | `92b3de0065766a17180d6219858717dc9d03cdce8a3ca9576c97fd75aabb64f3` |
| `yolo26x-pose.pt` | `08ed9e01d22a6f248b04f2f9992016aca9a32250b9ab57057d886a09d026700d` |
| `yolo26x-obb.pt` | `d7b1d805b0ec149890efde89119ef9d60481566ec83166052c04b5d89d9e3cb7` |
| `yolo26l-cls.pt` | `7a3aebb7de6f5d79f0153da627a5f68bd3d7df30e648c0fdb7f9217986ee129e` |

Variantes `end2end=False` geradas por `export_raw.py`:

| arquivo | sha256 |
|---|---|
| `yolo26n.pt` | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| `yolo26n-seg.pt` | `361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f` |
| `yolo26n-pose.pt` | `eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9` |
| `yolo26n-obb.pt` | `6f51c78197aacda4a33be77294065a9001675fb893f56227a179731b53dbd2b0` |

Modelos faciais: `buffalo_l.zip` do release **v0.7** de
`github.com/deepinsight/insightface`,
sha256 `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f`.

> `.pt` é pickle do Python: **executa código ao carregar**. Baixe só das fontes
> oficiais (releases acima ou `huggingface.co/Ultralytics/YOLO26`). `.onnx` é
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
| `raw/yolo26n.onnx` | `[1,3,640,640]` | `[1,84,8400]` | False |
| `raw/yolo26n-seg.onnx` | `[1,3,640,640]` | `[1,116,8400]` + `[1,32,160,160]` | False |
| `raw/yolo26n-pose.onnx` | `[1,3,640,640]` | `[1,56,8400]` | False |
| `raw/yolo26n-obb.onnx` | `[1,3,1024,1024]` | `[1,20,21504]` | False |

## Reproduzir

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install ultralytics onnx onnxslim onnxruntime
.venv/Scripts/python export_models.py
.venv/Scripts/python export_raw.py
```

## Scripts de inspeção

`dump_cols.py` e `dump_obb.py` imprimem o tensor de saída cru de um modelo —
faixa por coluna e primeiras linhas. Servem para conferir o layout de uma
cabeça antes de escrever ou ajustar um decoder.

```bash
.venv/Scripts/python dump_cols.py ../delphi/bin/yolo/yolo26x-obb.onnx ../delphi/bin/imagem/boats.jpg 1024
```

`face_ref.py` e `verificar_face.py` são a implementação de referência do
pipeline facial (SCRFD + alinhamento de 5 pontos + ArcFace) e conferem os
valores contra `referencia_face.json`.
