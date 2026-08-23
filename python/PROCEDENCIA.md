# Procedência dos modelos

Registro dos pesos usados e do ambiente que os exportou, para que os `.onnx`
possam ser reproduzidos bit a bit em outra máquina.

## Pesos oficiais

Baixados pela própria `ultralytics` a partir de `github.com/ultralytics/assets`
release **v8.4.0**. A Ultralytics distribui apenas `.pt`; não existe `.onnx`
oficial.

| tam | arquivo | MB (onnx) | sha256 do `.pt` |
|---|---|---:|---|
| `n` | `yolo26n.pt` | 9.9 | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| `n` | `yolo26n-seg.pt` | 11.2 | `361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f` |
| `n` | `yolo26n-pose.pt` | 12.1 | `eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9` |
| `n` | `yolo26n-obb.pt` | 10.2 | `6f51c78197aacda4a33be77294065a9001675fb893f56227a179731b53dbd2b0` |
| `n` | `yolo26n-cls.pt` | 11.3 | `0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81` |
| `s` | `yolo26s.pt` | 38.3 | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` |
| `s` | `yolo26s-seg.pt` | 41.9 | `3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4` |
| `s` | `yolo26s-pose.pt` | 41.9 | `a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646` |
| `s` | `yolo26s-obb.pt` | 39.4 | `38dbd72ef6804f9bbbea7ad20f486e6ca6e093c8cd9bc857207a846565bd6e0b` |
| `s` | `yolo26s-cls.pt` | 26.9 | `816790029d5df3fef358f03c8144b96339d8824ee25577aeda8be0963e5c5f09` |
| `m` | `yolo26m.pt` | 82.0 | `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7` |
| `m` | `yolo26m-seg.pt` | 94.6 | `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5` |
| `m` | `yolo26m-pose.pt` | 86.6 | `2fbf16367022256a226035695c5c389384c6706e8bb8ab8fcd0e7976f05443c4` |
| `m` | `yolo26m-obb.pt` | 85.3 | `23e0630f66857cf4b87535f6e705b065f1e8a33603640b8e61ace85b75312903` |
| `m` | `yolo26m-cls.pt` | 46.6 | `9f6546f33a70d910e2cd6dcca5265c4617b5670b19a1f287c39e99258bade01a` |
| `l` | `yolo26l.pt` | 99.6 | `9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26` |
| `l` | `yolo26l-seg.pt` | 112.3 | `636024306410afa1732692322fba57d22ea2b1c2f07613fcee131a93d7dd380c` |
| `l` | `yolo26l-pose.pt` | 104.2 | `ad33da8a29ea5772318c4c980844e47b56792d2b63815ad4e8e09c078c7d1abf` |
| `l` | `yolo26l-obb.pt` | 102.9 | `8674b0c24bf68aab5eb45009e0ac3808ce432237edf8cb5c50ae2191cb263a2b` |
| `l` | `yolo26l-cls.pt` | 56.5 | `7a3aebb7de6f5d79f0153da627a5f68bd3d7df30e648c0fdb7f9217986ee129e` |
| `x` | `yolo26x.pt` | 223.3 | `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92` |
| `x` | `yolo26x-seg.pt` | 251.7 | `92b3de0065766a17180d6219858717dc9d03cdce8a3ca9576c97fd75aabb64f3` |
| `x` | `yolo26x-pose.pt` | 230.7 | `08ed9e01d22a6f248b04f2f9992016aca9a32250b9ab57057d886a09d026700d` |
| `x` | `yolo26x-obb.pt` | 230.7 | `d7b1d805b0ec149890efde89119ef9d60481566ec83166052c04b5d89d9e3cb7` |
| `x` | `yolo26x-cls.pt` | 118.6 | `ee88a0c71e9596cdfdbb71892725929cb4489b874d24fb581d20bffe79724386` |

As variantes `end2end=False` de `export_raw.py` usam **os mesmos pesos** do
tamanho `n` acima; só mudam os argumentos de export.

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

O formato da saída depende só da tarefa, não do tamanho — os cinco tamanhos de
uma mesma tarefa são intercambiáveis para o decoder.

| tarefa | entrada | saídas | end2end |
|---|---|---|---|
| detect | `[1,3,640,640]` | `[1,300,6]` | True |
| segment | `[1,3,640,640]` | `[1,300,38]` + `[1,32,160,160]` | True |
| pose | `[1,3,640,640]` | `[1,300,57]` | True |
| obb | `[1,3,1024,1024]` | `[1,300,7]` | True |
| classify | `[1,3,224,224]` | `[1,1000]` | False |

Com `export_raw.py` (`end2end=False`), tamanho `n`:

| tarefa | saídas |
|---|---|
| detect | `[1,84,8400]` |
| segment | `[1,116,8400]` + `[1,32,160,160]` |
| pose | `[1,56,8400]` |
| obb | `[1,20,21504]` |

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
.venv/Scripts/python dump_cols.py ../delphi/bin/yolo/x/yolo26x-obb.onnx ../delphi/bin/imagem/boats.jpg 1024
```

`face_ref.py` e `verificar_face.py` são a implementação de referência do
pipeline facial (SCRFD + alinhamento de 5 pontos + ArcFace) e conferem os
valores contra `referencia_face.json`.
