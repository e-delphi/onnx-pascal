# ONNXDemo — visão computacional em Delphi com ONNX Runtime

Aplicação console Win64 que roda modelos ONNX de visão sem Python, OpenCV ou
qualquer wrapper: apenas `onnxruntime.dll` e a VCL para abrir e salvar imagem.
O binding para a C API 1.28 é escrito à mão e vive em `delphi/src/ONNX.*.pas`.

| Tarefa | Modelo de exemplo | Resultado |
|---|---|---|
| `classify` | SqueezeNet, YOLO26-cls | top-K classes com probabilidade |
| `detect` | YOLO26 | caixas + classe + score |
| `segment` | YOLO26-seg | caixas + máscara por instância |
| `pose` | YOLO26-pose | caixas + 17 keypoints + esqueleto |
| `obb` | YOLO26-obb | caixas rotacionadas + ângulo |
| `face` | SCRFD (buffalo_l) | caixas + 5 landmarks faciais |
| `embed` | ArcFace (buffalo_l) | vetor 512-D para comparar rostos |
| `text` | PP-OCRv6_medium_det (PaddleOCR) | linhas de texto como caixas rotacionadas |
| `--ocr` | PP-OCRv6_medium_det + _rec | texto lido, linha a linha, em ordem de leitura |

---

## Requisitos

- **Windows x64**
- **Delphi 12 (BDS 23.0)** — a Community Edition serve. O projeto usa só VCL e
  RTL, sem componentes de terceiros.
- **Python 3.11+** — opcional, necessário apenas para exportar os modelos.
- ~3 GB livres (modelos + venv).

> A Community Edition **não compila por linha de comando**. Todos os builds
> abaixo são feitos abrindo o `.dproj` no IDE e pressionando Shift+F9.

---

## Passo a passo

Os passos 1–3 já produzem um executável funcionando. Os passos 4–7 baixam os
modelos de cada tarefa; faça só os que interessarem.

### 1. Clonar

```bash
git clone <url-do-repositorio> onnx-pascal
cd onnx-pascal
```

O repositório contém apenas código, scripts e documentação — nenhum binário. O
`.gitignore` exclui de propósito o runtime, os modelos, as imagens e a galeria.

### 2. ONNX Runtime 1.28.1

Baixe o pacote oficial da Microsoft e coloque a DLL ao lado do executável:

```bash
curl -L -o ort.zip https://github.com/microsoft/onnxruntime/releases/download/v1.28.1/onnxruntime-win-x64-1.28.1.zip
unzip -j ort.zip "*/lib/onnxruntime.dll" -d delphi/bin/
rm ort.zip
```

Sem PowerShell/curl, baixe manualmente de
[github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases)
(tag `v1.28.1`, arquivo `onnxruntime-win-x64-1.28.1.zip`) e copie
`lib/onnxruntime.dll` para `delphi/bin/`.

> **A versão importa.** `delphi/src/ONNX.CApi.pas` declara
> `ORT_API_VERSION = 28` e mapeia a tabela de 424 slots da struct `OrtApi`,
> conferida contra o `onnxruntime_c_api.h` da 1.28.1. Uma DLL mais antiga
> falha na hora com mensagem clara; uma mais nova normalmente funciona,
> porque a tabela só cresce no fim.

### 3. Compilar

1. Abra `delphi/ONNXDemo.dproj` no Delphi 12
2. Selecione a plataforma **Windows 64-bit**
3. Shift+F9 (Build)

O executável sai em `delphi/bin/ONNXDemo.exe`. Confira:

```bash
cd delphi/bin && ./ONNXDemo.exe --help
```

### 4. Ambiente Python (para exportar modelos)

Só é necessário se você for gerar os modelos YOLO26. Se alguém já te passou os
`.onnx`, pule para o passo 6.

```bash
cd python && py -m venv .venv
```

```bash
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv/Scripts/python -m pip install ultralytics onnx onnxslim onnxruntime
```

> O PyTorch vem do índice **CPU-only** de propósito: ~250 MB em vez de ~3 GB, e
> é suficiente para exportar ONNX.

### 5. Modelos YOLO26

```bash
cd python && .venv/Scripts/python export_models.py
```

O script baixa os pesos oficiais do release
[`ultralytics/assets` v8.4.0](https://github.com/ultralytics/assets/releases),
exporta com `opset=12` e imprime o SHA256 de cada `.pt`. Os `.onnx` saem
organizados por tamanho em `delphi/bin/yolo/<n|s|m|l|x>/`, cada pasta com as
cinco tarefas — 25 modelos ao todo, o que permite comparar velocidade e
acurácia entre tamanhos.

Para gerar só um recorte da matriz:

```bash
cd python && .venv/Scripts/python export_models.py --sizes n,s --tasks detect,pose
```

Os hashes esperados estão em [python/PROCEDENCIA.md](python/PROCEDENCIA.md) —
compare, se quiser conferir a integridade.

Para gerar também as variantes sem NMS-free (que exercitam o caminho de
supressão de não-máximos):

```bash
cd python && .venv/Scripts/python export_raw.py
```

> **Não existe `.onnx` oficial de YOLO26.** A Ultralytics distribui só `.pt`,
> então qualquer `.onnx` pronto encontrado na internet é export de terceiro,
> sem garantia de que os argumentos e o formato de saída sejam os esperados.
> Exporte você mesmo.
>
> **Atenção:** `.pt` é pickle do Python e **executa código ao carregar**. Baixe
> apenas das fontes oficiais. O `.onnx` é protobuf, só dado — risco bem menor.

### 6. Modelos faciais

O pacote `buffalo_l` da InsightFace traz o detector e o reconhecedor com
pré-processamento consistente entre si:

```bash
curl -L -o buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
```

```bash
unzip -j buffalo_l.zip det_10g.onnx w600k_r50.onnx -d delphi/bin/face/
```

`det_10g.onnx` (17 MB) é o detector SCRFD com 5 landmarks; `w600k_r50.onnx`
(174 MB) é o ArcFace que gera o vetor de 512 dimensões. O zip traz ainda
`2d106det.onnx` (106 pontos 2D), `1k3d68.onnx` (68 pontos 3D) e
`genderage.onnx`, que este projeto ainda não usa.

> Os pesos da InsightFace são licenciados **apenas para pesquisa
> não-comercial**. Para produto, é preciso trocar o modelo ou licenciar.

### 7. Detecção e leitura de texto (PaddleOCR)

O `PP-OCRv6_medium_det` é o detector mais preciso da família PP-OCRv6 e o
`PP-OCRv6_medium_rec`, o reconhecedor mais preciso. Ao contrário do YOLO26,
**ambos têm `.onnx` oficial**, publicado pela própria PaddlePaddle:

```bash
mkdir -p delphi/bin/ocr
```

```bash
curl -L -o delphi/bin/ocr/PP-OCRv6_medium_det.onnx https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx/resolve/main/inference.onnx
```

```bash
curl -L -o delphi/bin/ocr/PP-OCRv6_medium_rec.onnx https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/main/inference.onnx
```

```bash
curl -L -o delphi/bin/ocr/PP-OCRv6_medium_rec.yml https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/main/inference.yml
```

O `.yml` do reconhecedor **é obrigatório**: o `.onnx` devolve índices, e o
dicionário de 18.708 caracteres (50 idiomas, acentos do português incluídos)
só existe nesse arquivo. Salve-o com o mesmo nome do modelo — o programa o
procura ali.

| modelo | Hmean médio | parâmetros | `.onnx` |
|---|---:|---:|---:|
| **PP-OCRv6_medium_det** | **86,2** | 15,5 M | 62 MB |
| PP-OCRv6_small_det | 84,1 | — | — |
| PP-OCRv6_tiny_det | 80,6 | — | — |
| PP-OCRv5_server_det | 81,6 | — | — |

| reconhecedor | acurácia média | parâmetros | `.onnx` |
|---|---:|---:|---:|
| **PP-OCRv6_medium_rec** | **83,2** | 19 M | 77 MB |
| PP-OCRv6_small_rec | 81,3 | — | — |
| PP-OCRv6_tiny_rec | 73,5 | — | — |
| PP-OCRv5_server_rec | 78,1 | — | — |

Números dos model cards. Os irmãos `small` e `tiny` (repositórios
`PaddlePaddle/PP-OCRv6_small_det_onnx`, `..._small_rec_onnx` etc.) têm as
mesmas cabeças e rodam sem mudar nada — trocam acurácia por velocidade. Use
`--ocr-det`/`--ocr-rec` e baixe o `.yml` do reconhecedor escolhido.

O detector não traz metadados: a tarefa é deduzida da saída única `[1, 1, H, W]`.
Os limiares padrão do programa são os do `inference.yml` que acompanha o
modelo (`thresh 0.2`, `box_thresh 0.45`, `unclip_ratio 1.4`, lado menor ≥ 736).

### 8. Imagens de teste

```bash
mkdir -p delphi/bin/imagem && cd delphi/bin/imagem
```

```bash
curl -L -O https://ultralytics.com/images/bus.jpg -O https://ultralytics.com/images/zidane.jpg -O https://ultralytics.com/images/boats.jpg
```

```bash
curl -L -O https://github.com/ageitgey/face_recognition/raw/master/examples/obama.jpg -O https://github.com/ageitgey/face_recognition/raw/master/examples/obama2.jpg -O https://github.com/ageitgey/face_recognition/raw/master/examples/biden.jpg
```

```bash
curl -L -o ocr_exemplo.png https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/3ul2Rq4Sk5Cn-l69D695U.png
```

`bus.jpg` (ônibus + 4 pessoas), `zidane.jpg` (2 rostos), `boats.jpg` (marina
aérea, para OBB), o trio Obama/Biden (duas fotos da mesma pessoa + uma de
outra, para validar reconhecimento facial) e `ocr_exemplo.png`, a página de
exemplo do model card do PP-OCRv6.

---

## Uso

### Inferência

Modelo e imagem são obrigatórios; o argumento terminado em `.onnx` é tomado
como modelo, o outro como imagem. Sem argumentos, o programa imprime a ajuda.

```bash
./ONNXDemo.exe yolo/n/yolo26n.onnx imagem/bus.jpg
```

```bash
./ONNXDemo.exe yolo/x/yolo26x-seg.onnx imagem/bus.jpg --conf=0.4
```

```bash
./ONNXDemo.exe squeezenet/squeezenet1_1.onnx imagem/dog.jpg --labels=squeezenet/labels.txt --topk=10
```

```bash
./ONNXDemo.exe ocr/PP-OCRv6_medium_det.onnx imagem/ocr_exemplo.png
```

A tarefa é lida dos metadados do `.onnx` e o layout da saída é decidido pelo
shape real do tensor em tempo de execução — não há nada a configurar. Use
`--task` só para forçar.

As imagens anotadas vão para `saida/`, nunca junto das entradas.

### Texto

As linhas saem em ordem de leitura, cada uma com score e retângulo rotacionado.
Na imagem anotada só o contorno é desenhado — rótulo em cada linha cobriria o
próprio texto.

| opção | padrão | efeito |
|---|---|---|
| `--conf` | 0,45 | score médio mínimo dentro da caixa (`box_thresh`) |
| `--db-thr` | 0,2 | limiar que binariza o mapa de probabilidade |
| `--unclip` | 1,4 | quanto a caixa cresce sobre a região detectada |
| `--det-side` | 736 | limite de lado no redimensionamento |
| `--det-limit` | `min` | `min`: lado menor ≥ limite; `max`: lado maior ≤ limite |
| `--max-det` | 3000 | máximo de regiões examinadas |

Em fotos grandes, `--det-limit=max --det-side=960` fica bem mais rápido, ao
custo de perder texto pequeno.

### OCR

```bash
./ONNXDemo.exe --ocr imagem/ocr_exemplo.png
```

Detecta as linhas, recorta cada uma reta e a lê. O console lista cada linha
com a confiança da detecção e da leitura; o texto completo vai para
`saida/<imagem>_ocr.txt` (UTF-8) e `saida/<imagem>_ocr.png` mostra a imagem com
as caixas ao lado do texto lido, cada um na sua posição.

| opção | padrão | efeito |
|---|---|---|
| `--ocr-det` | `ocr/PP-OCRv6_medium_det.onnx` | detector |
| `--ocr-rec` | `ocr/PP-OCRv6_medium_rec.onnx` | reconhecedor |
| `--ocr-dict` | o `.yml` ao lado do reconhecedor | dicionário de caracteres |
| `--rec-thr` | 0 | descarta linhas lidas com confiança menor |

As opções de detecção de texto da tabela anterior também valem aqui.

> **Limitação:** não há classificador de orientação de linha (o PaddleOCR o
> trata como um terceiro modelo opcional). Texto de cabeça para baixo, ou
> vertical escrito de baixo para cima, sai como lixo — igual ao PaddleOCR com
> `use_textline_orientation=False`. No `ocr_exemplo.png`, é o caso da
> anotação vertical do arXiv na margem.

O reconhecedor também roda sozinho sobre uma imagem que já seja uma linha
recortada:

```bash
./ONNXDemo.exe ocr/PP-OCRv6_medium_rec.onnx linha.png --task=rec --labels=ocr/PP-OCRv6_medium_rec.yml
```

### Rostos

```bash
./ONNXDemo.exe --enroll="Obama" imagem/obama.jpg
```

```bash
./ONNXDemo.exe --query imagem/obama2.jpg
```

```bash
./ONNXDemo.exe --compare imagem/obama.jpg imagem/biden.jpg
```

```bash
./ONNXDemo.exe --list
```

A galeria é um arquivo de texto (`faces.gallery`), uma linha por rosto:
nome, origem e o vetor. Inspecionável e diffável.

> A galeria contém **dados biométricos**, tratados como dados pessoais
> sensíveis pela LGPD. O `.gitignore` a exclui. Se o uso for além de pessoal,
> veja consentimento e retenção com quem cuida disso.

`ONNXDemo.exe --help` lista todas as opções.

---

## Verificação

Rode a bateria de regressão:

```bash
cd delphi && ./run_tests.sh
```

Ou confira manualmente estes valores, que são estáveis entre máquinas:

| comando | esperado |
|---|---|
| `yolo26x.onnx imagem/bus.jpg` | 5 objetos: `bus` 98,0% + 4 `person` |
| `yolo26x-pose.onnx imagem/zidane.jpg` | 2 `person`, 8 de 17 keypoints visíveis |
| `yolo26x-obb.onnx imagem/boats.jpg` | ~179 `ship`, ângulos 21–26° |
| `squeezenet1_1.onnx imagem/dog.jpg` | `Samoyed` 99,88% |
| `--compare imagem/obama.jpg imagem/obama2.jpg` | cosseno ≈ **0,77** → mesma pessoa |
| `--compare imagem/obama.jpg imagem/biden.jpg` | cosseno ≈ **−0,04** → diferentes |
| `PP-OCRv6_medium_det.onnx imagem/ocr_exemplo.png` | **98** linhas; a 1ª com 88,8% |
| `PP-OCRv6_medium_det.onnx imagem/bus.jpg` | **8** linhas (letreiro, "cero", "emisiones"…) |
| `--ocr imagem/ocr_exemplo.png` | 1ª linha: *Algorithms for the Markov Entropy Decomposition* |

O `dog.jpg` é a imagem canônica do PyTorch, cuja resposta documentada é
*Samoyed*: se der outra coisa, o pré-processamento está errado em algum ponto.

A detecção e a leitura de texto têm referência própria em `python/ocr_ref.py`,
que transcreve o pré e o pós-processamento do PaddleX com OpenCV. Em PNG o
resultado coincide (98 linhas, scores iguais até a 2ª casa). Em JPEG as caixas
coincidem, mas regiões de contraste baixo podem mudar alguns pontos de score:
o decodificador JPEG da VCL e o resize bilinear em ponto flutuante diferem do
OpenCV em ±1 nível de cinza, e o letreiro de LED do `bus.jpg` é sensível a
isso (74% aqui, 58% no Python).

Os valores de referência do pipeline facial (caixa, os 5 landmarks e as
primeiras posições do vetor), gerados pela implementação Python, estão em
`python/referencia_face.json`.

---

## Estrutura

```
.
├── delphi/
│   ├── ONNXDemo.dpr          composition root: só amarra as peças
│   ├── src/                  31 units — ver delphi/README.md
│   ├── run_tests.sh          bateria de regressão
│   └── bin/
│       ├── onnxruntime.dll   (passo 2)
│       ├── yolo/n|s|m|l|x/   (passo 5) — 5 tarefas por tamanho
│       ├── face/             (passo 6)
│       ├── ocr/              (passo 7) — 2 modelos + dicionário .yml
│       ├── imagem/           entradas (passo 8)
│       └── saida/            tudo que o programa gera
└── python/
    ├── export_models.py      pesos oficiais → ONNX
    ├── export_raw.py         variantes end2end=False
    ├── face_ref.py           referência do pipeline facial
    ├── verificar_face.py     confere a referência facial ponta a ponta
    ├── ocr_ref.py            referência do OCR PP-OCRv6 (detecção + leitura)
    ├── dump_cols.py          inspeciona o tensor cru de saída
    ├── dump_obb.py           idem, focado na cabeça OBB
    └── PROCEDENCIA.md        SHA256 dos pesos e ambiente do export
```

Os scripts em `python/` não fazem parte do produto: servem para gerar os
modelos e para inspecionar o formato real das saídas de uma cabeça — útil ao
escrever ou ajustar um decoder.

A arquitetura interna, o mapeamento SOLID e as decisões de projeto estão em
[delphi/README.md](delphi/README.md).

---

## Solução de problemas

**`Nao foi possivel carregar onnxruntime.dll (erro Win32 126)`**
A DLL não está ao lado do `.exe`. Refaça o passo 2.

**`A DLL nao suporta a ONNX Runtime C API versao 28`**
DLL anterior à 1.28. Baixe a versão certa.

**`Tarefa "unknown" nao suportada`**
O `.onnx` não traz metadados e o formato não foi dedutível. Force com
`--task=detect` (ou `segment`, `pose`, `obb`, `classify`, `text`).

**`Nenhuma saida com formato de predicao (rank 2 ou 3)`**
Export com cabeça não decodificada. Reexporte com `export_models.py`.

**`Dicionario do reconhecedor nao encontrado`**
Falta o `inference.yml` do reconhecedor, salvo como
`ocr/PP-OCRv6_medium_rec.yml` (passo 7), ou use `--ocr-dict=ARQ`.

**`Dicionario com N caracteres nao combina com a saida`**
O `.yml` é de outro reconhecedor. Cada modelo tem o seu.

**`Detector de rostos nao encontrado`**
Passo 6 não foi feito, ou use `--face-detect=CAMINHO`.

**Resultado sai vazio ou absurdo**
Quase sempre é pré-processamento. Rode o modelo pelo `python/dump_cols.py`
para ver o tensor cru e comparar com o que o decoder espera.

---

## Licenças

| componente | licença |
|---|---|
| este código | MIT — veja [LICENSE](LICENSE) |
| ONNX Runtime | MIT (Microsoft) |
| YOLO26 / Ultralytics | **AGPL-3.0** — uso comercial exige licença da Ultralytics |
| InsightFace buffalo_l | **apenas pesquisa não-comercial** |
| PaddleOCR PP-OCRv6 | Apache-2.0 — uso comercial permitido |

O código deste repositório é MIT e não contém nem redistribui modelo nenhum:
os `.onnx` são baixados pelo passo a passo acima e ficam fora do versionamento.
As licenças de YOLO26 e InsightFace se aplicam aos **pesos**, não a este código,
e restringem uso comercial. A arquitetura isola cada modelo atrás de um decoder,
então trocá-los é uma unit — mas convém decidir isso antes de construir em cima.
