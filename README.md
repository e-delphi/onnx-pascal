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

Os passos 1–3 já produzem um executável funcionando. Os passos 4–6 baixam os
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

### 7. Imagens de teste

```bash
mkdir -p delphi/bin/imagem && cd delphi/bin/imagem
```

```bash
curl -L -O https://ultralytics.com/images/bus.jpg -O https://ultralytics.com/images/zidane.jpg -O https://ultralytics.com/images/boats.jpg
```

```bash
curl -L -O https://github.com/ageitgey/face_recognition/raw/master/examples/obama.jpg -O https://github.com/ageitgey/face_recognition/raw/master/examples/obama2.jpg -O https://github.com/ageitgey/face_recognition/raw/master/examples/biden.jpg
```

`bus.jpg` (ônibus + 4 pessoas), `zidane.jpg` (2 rostos), `boats.jpg` (marina
aérea, para OBB) e o trio Obama/Biden (duas fotos da mesma pessoa + uma de
outra, para validar reconhecimento facial).

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

A tarefa é lida dos metadados do `.onnx` e o layout da saída é decidido pelo
shape real do tensor em tempo de execução — não há nada a configurar. Use
`--task` só para forçar.

As imagens anotadas vão para `saida/`, nunca junto das entradas.

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

O `dog.jpg` é a imagem canônica do PyTorch, cuja resposta documentada é
*Samoyed*: se der outra coisa, o pré-processamento está errado em algum ponto.

Os valores de referência do pipeline facial (caixa, os 5 landmarks e as
primeiras posições do vetor), gerados pela implementação Python, estão em
`python/referencia_face.json`.

---

## Estrutura

```
.
├── delphi/
│   ├── ONNXDemo.dpr          composition root: só amarra as peças
│   ├── src/                  19 units — ver delphi/README.md
│   ├── run_tests.sh          bateria de regressão
│   └── bin/
│       ├── onnxruntime.dll   (passo 2)
│       ├── yolo/n|s|m|l|x/   (passo 5) — 5 tarefas por tamanho
│       ├── face/             (passo 6)
│       ├── imagem/           entradas (passo 7)
│       └── saida/            tudo que o programa gera
└── python/
    ├── export_models.py      pesos oficiais → ONNX
    ├── export_raw.py         variantes end2end=False
    ├── face_ref.py           referência do pipeline facial
    ├── verificar_face.py     confere a referência facial ponta a ponta
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
`--task=detect` (ou `segment`, `pose`, `obb`, `classify`).

**`Nenhuma saida com formato de predicao (rank 2 ou 3)`**
Export com cabeça não decodificada. Reexporte com `export_models.py`.

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

O código deste repositório é MIT e não contém nem redistribui modelo nenhum:
os `.onnx` são baixados pelo passo a passo acima e ficam fora do versionamento.
As licenças das duas últimas linhas se aplicam aos **pesos**, não a este código,
e restringem uso comercial. A arquitetura isola cada modelo atrás de um decoder,
então trocá-los é uma unit — mas convém decidir isso antes de construir em cima.
