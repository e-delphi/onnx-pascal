# ONNXDemo — arquitetura interna

> Para instalar, obter os modelos e usar, veja o [README da raiz](../README.md).
> Este documento é sobre como o código está organizado e por quê.

---

## Camadas

```
ONNXDemo.dpr              composition root: só amarra as peças
src/
  ONNX.CApi.pas           ABI da C API 1.28 (tabela de 424 slots, tipada)
  ONNX.Types.pas          interfaces e tipos compartilhados da camada ONNX
  ONNX.Runtime.pas        DLL + OrtEnv + allocator (IONNXRuntime, IOrtCore)
  ONNX.Session.pas        execução do grafo, N inputs/N outputs, metadados

  Vision.Types.pas        vocabulário do domínio (caixa, keypoint, máscara…)
  Vision.Image.pas        buffer RGB + carregador VCL + resample bilinear
  Vision.Preprocess.pas   IImagePreprocessor: letterbox, resize+crop
  Vision.Nms.pas          NMS axis-aligned e rotacionado (interseção real)
  Vision.Model.pas        TModelSpec: metadados + heurística de shape

  Vision.Decoder.pas          IResultDecoder, registro, base comum
  Vision.Decoder.Classify.pas
  Vision.Decoder.Detect.pas
  Vision.Decoder.Segment.pas
  Vision.Decoder.Pose.pas
  Vision.Decoder.Obb.pas
  Vision.Decoder.Scrfd.pas    detecção facial (9 saídas, 3 strides)
  Vision.Decoder.Embed.pas    [1,512] + normalização L2

  Vision.Predictor.pas    orquestrador + fábrica
  Vision.Face.Align.pas   Procrustes de 5 pontos + warp afim
  Vision.Face.pas         encadeia detector + alinhador + embedder
  Vision.Embedding.pas    vetores, cosseno e galeria em arquivo

  Vision.Render.pas       desenho de caixas, esqueleto, máscara, OBB
  Vision.Report.pas       saída de texto (IResultReporter)
  Vision.Report.Face.pas  saída de texto do modo facial (IFaceReporter)
  App.Options.pas         linha de comando
```

## Como o SOLID se aplica aqui

- **SRP** — cada unit tem um motivo para mudar. `ONNX.Session` não sabe o que é
  uma caixa; `Vision.Render` não sabe o que é um tensor; o `.dpr` não contém
  nenhuma regra de modelo.
- **OCP** — adicionar uma cabeça nova é criar uma unit que se registra no
  `initialization` e incluí-la no `uses` do projeto. Nenhum `case` existente
  precisa ser tocado.
- **LSP** — todos os decoders honram `IResultDecoder`; o predictor funciona
  igual com qualquer um.
- **ISP** — o runtime expõe `IONNXRuntime` para a aplicação e `IOrtCore`
  (api/env/allocator/check) só para a sessão. Da mesma forma, `IFaceReporter`
  é separado de `IResultReporter`.
- **DIP** — `TVisionPredictor` recebe sessão, preprocessor, decoder e loader
  pelo construtor. A escolha concreta vive só em `TVisionPredictorFactory`.

Dois episódios em que isso pagou:

- Ao descobrir que a cabeça OBB emite `cx,cy,w,h` e não `x1,y1,x2,y2`, a
  correção foram **4 linhas em 1 arquivo**. Nenhum outro decoder, nem o
  predictor, nem a sessão, nem o renderizador foram tocados.
- O pipeline facial encadeia **dois modelos**, mas não exigiu mecânica nova:
  detector e embedder são dois `IVisionPredictor` comuns com um `IFaceAligner`
  no meio. A normalização do ArcFace `(x−127.5)/127.5` caiu exata no
  `TCropClassifierPreprocessor` que já existia, com média 0,5 e desvio 0,5.

## Detecção de formato em tempo de execução

Nada de configuração: o layout é decidido pelo shape real do tensor.

| shape observado | interpretação |
|---|---|
| `[1, 300, 6]` | detect end-to-end (YOLO26 NMS-free) |
| `[1, 84, 8400]` | detect cru → NMS |
| `[1, N, 85]` | YOLOv5, com coluna de objectness |
| `[1, 300, 38]` + `[1,32,160,160]` | segment end-to-end |
| `[1, 300, 57]` | pose end-to-end (6 + 17×3) |
| `[1, 300, 7]` | obb end-to-end |
| 9× rank 2 com colunas 1/4/10 | SCRFD facial |
| `[1, nc]` | classify |

A distinção cru × decodificado sai da orientação do tensor: saída crua é sempre
canais-primeiro (`dim1 < dim2`), decodificada é linhas-primeiro. Uma verificação
extra exige que as linhas caibam em `max_det`, para descartar o caso degenerado
de um modelo cru com pouquíssimas classes.

**Formatos que diferem entre cabeças** — todos verificados contra a saída real,
não deduzidos da documentação:

- `detect`, `segment` e `pose` emitem a caixa em **cantos** `x1,y1,x2,y2`
- `obb` emite em **centro** `cx,cy,w,h`. Faz sentido: para uma caixa rotacionada
  os cantos alinhados ao eixo não descrevem a geometria.
- SCRFD emite bbox como **distâncias** `l,t,r,b` do centro da âncora, em
  unidades de stride, e cola a imagem no **canto superior esquerdo** do
  letterbox (o YOLO centraliza).

## Reconhecimento facial

O ArcFace não aceita um recorte qualquer: exige a face levada a um template
canônico de 112×112 por transformada de similaridade sobre 5 pontos. Sem isso a
acurácia despenca — é a causa mais comum de números irreproduzíveis com esses
modelos.

A transformada tem 4 graus de liberdade, então 5 pares de pontos
sobredeterminam o sistema e a solução de mínimos quadrados tem forma fechada
(Procrustes), sem SVD. Conferido contra o Umeyama de referência: diferença
máxima de 9,5e-7 nos coeficientes, recortes idênticos pixel a pixel e cosseno
1,0 entre os embeddings. A forma fechada ainda tem a vantagem de **não poder
produzir reflexão**, que para rosto nunca é desejada.

Separação medida com `buffalo_l`: mesma pessoa ≈ 0,78, pessoas diferentes ≈
0,00 — margem de 0,80, daí o limiar padrão de 0,40.

> Mais landmarks **não** melhoram isso. Com 4 graus de liberdade, 5 pontos já
> sobredeterminam a transformada; 106 pontos só redistribuiriam o ajuste para
> regiões que o template canônico nem define. Pontos extras servem para outras
> coisas — pose de cabeça, malha 3D, recorte preciso.

## Correções em relação à versão original

- `SetLength(AverageOutput, 1000)` era fixo: estourava o array em qualquer
  modelo que não tivesse exatamente 1000 classes.
- `StretchDraw` da VCL não interpola; trocado por resample bilinear com
  alinhamento por centro de pixel (mesma convenção do `cv2.resize`).
- Imagens menores que 256 px no lado menor geravam `MaxLeft`/`MaxTop`
  negativos e `ScanLine` fora dos limites.
- PNG/GIF com transparência eram desenhados sobre lixo de memória.
- Logits eram impressos como se fossem score; agora há softmax quando a saída
  não é um vetor de probabilidade.
- `224`, `1000` e as constantes de normalização estavam hard-coded no `.dpr`.
- A sessão suportava um input e um output só — segmentação era impossível.
- Saídas `INT64`/`FLOAT16`/`INT32` causavam leitura de lixo.
- Os handles ORT em `Run` vazavam se o `Run` falhasse no meio.
- Os `Cast*` por slot numérico viraram uma tabela tipada resolvida uma vez,
  conferida contra `onnxruntime_c_api.h` 1.28.1 (424 slots).
- `ExtractFilePath` trunca no lugar errado quando o caminho vem com `/`: a
  imagem anotada ia parar na pasta do executável.
- `%+.4f` não existe no `Format` do Delphi (o `+` é do printf do C) e fazia o
  parser abortar, imprimindo a linha truncada sem o número.

## Limitações conhecidas

- **Batch 1.** `TPredictionView` rejeita explicitamente batch > 1.
- **CPU apenas.** Os slots `SessionOptionsAppendExecutionProvider_*` já estão
  mapeados em `ONNX.CApi.pas` para quem quiser adicionar GPU.
- A galeria faz busca linear. Suficiente para centenas de rostos; acima disso
  convém um índice.
