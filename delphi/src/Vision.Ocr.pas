unit Vision.Ocr;

{
  Pipeline de OCR: detectar linhas -> recortar -> ler.

  Mesma receita do modulo facial: compoe dois IVisionPredictor comuns - o
  detector DBNet (tarefa text) e o reconhecedor CTC (tarefa rec) - com um
  ITextLineCropper no meio. Sessao, preprocessamento e decode de cada
  modelo continuam sendo os do resto do projeto.

  Equivale ao pipeline OCR do PaddleX sem os modulos opcionais (orientacao
  do documento, desentortamento e orientacao de linha), que sao outros
  modelos. Sem o classificador de orientacao de linha, texto de cabeca para
  baixo ou vertical lido "ao contrario" sai como lixo - igual ao original
  com use_textline_orientation=False.

  Leitura em lotes: o PaddleX ordena as linhas pela proporcao
  largura/altura e as le em grupos de RecognitionBatchSize (6 no pipeline
  oficial), preenchendo cada uma com zero ate a mais larga do grupo. Como o
  reconhecedor tem atencao global, esse padding muda a leitura de uma ou
  outra linha: com lote 6 o resultado e o do PaddleOCR de fabrica. Na GPU
  os lotes tambem sao 3x mais rapidos; na CPU o padding custa caro, entao
  la o padrao e 1.
}

interface

uses
  System.SysUtils,
  System.Math,
  System.Diagnostics,
  System.Generics.Collections,
  System.Generics.Defaults,
  ONNX.Types,
  Vision.Types,
  Vision.Image,
  Vision.Preprocess,
  Vision.Predictor,
  Vision.Ocr.Crop;

type
  EOcrError = class(Exception);

  TOcrLine = record
    { Caixa rotacionada (HasObb) e score da deteccao. }
    Detection: TDetection;
    Text: string;
    TextScore: Single;
  end;

  TOcrLines = TArray<TOcrLine>;

  TOcrResult = record
    ImageWidth: Integer;
    ImageHeight: Integer;
    Lines: TOcrLines;
    DetectMs: Double;
    RecognizeMs: Double;
    { Texto completo, uma linha detectada por linha, em ordem de leitura. }
    function FullText: string;
  end;

  TOcrOptions = record
    { Linhas lidas com confianca abaixo disto sao descartadas (o
      text_rec_score_thresh do PaddleX, 0 por padrao). }
    RecognitionThreshold: Single;
    { Linhas lidas por execucao do reconhecedor. 0 = automatico: 6 na GPU
      (o lote do PaddleX), 1 na CPU. }
    RecognitionBatchSize: Integer;
    class function Default: TOcrOptions; static;
  end;

  IOcrEngine = interface
    ['{8B4D2E71-A6C3-4F09-9E58-1C7A3D6B0F42}']
    function Read(const Image: IImage): TOcrResult;
    function DetectorDescription: string;
    function RecognizerDescription: string;
    function DictionarySize: Integer;
  end;

  TOcrEngine = class(TInterfacedObject, IOcrEngine)
  private
    FDetector: IVisionPredictor;
    FRecognizer: IVisionPredictor;
    FCropper: ITextLineCropper;
    FRecPreprocessor: IImagePreprocessor;
    FOptions: TOcrOptions;
    procedure RecognizeBatched(const Crops: TArray<IImage>;
      var Texts: TArray<string>; var Scores: TArray<Single>);
  public
    constructor Create(const ADetector, ARecognizer: IVisionPredictor;
      const ACropper: ITextLineCropper; const AOptions: TOcrOptions);
    function Read(const Image: IImage): TOcrResult;
    function DetectorDescription: string;
    function RecognizerDescription: string;
    function DictionarySize: Integer;
  end;

  TOcrEngineFactory = class
  public
    { DetectorOptions vale para o detector (limiares do DB, --det-side...).
      O reconhecedor recebe o dicionario de DictionaryPath. }
    class function Build(const Runtime: IONNXRuntime;
      const DetectorPath, RecognizerPath, DictionaryPath: string;
      const DetectorOptions: TPredictorOptions; const Options: TOcrOptions;
      const SessionConfig: TSessionConfig): IOcrEngine; static;
  end;

implementation

uses
  Vision.Decoder.Ctc;

{ TOcrResult }

function TOcrResult.FullText: string;
var
  Builder: TStringBuilder;
  I: Integer;
begin
  Builder := TStringBuilder.Create;
  try
    for I := 0 to High(Lines) do
    begin
      if I > 0 then
        Builder.AppendLine;
      Builder.Append(Lines[I].Text);
    end;
    Result := Builder.ToString;
  finally
    Builder.Free;
  end;
end;

{ TOcrOptions }

class function TOcrOptions.Default: TOcrOptions;
begin
  Result.RecognitionThreshold := 0;
  Result.RecognitionBatchSize := 0;
end;

{ TOcrEngine }

constructor TOcrEngine.Create(const ADetector, ARecognizer: IVisionPredictor;
  const ACropper: ITextLineCropper; const AOptions: TOcrOptions);
begin
  inherited Create;
  if ADetector = nil then
    raise EArgumentNilException.Create('Detector de texto nao informado');
  if ARecognizer = nil then
    raise EArgumentNilException.Create('Reconhecedor nao informado');
  if ACropper = nil then
    raise EArgumentNilException.Create('Recortador nao informado');

  FDetector := ADetector;
  FRecognizer := ARecognizer;
  FCropper := ACropper;
  FOptions := AOptions;
  FOptions.RecognitionBatchSize := Max(1, FOptions.RecognitionBatchSize);
  FRecPreprocessor := TTextRecPreprocessor.Create;
end;

procedure TOcrEngine.RecognizeBatched(const Crops: TArray<IImage>;
  var Texts: TArray<string>; var Scores: TArray<Single>);
var
  Order: TArray<Integer>;
  Prepared: TArray<TPreparedInput>;
  Inputs, Outputs: TTensorArray;
  Output: TTensor;
  Data: TArray<Single>;
  Transform: TGeometryTransform;
  First, Count, I, J, Channel, Y, Height, MaxWidth, Width: Integer;
  Steps, Classes: Integer;
  Ratios: TArray<Double>;
begin
  // Ordem do PaddleX: proporcao crescente; o indice desempata (sort estavel).
  SetLength(Order, Length(Crops));
  SetLength(Ratios, Length(Crops));
  for I := 0 to High(Order) do
  begin
    Order[I] := I;
    Ratios[I] := Crops[I].Width / Crops[I].Height;
  end;
  TArray.Sort<Integer>(Order, TComparer<Integer>.Construct(
    function(const L, R: Integer): Integer
    begin
      Result := CompareValue(Ratios[L], Ratios[R]);
      if Result = 0 then
        Result := L - R;
    end));

  Height := FRecognizer.Spec.InputHeight;
  SetLength(Inputs, 1);
  First := 0;
  while First < Length(Order) do
  begin
    Count := Min(FOptions.RecognitionBatchSize, Length(Order) - First);

    SetLength(Prepared, Count);
    MaxWidth := 0;
    for J := 0 to Count - 1 do
    begin
      Prepared[J] := FRecPreprocessor.Prepare(Crops[Order[First + J]], 0, 0,
        Height, Transform);
      MaxWidth := Max(MaxWidth, Integer(Prepared[J].Shape[3]));
    end;

    // [Count, 3, H, MaxWidth]: cada linha colada a esquerda, o resto em zero.
    Data := nil;
    SetLength(Data, Count * 3 * Height * MaxWidth);
    for J := 0 to Count - 1 do
    begin
      Width := Prepared[J].Shape[3];
      for Channel := 0 to 2 do
        for Y := 0 to Height - 1 do
          Move(Prepared[J].Data[(Channel * Height + Y) * Width],
            Data[((J * 3 + Channel) * Height + Y) * MaxWidth],
            Width * SizeOf(Single));
    end;

    Inputs[0] := TTensor.Create(FRecognizer.Spec.InputName,
      TArray<Int64>.Create(Count, 3, Height, MaxWidth), Data);
    Outputs := FRecognizer.Session.Run(Inputs);

    Output := Default(TTensor);
    for I := 0 to High(Outputs) do
      if Outputs[I].Rank = 3 then
        Output := Outputs[I];
    if Output.Rank <> 3 then
      raise EOcrError.Create('O reconhecedor nao devolveu a saida CTC [N, T, C]');
    Steps := Output.DimAsInt(1);
    Classes := Output.DimAsInt(2);
    if Classes <> FRecognizer.Spec.ClassCount + 2 then
      raise EOcrError.CreateFmt(
        'Dicionario com %d caracteres nao combina com a saida %s.',
        [FRecognizer.Spec.ClassCount, Output.ShapeText]);

    for J := 0 to Count - 1 do
      CtcGreedyDecode(Output.Data, J, Steps, Classes,
        FRecognizer.Spec.ClassNames, Texts[Order[First + J]],
        Scores[Order[First + J]]);

    Inc(First, Count);
  end;
end;

function TOcrEngine.Read(const Image: IImage): TOcrResult;
var
  Detected, Recognized: TVisionResult;
  Crops: TArray<IImage>;
  Owners: TArray<Integer>;
  Texts: TArray<string>;
  Scores: TArray<Single>;
  List: TList<TOcrLine>;
  Line: TOcrLine;
  Crop: IImage;
  Watch: TStopwatch;
  I, Count: Integer;
begin
  if Image = nil then
    raise EArgumentNilException.Create('Imagem nao informada');

  Result := Default(TOcrResult);
  Result.ImageWidth := Image.Width;
  Result.ImageHeight := Image.Height;

  Detected := FDetector.Predict(Image);
  Result.DetectMs := Detected.TotalMs;

  Watch := TStopwatch.StartNew;

  // Recortes validos e a deteccao de onde cada um veio.
  SetLength(Crops, Length(Detected.Detections));
  SetLength(Owners, Length(Detected.Detections));
  Count := 0;
  for I := 0 to High(Detected.Detections) do
  begin
    Crop := FCropper.Crop(Image, Detected.Detections[I].Obb);
    if Crop = nil then
      Continue;
    Crops[Count] := Crop;
    Owners[Count] := I;
    Inc(Count);
  end;
  SetLength(Crops, Count);
  SetLength(Owners, Count);
  SetLength(Texts, Count);
  SetLength(Scores, Count);

  if FOptions.RecognitionBatchSize > 1 then
    RecognizeBatched(Crops, Texts, Scores)
  else
    for I := 0 to Count - 1 do
    begin
      Recognized := FRecognizer.Predict(Crops[I]);
      Texts[I] := Recognized.Text;
      Scores[I] := Recognized.TextScore;
    end;

  List := TList<TOcrLine>.Create;
  try
    for I := 0 to Count - 1 do
    begin
      if Scores[I] < FOptions.RecognitionThreshold then
        Continue;
      Line.Detection := Detected.Detections[Owners[I]];
      Line.Text := Texts[I];
      Line.TextScore := Scores[I];
      List.Add(Line);
    end;
    Result.Lines := List.ToArray;
  finally
    List.Free;
  end;
  Watch.Stop;
  Result.RecognizeMs := Watch.Elapsed.TotalMilliseconds;
end;

function TOcrEngine.DetectorDescription: string;
begin
  Result := Format('%s | %s', [FDetector.Spec.Summary,
    FDetector.PreprocessorDescription]);
end;

function TOcrEngine.RecognizerDescription: string;
begin
  Result := Format('%s | %s | lote %d', [FRecognizer.Spec.Summary,
    FRecognizer.PreprocessorDescription, FOptions.RecognitionBatchSize]);
end;

function TOcrEngine.DictionarySize: Integer;
begin
  Result := FRecognizer.Spec.ClassCount;
end;

{ TOcrEngineFactory }

class function TOcrEngineFactory.Build(const Runtime: IONNXRuntime;
  const DetectorPath, RecognizerPath, DictionaryPath: string;
  const DetectorOptions: TPredictorOptions; const Options: TOcrOptions;
  const SessionConfig: TSessionConfig): IOcrEngine;
var
  DetOptions, RecOptions: TPredictorOptions;
  Detector, Recognizer: IVisionPredictor;
  Effective: TOcrOptions;
begin
  if Runtime = nil then
    raise EArgumentNilException.Create('Runtime ONNX nao informado');
  if not FileExists(DictionaryPath) then
    raise EOcrError.CreateFmt(
      'Dicionario do reconhecedor nao encontrado: %s'#13#10 +
      'Baixe o inference.yml do mesmo repositorio do modelo, ou informe ' +
      '--ocr-dict=ARQ.', [DictionaryPath]);

  // Nenhum dos dois .onnx traz metadados: a tarefa e forcada aqui.
  DetOptions := DetectorOptions;
  DetOptions.TaskOverride := vtText;
  DetOptions.LabelsPath := '';
  Detector := TVisionPredictorFactory.Build(Runtime, DetectorPath,
    DetOptions, SessionConfig);

  RecOptions := TPredictorOptions.Default;
  RecOptions.TaskOverride := vtTextRec;
  RecOptions.LabelsPath := DictionaryPath;
  Recognizer := TVisionPredictorFactory.Build(Runtime, RecognizerPath,
    RecOptions, SessionConfig);

  // Automatico: o lote oficial na GPU, uma linha por vez na CPU.
  Effective := Options;
  if Effective.RecognitionBatchSize <= 0 then
    if SessionConfig.Provider = epDirectML then
      Effective.RecognitionBatchSize := 6
    else
      Effective.RecognitionBatchSize := 1;

  Result := TOcrEngine.Create(Detector, Recognizer, TTextLineCropper.Create,
    Effective);
end;

end.
