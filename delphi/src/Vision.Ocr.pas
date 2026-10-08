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
}

interface

uses
  System.SysUtils,
  System.Diagnostics,
  System.Generics.Collections,
  ONNX.Types,
  Vision.Types,
  Vision.Image,
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
    FOptions: TOcrOptions;
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
end;

function TOcrEngine.Read(const Image: IImage): TOcrResult;
var
  Detected, Recognized: TVisionResult;
  List: TList<TOcrLine>;
  Line: TOcrLine;
  Crop: IImage;
  Watch: TStopwatch;
  I: Integer;
begin
  if Image = nil then
    raise EArgumentNilException.Create('Imagem nao informada');

  Result := Default(TOcrResult);
  Result.ImageWidth := Image.Width;
  Result.ImageHeight := Image.Height;

  Detected := FDetector.Predict(Image);
  Result.DetectMs := Detected.TotalMs;

  Watch := TStopwatch.StartNew;
  List := TList<TOcrLine>.Create;
  try
    // Uma linha por vez: batch 1 e o que TPredictionView/decoders aceitam,
    // e evita o padding de largura que um lote imporia as linhas curtas.
    for I := 0 to High(Detected.Detections) do
    begin
      Crop := FCropper.Crop(Image, Detected.Detections[I].Obb);
      if Crop = nil then
        Continue;

      Recognized := FRecognizer.Predict(Crop);
      if Recognized.TextScore < FOptions.RecognitionThreshold then
        Continue;

      Line.Detection := Detected.Detections[I];
      Line.Text := Recognized.Text;
      Line.TextScore := Recognized.TextScore;
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
  Result := Format('%s | %s', [FRecognizer.Spec.Summary,
    FRecognizer.PreprocessorDescription]);
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

  Result := TOcrEngine.Create(Detector, Recognizer, TTextLineCropper.Create,
    Options);
end;

end.
