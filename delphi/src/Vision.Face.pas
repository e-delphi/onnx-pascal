unit Vision.Face;

{
  Pipeline de reconhecimento facial: detectar -> alinhar -> embutir.

  E o primeiro caso do projeto que encadeia dois modelos. Em vez de inventar
  uma mecanica nova, compoe dois IVisionPredictor - o detector SCRFD e o
  embedder ArcFace sao ambos predictors comuns - com um IFaceAligner no meio.
  Toda a plumbing de sessao, preprocessamento e decode continua reaproveitada.

  A normalizacao do ArcFace, (x - 127.5) / 127.5, cai exatamente no
  TCropClassifierPreprocessor existente com media 0,5 e desvio 0,5 sobre a
  faixa 0..1, com razao de recorte 1,0 (o recorte alinhado ja sai 112x112,
  entao o redimensionamento e identidade).
}

interface

uses
  System.SysUtils,
  System.Math,
  System.Generics.Collections,
  ONNX.Types,
  Vision.Types,
  Vision.Image,
  Vision.Model,
  Vision.Preprocess,
  Vision.Decoder,
  Vision.Predictor,
  Vision.Face.Align,
  Vision.Embedding;

type
  EFaceError = class(Exception);

  TDetectedFace = record
    Detection: TDetection;
    Embedding: TEmbedding;
    Aligned: IImage;
    function BoxArea: Single;
  end;

  TDetectedFaces = TArray<TDetectedFace>;

  TFaceOptions = record
    DetectThreshold: Single;
    NmsThreshold: Single;
    MaxFaces: Integer;
    { Rostos menores que isto (lado da caixa, em pixels da imagem original)
      sao descartados: recorte pequeno demais vira embedding sem valor. }
    MinFaceSize: Integer;
    KeepAligned: Boolean;
    class function Default: TFaceOptions; static;
  end;

  IFaceEncoder = interface
    ['{4A81C6D2-59E3-4B07-9F26-8D3C0B5A7E19}']
    function Encode(const Image: IImage): TDetectedFaces;
    function EncodeFile(const FileName: string): TDetectedFaces;
    { Rosto de maior area, para cadastro em galeria. Levanta erro se a
      imagem nao tiver rosto nenhum. }
    function EncodeSingle(const FileName: string): TDetectedFace;
    function DetectorDescription: string;
    function EmbedderDescription: string;
    function EmbeddingDimension: Integer;
  end;

  TFaceEncoder = class(TInterfacedObject, IFaceEncoder)
  private
    FDetector: IVisionPredictor;
    FEmbedder: IVisionPredictor;
    FAligner: IFaceAligner;
    FLoader: IImageLoader;
    FOptions: TFaceOptions;
  public
    constructor Create(const ADetector, AEmbedder: IVisionPredictor;
      const AAligner: IFaceAligner; const ALoader: IImageLoader;
      const AOptions: TFaceOptions);

    function Encode(const Image: IImage): TDetectedFaces;
    function EncodeFile(const FileName: string): TDetectedFaces;
    function EncodeSingle(const FileName: string): TDetectedFace;
    function DetectorDescription: string;
    function EmbedderDescription: string;
    function EmbeddingDimension: Integer;
  end;

  TFaceEncoderFactory = class
  public
    class function Build(const Runtime: IONNXRuntime;
      const DetectorPath, EmbedderPath: string;
      const Options: TFaceOptions;
      const SessionConfig: TSessionConfig): IFaceEncoder; static;
  end;

implementation

{ TDetectedFace }

function TDetectedFace.BoxArea: Single;
begin
  Result := Detection.Box.Area;
end;

{ TFaceOptions }

class function TFaceOptions.Default: TFaceOptions;
begin
  Result.DetectThreshold := 0.5;
  Result.NmsThreshold := 0.4;
  Result.MaxFaces := 0;
  Result.MinFaceSize := 24;
  Result.KeepAligned := False;
end;

{ TFaceEncoder }

constructor TFaceEncoder.Create(const ADetector, AEmbedder: IVisionPredictor;
  const AAligner: IFaceAligner; const ALoader: IImageLoader;
  const AOptions: TFaceOptions);
begin
  inherited Create;

  if ADetector = nil then
    raise EArgumentNilException.Create('Detector de rostos nao informado');
  if AEmbedder = nil then
    raise EArgumentNilException.Create('Modelo de embedding nao informado');
  if AAligner = nil then
    raise EArgumentNilException.Create('Alinhador nao informado');
  if ALoader = nil then
    raise EArgumentNilException.Create('Carregador de imagem nao informado');

  FDetector := ADetector;
  FEmbedder := AEmbedder;
  FAligner := AAligner;
  FLoader := ALoader;
  FOptions := AOptions;
end;

function TFaceEncoder.DetectorDescription: string;
begin
  Result := FDetector.DecoderDescription;
end;

function TFaceEncoder.EmbedderDescription: string;
begin
  Result := FEmbedder.DecoderDescription;
end;

function TFaceEncoder.EmbeddingDimension: Integer;
begin
  Result := Integer(FEmbedder.Session.OutputInfo(0).Dim(1));
end;

function TFaceEncoder.Encode(const Image: IImage): TDetectedFaces;
var
  Detected: TVisionResult;
  Faces: TList<TDetectedFace>;
  Face: TDetectedFace;
  Aligned: IImage;
  Embedded: TVisionResult;
  I: Integer;
  Side: Single;
begin
  if Image = nil then
    raise EArgumentNilException.Create('Imagem nao informada');

  Detected := FDetector.Predict(Image);

  Faces := TList<TDetectedFace>.Create;
  try
    for I := 0 to High(Detected.Detections) do
    begin
      if (FOptions.MaxFaces > 0) and (Faces.Count >= FOptions.MaxFaces) then
        Break;

      Face := Default(TDetectedFace);
      Face.Detection := Detected.Detections[I];

      Side := Min(Face.Detection.Box.Width, Face.Detection.Box.Height);
      if Side < FOptions.MinFaceSize then
        Continue;

      if not Face.Detection.HasKeypoints then
        raise EFaceError.Create(
          'O detector nao devolveu landmarks. O alinhamento do ArcFace ' +
          'depende dos 5 pontos; um detector so de caixa nao serve.');

      Aligned := FAligner.Align(Image, Face.Detection.Keypoints);

      Embedded := FEmbedder.Predict(Aligned);
      Face.Embedding := TEmbedding.FromArray(Embedded.Embedding);

      if FOptions.KeepAligned then
        Face.Aligned := Aligned;

      Faces.Add(Face);
    end;

    Result := Faces.ToArray;
  finally
    Faces.Free;
  end;
end;

function TFaceEncoder.EncodeFile(const FileName: string): TDetectedFaces;
begin
  Result := Encode(FLoader.Load(FileName));
end;

function TFaceEncoder.EncodeSingle(const FileName: string): TDetectedFace;
var
  Faces: TDetectedFaces;
  I, Best: Integer;
begin
  Faces := EncodeFile(FileName);

  if Length(Faces) = 0 then
    raise EFaceError.CreateFmt('Nenhum rosto encontrado em %s',
      [ExtractFileName(FileName)]);

  Best := 0;
  for I := 1 to High(Faces) do
    if Faces[I].BoxArea > Faces[Best].BoxArea then
      Best := I;

  Result := Faces[Best];
end;

{ TFaceEncoderFactory }

class function TFaceEncoderFactory.Build(const Runtime: IONNXRuntime;
  const DetectorPath, EmbedderPath: string; const Options: TFaceOptions;
  const SessionConfig: TSessionConfig): IFaceEncoder;
var
  DetectorSession, EmbedderSession: IONNXSession;
  DetectorSpec, EmbedderSpec: TModelSpec;
  DetectorOptions, EmbedderOptions: TPredictorOptions;
  Detector, Embedder: IVisionPredictor;
begin
  if Runtime = nil then
    raise EArgumentNilException.Create('Runtime ONNX nao informado');

  DetectorSession := Runtime.CreateSession(DetectorPath, SessionConfig);
  DetectorSpec := TModelSpecReader.Read(DetectorSession, nil);
  // O SCRFD nao traz metadados e as 9 saidas rank 2 nao permitem deduzir a
  // tarefa: informamos explicitamente.
  DetectorSpec.Task := vtFace;
  DetectorSpec.TaskFromMetadata := False;

  DetectorOptions := TPredictorOptions.Default;
  DetectorOptions.ConfidenceThreshold := Options.DetectThreshold;
  DetectorOptions.ConfidenceWasSet := True;
  DetectorOptions.IoUThreshold := Options.NmsThreshold;
  DetectorOptions.MaxDetections := 0;

  Detector := TVisionPredictor.Create(
    DetectorSession, DetectorSpec,
    TLetterboxPreprocessor.Scrfd,
    TDecoderRegistry.CreateFor(vtFace),
    TVclImageLoader.Create,
    DetectorOptions);

  EmbedderSession := Runtime.CreateSession(EmbedderPath, SessionConfig);
  EmbedderSpec := TModelSpecReader.Read(EmbedderSession, nil);
  EmbedderSpec.Task := vtEmbed;
  EmbedderSpec.TaskFromMetadata := False;

  EmbedderOptions := TPredictorOptions.Default;
  EmbedderOptions.ConfidenceThreshold := 0;
  EmbedderOptions.ConfidenceWasSet := True;

  // (v/255 - 0,5) / 0,5 = (v - 127,5) / 127,5, que e a normalizacao do
  // ArcFace. Razao 1,0 porque o recorte ja chega no tamanho da rede.
  Embedder := TVisionPredictor.Create(
    EmbedderSession, EmbedderSpec,
    TCropClassifierPreprocessor.Create(
      TArray<Single>.Create(0.5, 0.5, 0.5),
      TArray<Single>.Create(0.5, 0.5, 0.5),
      1.0, False),
    TDecoderRegistry.CreateFor(vtEmbed),
    TVclImageLoader.Create,
    EmbedderOptions);

  Result := TFaceEncoder.Create(Detector, Embedder, TArcFaceAligner.Create,
    TVclImageLoader.Create, Options);
end;

end.
