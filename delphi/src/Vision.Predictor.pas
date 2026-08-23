unit Vision.Predictor;

{
  Orquestrador do pipeline.

  Nao contem regra de nenhum modelo especifico: apenas amarra
  preprocessor -> sessao -> decoder e mede o tempo de cada etapa.
  Todas as tres pecas chegam pelo construtor (injecao de dependencia), o que
  torna o predictor testavel com dublês e imune a mudanca de modelo.

  TVisionPredictorFactory concentra a decisao de qual preprocessor e qual
  decoder combinam com o modelo carregado.
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
  Vision.Model,
  Vision.Preprocess,
  Vision.Decoder;

type
  TNormalizationChoice = (ncAuto, ncImageNet, ncUnit);

  TPredictorOptions = record
    ConfidenceThreshold: Single;
    ConfidenceWasSet: Boolean;
    IoUThreshold: Single;
    MaxDetections: Integer;
    MaskThreshold: Single;
    TopK: Integer;
    ClassAgnosticNms: Boolean;
    TaskOverride: TVisionTask;
    LabelsPath: string;
    MultiCropClassify: Boolean;
    Normalization: TNormalizationChoice;
    class function Default: TPredictorOptions; static;
  end;

  IVisionPredictor = interface
    ['{5B8D2F60-91A4-47C3-8E15-0D6C3A7B4E29}']
    function Spec: TModelSpec;
    function Session: IONNXSession;
    function PreprocessorDescription: string;
    function DecoderDescription: string;
    function Predict(const Image: IImage): TVisionResult;
    function PredictFile(const FileName: string): TVisionResult;
  end;

  TVisionPredictor = class(TInterfacedObject, IVisionPredictor)
  private
    FSession: IONNXSession;
    FSpec: TModelSpec;
    FPreprocessor: IImagePreprocessor;
    FDecoder: IResultDecoder;
    FLoader: IImageLoader;
    FOptions: TPredictorOptions;
    function BuildContext(const Transform: TGeometryTransform): TDecodeContext;
    function MergeResults(const Partial: TArray<TVisionResult>): TVisionResult;
  public
    constructor Create(const ASession: IONNXSession; const ASpec: TModelSpec;
      const APreprocessor: IImagePreprocessor; const ADecoder: IResultDecoder;
      const ALoader: IImageLoader; const AOptions: TPredictorOptions);

    function Spec: TModelSpec;
    function Session: IONNXSession;
    function PreprocessorDescription: string;
    function DecoderDescription: string;
    function Predict(const Image: IImage): TVisionResult;
    function PredictFile(const FileName: string): TVisionResult;
  end;

  TVisionPredictorFactory = class
  public
    class function ChoosePreprocessor(const ASpec: TModelSpec;
      const AOptions: TPredictorOptions): IImagePreprocessor; static;
    class function Build(const Runtime: IONNXRuntime; const ModelPath: string;
      const Options: TPredictorOptions;
      const SessionConfig: TSessionConfig): IVisionPredictor; static;
  end;

implementation

uses
  Vision.Nms;

{ TPredictorOptions }

class function TPredictorOptions.Default: TPredictorOptions;
begin
  Result.ConfidenceThreshold := 0.25;
  Result.ConfidenceWasSet := False;
  Result.IoUThreshold := 0.45;
  Result.MaxDetections := 300;
  Result.MaskThreshold := 0.5;
  Result.TopK := 5;
  Result.ClassAgnosticNms := False;
  Result.TaskOverride := vtUnknown;
  Result.LabelsPath := '';
  Result.MultiCropClassify := False;
  Result.Normalization := ncAuto;
end;

{ TVisionPredictor }

constructor TVisionPredictor.Create(const ASession: IONNXSession;
  const ASpec: TModelSpec; const APreprocessor: IImagePreprocessor;
  const ADecoder: IResultDecoder; const ALoader: IImageLoader;
  const AOptions: TPredictorOptions);
begin
  inherited Create;

  if ASession = nil then
    raise EArgumentNilException.Create('Sessao ONNX nao informada');
  if APreprocessor = nil then
    raise EArgumentNilException.Create('Preprocessor nao informado');
  if ADecoder = nil then
    raise EArgumentNilException.Create('Decoder nao informado');
  if ALoader = nil then
    raise EArgumentNilException.Create('Carregador de imagem nao informado');

  FSession := ASession;
  FSpec := ASpec;
  FPreprocessor := APreprocessor;
  FDecoder := ADecoder;
  FLoader := ALoader;
  FOptions := AOptions;
end;

function TVisionPredictor.Spec: TModelSpec;
begin
  Result := FSpec;
end;

function TVisionPredictor.Session: IONNXSession;
begin
  Result := FSession;
end;

function TVisionPredictor.PreprocessorDescription: string;
begin
  Result := FPreprocessor.Describe;
end;

function TVisionPredictor.DecoderDescription: string;
begin
  Result := FDecoder.Describe;
end;

function TVisionPredictor.BuildContext(
  const Transform: TGeometryTransform): TDecodeContext;
begin
  Result.Spec := FSpec;
  Result.Transform := Transform;
  Result.ConfidenceThreshold := FOptions.ConfidenceThreshold;
  Result.IoUThreshold := FOptions.IoUThreshold;
  Result.MaxDetections := FOptions.MaxDetections;
  Result.MaskThreshold := FOptions.MaskThreshold;
  Result.TopK := FOptions.TopK;
  Result.ClassAgnosticNms := FOptions.ClassAgnosticNms;
end;

function TVisionPredictor.PredictFile(const FileName: string): TVisionResult;
begin
  Result := Predict(FLoader.Load(FileName));
end;

function TVisionPredictor.Predict(const Image: IImage): TVisionResult;
var
  Partial: TArray<TVisionResult>;
  Prepared: TPreparedInput;
  Transform: TGeometryTransform;
  Inputs, Outputs: TTensorArray;
  Watch: TStopwatch;
  Pass, Passes: Integer;
  PreprocessMs, InferenceMs, PostprocessMs: Double;
begin
  if Image = nil then
    raise EArgumentNilException.Create('Imagem nao informada');

  Passes := Max(1, FPreprocessor.PassCount);
  SetLength(Partial, Passes);
  SetLength(Inputs, 1);

  PreprocessMs := 0;
  InferenceMs := 0;
  PostprocessMs := 0;

  for Pass := 0 to Passes - 1 do
  begin
    Watch := TStopwatch.StartNew;
    Prepared := FPreprocessor.Prepare(Image, Pass,
      FSpec.InputWidth, FSpec.InputHeight, Transform);
    Watch.Stop;
    PreprocessMs := PreprocessMs + Watch.Elapsed.TotalMilliseconds;

    Inputs[0] := TTensor.Create(FSpec.InputName, Prepared.Shape, Prepared.Data);

    Watch := TStopwatch.StartNew;
    Outputs := FSession.Run(Inputs);
    Watch.Stop;
    InferenceMs := InferenceMs + Watch.Elapsed.TotalMilliseconds;

    Watch := TStopwatch.StartNew;
    Partial[Pass] := FDecoder.Decode(Outputs, BuildContext(Transform));
    Watch.Stop;
    PostprocessMs := PostprocessMs + Watch.Elapsed.TotalMilliseconds;
  end;

  Result := MergeResults(Partial);
  Result.ImageWidth := Image.Width;
  Result.ImageHeight := Image.Height;
  Result.PreprocessMs := PreprocessMs;
  Result.InferenceMs := InferenceMs;
  Result.PostprocessMs := PostprocessMs;
end;

function TVisionPredictor.MergeResults(
  const Partial: TArray<TVisionResult>): TVisionResult;
var
  Totals: TDictionary<Integer, Single>;
  Names: TDictionary<Integer, string>;
  Merged: TList<TClassScore>;
  All: TList<TDetection>;
  Item: TClassScore;
  Pair: TPair<Integer, Single>;
  I, J: Integer;
  Accumulated: Single;
  Mode: TNmsMode;
begin
  if Length(Partial) = 0 then
    Exit(Default(TVisionResult));

  if Length(Partial) = 1 then
    Exit(Partial[0]);

  Result := Default(TVisionResult);
  Result.Task := Partial[0].Task;

  if Result.Task = vtClassify then
  begin
    Totals := TDictionary<Integer, Single>.Create;
    Names := TDictionary<Integer, string>.Create;
    Merged := TList<TClassScore>.Create;
    try
      for I := 0 to High(Partial) do
        for J := 0 to High(Partial[I].Classes) do
        begin
          Item := Partial[I].Classes[J];
          if not Totals.TryGetValue(Item.ClassId, Accumulated) then
            Accumulated := 0;
          Totals.AddOrSetValue(Item.ClassId, Accumulated + Item.Score);
          Names.AddOrSetValue(Item.ClassId, Item.ClassName);
        end;

      for Pair in Totals do
      begin
        Item.ClassId := Pair.Key;
        Item.Score := Pair.Value / Length(Partial);
        Names.TryGetValue(Pair.Key, Item.ClassName);
        Merged.Add(Item);
      end;

      Merged.Sort(TComparer<TClassScore>.Construct(
        function(const L, R: TClassScore): Integer
        begin
          if L.Score > R.Score then
            Result := -1
          else if L.Score < R.Score then
            Result := 1
          else
            Result := 0;
        end));

      while (FOptions.TopK > 0) and (Merged.Count > FOptions.TopK) do
        Merged.Delete(Merged.Count - 1);

      Result.Classes := Merged.ToArray;
    finally
      Merged.Free;
      Names.Free;
      Totals.Free;
    end;
    Exit;
  end;

  All := TList<TDetection>.Create;
  try
    for I := 0 to High(Partial) do
      All.AddRange(Partial[I].Detections);

    if FOptions.ClassAgnosticNms then
      Mode := nmClassAgnostic
    else
      Mode := nmClassAware;

    Result.Detections := ApplyNms(All.ToArray, FOptions.IoUThreshold, Mode,
      FOptions.MaxDetections);
  finally
    All.Free;
  end;
end;

{ TVisionPredictorFactory }

class function TVisionPredictorFactory.ChoosePreprocessor(const ASpec: TModelSpec;
  const AOptions: TPredictorOptions): IImagePreprocessor;
begin
  if ASpec.Task <> vtClassify then
    Exit(TLetterboxPreprocessor.Create);

  case AOptions.Normalization of
    ncImageNet:
      Result := TCropClassifierPreprocessor.ImageNet(AOptions.MultiCropClassify);
    ncUnit:
      Result := TCropClassifierPreprocessor.UnitScale(AOptions.MultiCropClassify);
  else
    // Classificadores Ultralytics ja treinam em 0..1; os classificadores do
    // ONNX Model Zoo (SqueezeNet, ResNet, ...) esperam media/desvio ImageNet.
    if ASpec.IsUltralytics then
      Result := TCropClassifierPreprocessor.UnitScale(AOptions.MultiCropClassify)
    else
      Result := TCropClassifierPreprocessor.ImageNet(AOptions.MultiCropClassify);
  end;
end;

class function TVisionPredictorFactory.Build(const Runtime: IONNXRuntime;
  const ModelPath: string; const Options: TPredictorOptions;
  const SessionConfig: TSessionConfig): IVisionPredictor;
var
  Session: IONNXSession;
  Spec: TModelSpec;
  Effective: TPredictorOptions;
  Decoder: IResultDecoder;
  Preprocessor: IImagePreprocessor;
begin
  if Runtime = nil then
    raise EArgumentNilException.Create('Runtime ONNX nao informado');

  Effective := Options;

  Session := Runtime.CreateSession(ModelPath, SessionConfig);
  Spec := TModelSpecReader.Read(Session, LoadLabelsFile(Effective.LabelsPath));

  if Effective.TaskOverride <> vtUnknown then
  begin
    Spec.Task := Effective.TaskOverride;
    Spec.TaskFromMetadata := False;
  end;

  if not TDecoderRegistry.IsSupported(Spec.Task) then
    raise EDecodeError.CreateFmt(
      'Tarefa "%s" nao suportada. Use --task para forcar detect, segment, ' +
      'pose, obb ou classify.', [VisionTaskToString(Spec.Task)]);

  // Em classificacao o limiar padrao de deteccao esconderia o top-K inteiro.
  if (Spec.Task = vtClassify) and (not Effective.ConfidenceWasSet) then
    Effective.ConfidenceThreshold := 0;

  Preprocessor := ChoosePreprocessor(Spec, Effective);
  Decoder := TDecoderRegistry.CreateFor(Spec.Task);

  Result := TVisionPredictor.Create(Session, Spec, Preprocessor, Decoder,
    TVclImageLoader.Create, Effective);
end;

end.
