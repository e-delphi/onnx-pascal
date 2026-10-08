unit Vision.Report;

{
  Apresentacao textual dos resultados.

  Isolado para que o pipeline nunca chame Writeln: trocar console por log,
  arquivo ou GUI e implementar IResultReporter de outro jeito.
}

interface

uses
  System.SysUtils,
  System.Math,
  ONNX.Types,
  Vision.Types,
  Vision.Model,
  Vision.Predictor;

type
  IResultReporter = interface
    ['{0A9E4B37-6C25-4F18-9B70-8D3A1E6C5027}']
    procedure ReportRuntime(const Runtime: IONNXRuntime);
    procedure ReportModel(const Predictor: IVisionPredictor);
    procedure ReportResult(const Value: TVisionResult);
  end;

  TConsoleReporter = class(TInterfacedObject, IResultReporter)
  private
    FVerbose: Boolean;
    procedure ReportClassification(const Value: TVisionResult);
    procedure ReportDetections(const Value: TVisionResult);
  public
    constructor Create(AVerbose: Boolean = True);
    procedure ReportRuntime(const Runtime: IONNXRuntime);
    procedure ReportModel(const Predictor: IVisionPredictor);
    procedure ReportResult(const Value: TVisionResult);
  end;

implementation

constructor TConsoleReporter.Create(AVerbose: Boolean);
begin
  inherited Create;
  FVerbose := AVerbose;
end;

procedure TConsoleReporter.ReportRuntime(const Runtime: IONNXRuntime);
var
  Providers: TArray<string>;
begin
  Writeln('ONNX Runtime : ', Runtime.Version);
  if FVerbose then
  begin
    Providers := Runtime.AvailableProviders;
    if Length(Providers) > 0 then
      Writeln('Providers    : ', string.Join(', ', Providers));
  end;
end;

procedure TConsoleReporter.ReportModel(const Predictor: IVisionPredictor);
var
  Session: IONNXSession;
  Spec: TModelSpec;
  I: Integer;
begin
  Session := Predictor.Session;
  Spec := Predictor.Spec;

  Writeln;
  Writeln('--- MODELO ---');
  Writeln('Arquivo      : ', Session.ModelPath);
  if Spec.ProducerName <> '' then
    Writeln('Produtor     : ', Spec.ProducerName);
  Writeln('Descricao    : ', Spec.Summary);
  if Spec.TaskFromMetadata then
    Writeln('Tarefa       : lida dos metadados do .onnx')
  else
    Writeln('Tarefa       : deduzida do formato das saidas');

  Writeln('Preprocess   : ', Predictor.PreprocessorDescription);
  Writeln('Decoder      : ', Predictor.DecoderDescription);

  if not FVerbose then
    Exit;

  Writeln;
  for I := 0 to Session.InputCount - 1 do
    Writeln(Format('Input  %d     : %s', [I, Session.InputInfo(I).ToString]));
  for I := 0 to Session.OutputCount - 1 do
    Writeln(Format('Output %d     : %s', [I, Session.OutputInfo(I).ToString]));
end;

procedure TConsoleReporter.ReportResult(const Value: TVisionResult);
begin
  Writeln;
  Writeln('--- RESULTADO ---');
  Writeln(Format('Imagem       : %dx%d', [Value.ImageWidth, Value.ImageHeight]));
  Writeln(Format('Tempos       : pre %.1f ms | inferencia %.1f ms | pos %.1f ms | total %.1f ms',
    [Value.PreprocessMs, Value.InferenceMs, Value.PostprocessMs, Value.TotalMs]));
  Writeln;

  if Value.Task = vtClassify then
    ReportClassification(Value)
  else if Value.Task = vtTextRec then
    Writeln(Format('Texto lido   : "%s"  (confianca %.2f%%)',
      [Value.Text, Value.TextScore * 100]))
  else
    ReportDetections(Value);
end;

procedure TConsoleReporter.ReportClassification(const Value: TVisionResult);
var
  I: Integer;
begin
  if Length(Value.Classes) = 0 then
  begin
    Writeln('Nenhuma classe acima do limiar.');
    Exit;
  end;

  Writeln(Format('Top %d classes:', [Length(Value.Classes)]));
  for I := 0 to High(Value.Classes) do
    Writeln(Format('  #%d  %-28s  %6.2f%%  (id %d)',
      [I + 1, Value.Classes[I].ClassName, Value.Classes[I].Score * 100,
       Value.Classes[I].ClassId]));
end;

procedure TConsoleReporter.ReportDetections(const Value: TVisionResult);
var
  I, K, Visible: Integer;
  Detection: TDetection;
begin
  if Length(Value.Detections) = 0 then
  begin
    Writeln('Nenhum objeto acima do limiar.');
    Exit;
  end;

  if Value.Task = vtText then
    Writeln(Format('%d linha(s) de texto, em ordem de leitura:',
      [Length(Value.Detections)]))
  else
    Writeln(Format('%d objeto(s):', [Length(Value.Detections)]));
  for I := 0 to High(Value.Detections) do
  begin
    Detection := Value.Detections[I];

    Writeln(Format('  #%-3d %-22s %6.2f%%  caixa %s',
      [I + 1, Detection.DisplayName, Detection.Score * 100,
       Detection.Box.ToString]));

    if Detection.HasObb then
      Writeln(Format('       orientada: centro (%.1f, %.1f)  %.0fx%.0f  angulo %.1f graus',
        [Detection.Obb.CX, Detection.Obb.CY, Detection.Obb.W, Detection.Obb.H,
         RadToDeg(Detection.Obb.Angle)]));

    if Detection.HasKeypoints then
    begin
      Visible := 0;
      for K := 0 to High(Detection.Keypoints) do
        if Detection.Keypoints[K].Score >= 0.5 then
          Inc(Visible);
      Writeln(Format('       keypoints: %d de %d visiveis',
        [Visible, Length(Detection.Keypoints)]));

      if FVerbose then
        for K := 0 to High(Detection.Keypoints) do
          Writeln(Format('         kp%-2d (%7.1f, %7.1f)  conf %.2f',
            [K, Detection.Keypoints[K].X, Detection.Keypoints[K].Y,
             Detection.Keypoints[K].Score]));
    end;

    if Detection.HasMask then
      Writeln(Format('       mascara: %dx%d a partir de (%d, %d)',
        [Detection.Mask.Width, Detection.Mask.Height,
         Detection.Mask.OffsetX, Detection.Mask.OffsetY]));
  end;
end;

end.
