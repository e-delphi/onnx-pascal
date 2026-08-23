unit Vision.Decoder;

{
  Contrato dos decodificadores de saida e registro de implementacoes.

  Cada cabeca de rede (classify, detect, segment, pose, obb) tem um decoder
  proprio. O registro e populado no initialization de cada unit concreta:
  para dar suporte a uma cabeca nova basta criar a unit e coloca-la na
  clausula uses do projeto - nenhum arquivo existente precisa mudar
  (Open/Closed).

  A escolha "saida crua" x "saida ja decodificada" (YOLO26 end2end) e feita
  aqui pelo formato real do tensor devolvido pelo Run, nao por configuracao.
}

interface

uses
  System.SysUtils,
  System.Math,
  System.Generics.Collections,
  ONNX.Types,
  Vision.Types,
  Vision.Model,
  Vision.Preprocess;

const
  { Limite de linhas plausivel para uma saida ja filtrada (max_det do
    Ultralytics e 300). Acima disso o tensor so pode ser saida crua. }
  MAX_END_TO_END_ROWS = 1024;

type
  EDecodeError = class(Exception);

  { Acesso uniforme a um tensor de predicao [1, A, B], escondendo se o
    modelo emite canais primeiro ([1, 4+nc, 8400]) ou linhas primeiro
    ([1, 300, 6]). }
  TPredictionView = record
  private
    FData: TArray<Single>;
    FAnchors: Integer;
    FChannels: Integer;
    FAnchorsFirst: Boolean;
    FValid: Boolean;
  public
    class function FromTensor(const Tensor: TTensor): TPredictionView; static;
    function Valid: Boolean;
    function Anchors: Integer;
    function Channels: Integer;
    function AnchorsFirst: Boolean;
    function Value(Anchor, Channel: Integer): Single;
    function LooksDecoded(ExpectedChannels: Integer): Boolean;
  end;

  TDecodeContext = record
    Spec: TModelSpec;
    Transform: TGeometryTransform;
    ConfidenceThreshold: Single;
    IoUThreshold: Single;
    MaxDetections: Integer;
    MaskThreshold: Single;
    TopK: Integer;
    ClassAgnosticNms: Boolean;
  end;

  IResultDecoder = interface
    ['{7A3C9E52-0B14-4D6A-8E77-2F5B1C4D9A08}']
    function Task: TVisionTask;
    function Describe: string;
    function Decode(const Outputs: TTensorArray;
      const Context: TDecodeContext): TVisionResult;
  end;

  TDecoderFactory = reference to function: IResultDecoder;

  TDecoderRegistry = class
  strict private
    class var FFactories: TDictionary<TVisionTask, TDecoderFactory>;
  public
    class constructor Create;
    class destructor Destroy;
    class procedure RegisterDecoder(Task: TVisionTask; const Factory: TDecoderFactory);
    class function IsSupported(Task: TVisionTask): Boolean;
    class function CreateFor(Task: TVisionTask): IResultDecoder;
    class function SupportedTasks: TArray<TVisionTask>;
  end;

  { Comportamento comum a todas as cabecas derivadas de deteccao. }
  TDetectionDecoderBase = class(TInterfacedObject)
  protected
    function FindPredictionTensor(const Outputs: TTensorArray): TTensor;
    function TryFindProtoTensor(const Outputs: TTensorArray;
      out Proto: TTensor): Boolean;

    function BoxFromCenterChannels(const View: TPredictionView; Anchor: Integer;
      const Context: TDecodeContext): TBoxF;
    function BoxFromCornerChannels(const View: TPredictionView; Anchor: Integer;
      const Context: TDecodeContext): TBoxF;

    function BestClass(const View: TPredictionView; Anchor, FirstChannel,
      ClassCount: Integer; out ClassId: Integer; out Score: Single): Boolean;

    procedure FillClassName(var Detection: TDetection; const Context: TDecodeContext);
    function FinalizeDetections(const Input: TDetections;
      const Context: TDecodeContext; Rotated: Boolean): TDetections;
  end;

implementation

uses
  Vision.Nms;

{ TPredictionView }

class function TPredictionView.FromTensor(const Tensor: TTensor): TPredictionView;
var
  D1, D2: Integer;
begin
  Result := Default(TPredictionView);
  Result.FData := Tensor.Data;

  case Tensor.Rank of
    2:
      begin
        D1 := Tensor.DimAsInt(0);
        D2 := Tensor.DimAsInt(1);
      end;
    3:
      begin
        if Tensor.Dim(0) <> 1 then
          raise EDecodeError.CreateFmt(
            'Somente batch 1 e suportado; recebido shape %s', [Tensor.ShapeText]);
        D1 := Tensor.DimAsInt(1);
        D2 := Tensor.DimAsInt(2);
      end;
  else
    Exit;
  end;

  if (D1 <= 0) or (D2 <= 0) then
    Exit;

  if D1 >= D2 then
  begin
    Result.FAnchorsFirst := True;
    Result.FAnchors := D1;
    Result.FChannels := D2;
  end
  else
  begin
    Result.FAnchorsFirst := False;
    Result.FChannels := D1;
    Result.FAnchors := D2;
  end;

  Result.FValid := Length(Result.FData) >= Result.FAnchors * Result.FChannels;
end;

function TPredictionView.Valid: Boolean;
begin
  Result := FValid;
end;

function TPredictionView.Anchors: Integer;
begin
  Result := FAnchors;
end;

function TPredictionView.Channels: Integer;
begin
  Result := FChannels;
end;

function TPredictionView.AnchorsFirst: Boolean;
begin
  Result := FAnchorsFirst;
end;

function TPredictionView.Value(Anchor, Channel: Integer): Single;
begin
  if FAnchorsFirst then
    Result := FData[Anchor * FChannels + Channel]
  else
    Result := FData[Channel * FAnchors + Anchor];
end;

function TPredictionView.LooksDecoded(ExpectedChannels: Integer): Boolean;
begin
  Result := FValid and FAnchorsFirst and
            (FChannels = ExpectedChannels) and
            (FAnchors <= MAX_END_TO_END_ROWS);
end;

{ TDecoderRegistry }

class constructor TDecoderRegistry.Create;
begin
  FFactories := TDictionary<TVisionTask, TDecoderFactory>.Create;
end;

class destructor TDecoderRegistry.Destroy;
begin
  FFactories.Free;
end;

class procedure TDecoderRegistry.RegisterDecoder(Task: TVisionTask;
  const Factory: TDecoderFactory);
begin
  FFactories.AddOrSetValue(Task, Factory);
end;

class function TDecoderRegistry.IsSupported(Task: TVisionTask): Boolean;
begin
  Result := FFactories.ContainsKey(Task);
end;

class function TDecoderRegistry.CreateFor(Task: TVisionTask): IResultDecoder;
var
  Factory: TDecoderFactory;
begin
  if not FFactories.TryGetValue(Task, Factory) then
    raise EDecodeError.CreateFmt(
      'Nenhum decoder registrado para a tarefa "%s"', [VisionTaskToString(Task)]);
  Result := Factory();
end;

class function TDecoderRegistry.SupportedTasks: TArray<TVisionTask>;
begin
  Result := FFactories.Keys.ToArray;
end;

{ TDetectionDecoderBase }

function TDetectionDecoderBase.FindPredictionTensor(
  const Outputs: TTensorArray): TTensor;
var
  I, BestIndex: Integer;
begin
  BestIndex := -1;
  for I := 0 to High(Outputs) do
  begin
    if (Outputs[I].Rank <> 2) and (Outputs[I].Rank <> 3) then
      Continue;
    if (BestIndex < 0) or
       (Outputs[I].ElementCount > Outputs[BestIndex].ElementCount) then
      BestIndex := I;
  end;

  if BestIndex < 0 then
    raise EDecodeError.Create(
      'Nenhuma saida com formato de predicao (rank 2 ou 3) foi encontrada');

  Result := Outputs[BestIndex];
end;

function TDetectionDecoderBase.TryFindProtoTensor(const Outputs: TTensorArray;
  out Proto: TTensor): Boolean;
var
  I: Integer;
begin
  for I := 0 to High(Outputs) do
    if Outputs[I].Rank = 4 then
    begin
      Proto := Outputs[I];
      Exit(True);
    end;

  Proto := Default(TTensor);
  Result := False;
end;

function TDetectionDecoderBase.BoxFromCenterChannels(const View: TPredictionView;
  Anchor: Integer; const Context: TDecodeContext): TBoxF;
var
  CX, CY, W, H: Single;
begin
  CX := Context.Transform.NetToSourceX(View.Value(Anchor, 0));
  CY := Context.Transform.NetToSourceY(View.Value(Anchor, 1));
  W := Context.Transform.NetToSourceLength(View.Value(Anchor, 2));
  H := Context.Transform.NetToSourceLength(View.Value(Anchor, 3));
  Result := TBoxF.FromCenter(CX, CY, W, H);
end;

function TDetectionDecoderBase.BoxFromCornerChannels(const View: TPredictionView;
  Anchor: Integer; const Context: TDecodeContext): TBoxF;
begin
  Result := TBoxF.FromLTRB(
    Context.Transform.NetToSourceX(View.Value(Anchor, 0)),
    Context.Transform.NetToSourceY(View.Value(Anchor, 1)),
    Context.Transform.NetToSourceX(View.Value(Anchor, 2)),
    Context.Transform.NetToSourceY(View.Value(Anchor, 3)));
end;

function TDetectionDecoderBase.BestClass(const View: TPredictionView;
  Anchor, FirstChannel, ClassCount: Integer; out ClassId: Integer;
  out Score: Single): Boolean;
var
  C: Integer;
  V: Single;
begin
  ClassId := -1;
  Score := 0;

  for C := 0 to ClassCount - 1 do
  begin
    V := View.Value(Anchor, FirstChannel + C);
    if (ClassId < 0) or (V > Score) then
    begin
      ClassId := C;
      Score := V;
    end;
  end;

  Result := ClassId >= 0;
end;

procedure TDetectionDecoderBase.FillClassName(var Detection: TDetection;
  const Context: TDecodeContext);
begin
  Detection.ClassName := Context.Spec.ClassNameOf(Detection.ClassId);
end;

function TDetectionDecoderBase.FinalizeDetections(const Input: TDetections;
  const Context: TDecodeContext; Rotated: Boolean): TDetections;
var
  Mode: TNmsMode;
  Limit: Integer;
begin
  if Context.ClassAgnosticNms then
    Mode := nmClassAgnostic
  else
    Mode := nmClassAware;

  Limit := Context.MaxDetections;

  if Rotated then
    Result := ApplyRotatedNms(Input, Context.IoUThreshold, Mode, Limit)
  else
    Result := ApplyNms(Input, Context.IoUThreshold, Mode, Limit);
end;

end.
