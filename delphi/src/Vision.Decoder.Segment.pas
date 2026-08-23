unit Vision.Decoder.Segment;

{
  Cabeca de segmentacao de instancias.

  O modelo devolve dois tensores:
    - predicoes:  [1, 4 + nc + C, N]  ou  [1, 300, 6 + C] (end-to-end)
    - prototipos: [1, C, mh, mw]

  A mascara de cada instancia e a combinacao linear dos C prototipos pelos
  C coeficientes da linha, seguida de sigmoide. O resultado e recortado pela
  caixa e reamostrado para as coordenadas da imagem original, que e o que o
  renderizador espera.
}

interface

uses
  System.SysUtils,
  System.Math,
  System.Generics.Collections,
  ONNX.Types,
  Vision.Types,
  Vision.Model,
  Vision.Preprocess,
  Vision.Decoder,
  Vision.Decoder.Detect;

type
  TProtoPlanes = record
    Data: TArray<Single>;
    Channels: Integer;
    Width: Integer;
    Height: Integer;
    function IsValid: Boolean;
    class function FromTensor(const Tensor: TTensor): TProtoPlanes; static;
  end;

  TSegmentDecoder = class(TDetectionDecoderBase, IResultDecoder)
  private
    function BuildMaskPlane(const Proto: TProtoPlanes;
      const Coefficients: TArray<Single>): TArray<Single>;
    function CropMaskToBox(const Plane: TArray<Single>; const Proto: TProtoPlanes;
      const Box: TBoxF; const Context: TDecodeContext): TMaskData;
  public
    function Task: TVisionTask;
    function Describe: string;
    function Decode(const Outputs: TTensorArray;
      const Context: TDecodeContext): TVisionResult;
  end;

implementation

{ TProtoPlanes }

class function TProtoPlanes.FromTensor(const Tensor: TTensor): TProtoPlanes;
begin
  Result := Default(TProtoPlanes);
  if Tensor.Rank <> 4 then
    Exit;

  Result.Channels := Tensor.DimAsInt(1);
  Result.Height := Tensor.DimAsInt(2);
  Result.Width := Tensor.DimAsInt(3);
  Result.Data := Tensor.Data;
end;

function TProtoPlanes.IsValid: Boolean;
begin
  Result := (Channels > 0) and (Width > 0) and (Height > 0) and
            (Length(Data) >= Channels * Width * Height);
end;

{ TSegmentDecoder }

function TSegmentDecoder.Task: TVisionTask;
begin
  Result := vtSegment;
end;

function TSegmentDecoder.Describe: string;
begin
  Result := 'segmentacao de instancias (predicoes + prototipos de mascara)';
end;

function TSegmentDecoder.BuildMaskPlane(const Proto: TProtoPlanes;
  const Coefficients: TArray<Single>): TArray<Single>;
var
  PlaneSize, C, I, Base: Integer;
  Coefficient: Single;
begin
  PlaneSize := Proto.Width * Proto.Height;
  SetLength(Result, PlaneSize);

  for C := 0 to Proto.Channels - 1 do
  begin
    if C >= Length(Coefficients) then
      Break;

    Coefficient := Coefficients[C];
    if Coefficient = 0 then
      Continue;

    Base := C * PlaneSize;
    for I := 0 to PlaneSize - 1 do
      Result[I] := Result[I] + Coefficient * Proto.Data[Base + I];
  end;

  for I := 0 to PlaneSize - 1 do
    Result[I] := Sigmoid(Result[I]);
end;

function TSegmentDecoder.CropMaskToBox(const Plane: TArray<Single>;
  const Proto: TProtoPlanes; const Box: TBoxF;
  const Context: TDecodeContext): TMaskData;
var
  X, Y, X0, Y0, X1, Y1: Integer;
  SourceX, SourceY, NetX, NetY, ProtoX, ProtoY, FX, FY, Value: Single;
  ScaleX, ScaleY: Single;
begin
  Result := Default(TMaskData);

  Result.OffsetX := Max(0, Floor(Box.Left));
  Result.OffsetY := Max(0, Floor(Box.Top));
  Result.Width := Max(0, Ceil(Box.Right) - Result.OffsetX);
  Result.Height := Max(0, Ceil(Box.Bottom) - Result.OffsetY);

  if (Result.Width <= 0) or (Result.Height <= 0) then
    Exit;

  SetLength(Result.Values, Result.Width * Result.Height);

  // Rede -> prototipo: os protos tem 1/4 da resolucao da entrada.
  ScaleX := Proto.Width / Context.Transform.NetWidth;
  ScaleY := Proto.Height / Context.Transform.NetHeight;

  for Y := 0 to Result.Height - 1 do
  begin
    SourceY := Result.OffsetY + Y + 0.5;
    NetY := Context.Transform.SourceToNetY(SourceY);
    ProtoY := NetY * ScaleY - 0.5;

    Y0 := Floor(ProtoY);
    FY := ProtoY - Y0;
    Y0 := Min(Max(Y0, 0), Proto.Height - 1);
    Y1 := Min(Y0 + 1, Proto.Height - 1);

    for X := 0 to Result.Width - 1 do
    begin
      SourceX := Result.OffsetX + X + 0.5;
      NetX := Context.Transform.SourceToNetX(SourceX);
      ProtoX := NetX * ScaleX - 0.5;

      X0 := Floor(ProtoX);
      FX := ProtoX - X0;
      X0 := Min(Max(X0, 0), Proto.Width - 1);
      X1 := Min(X0 + 1, Proto.Width - 1);

      Value :=
        Plane[Y0 * Proto.Width + X0] * (1 - FX) * (1 - FY) +
        Plane[Y0 * Proto.Width + X1] * FX * (1 - FY) +
        Plane[Y1 * Proto.Width + X0] * (1 - FX) * FY +
        Plane[Y1 * Proto.Width + X1] * FX * FY;

      Result.Values[Y * Result.Width + X] :=
        Byte(Round(Min(255.0, Max(0.0, Value * 255))));
    end;
  end;
end;

function TSegmentDecoder.Decode(const Outputs: TTensorArray;
  const Context: TDecodeContext): TVisionResult;
var
  Tensor, ProtoTensor: TTensor;
  View: TPredictionView;
  Proto: TProtoPlanes;
  List: TList<TDetection>;
  Detection: TDetection;
  Detections: TDetections;
  Coefficients: TArray<Single>;
  Plane: TArray<Single>;
  A, I, C, ClassId, FirstClass, ClassCount, CoefficientFirstChannel: Integer;
  HasObjectness, EndToEnd: Boolean;
  Score, Objectness: Single;
begin
  Result := Default(TVisionResult);
  Result.Task := vtSegment;
  Result.ImageWidth := Context.Transform.SourceWidth;
  Result.ImageHeight := Context.Transform.SourceHeight;

  Tensor := FindPredictionTensor(Outputs);
  View := TPredictionView.FromTensor(Tensor);
  if not View.Valid then
    raise EDecodeError.CreateFmt(
      'Saida de segmentacao com formato inesperado: %s', [Tensor.ShapeText]);

  if not TryFindProtoTensor(Outputs, ProtoTensor) then
    raise EDecodeError.Create(
      'O modelo de segmentacao nao devolveu o tensor de prototipos (rank 4)');

  Proto := TProtoPlanes.FromTensor(ProtoTensor);
  if not Proto.IsValid then
    raise EDecodeError.CreateFmt(
      'Tensor de prototipos invalido: %s', [ProtoTensor.ShapeText]);

  EndToEnd := View.LooksDecoded(6 + Proto.Channels);

  if EndToEnd then
  begin
    FirstClass := 5;
    ClassCount := 1;
    HasObjectness := False;
    CoefficientFirstChannel := 6;
  end
  else
  begin
    ResolveRawClassLayout(View.Channels, Proto.Channels, Context.Spec.ClassCount,
      FirstClass, ClassCount, HasObjectness);
    CoefficientFirstChannel := FirstClass + ClassCount;
  end;

  SetLength(Coefficients, Proto.Channels);

  List := TList<TDetection>.Create;
  try
    for A := 0 to View.Anchors - 1 do
    begin
      Detection := Default(TDetection);

      if EndToEnd then
      begin
        Score := View.Value(A, 4);
        if Score < Context.ConfidenceThreshold then
          Break;
        ClassId := Round(View.Value(A, 5));
        Detection.Box := BoxFromCornerChannels(View, A, Context);
      end
      else
      begin
        if HasObjectness then
        begin
          Objectness := View.Value(A, 4);
          if Objectness < Context.ConfidenceThreshold then
            Continue;
        end
        else
          Objectness := 1;

        if not BestClass(View, A, FirstClass, ClassCount, ClassId, Score) then
          Continue;

        Score := Score * Objectness;
        if Score < Context.ConfidenceThreshold then
          Continue;

        Detection.Box := BoxFromCenterChannels(View, A, Context);
      end;

      Detection.ClassId := ClassId;
      Detection.Score := Score;
      Detection.Box := Detection.Box.ClampTo(
        Context.Transform.SourceWidth, Context.Transform.SourceHeight);

      if Detection.Box.IsEmpty then
        Continue;

      // A mascara so e montada depois do NMS, para nao gastar tempo com
      // caixas que serao descartadas. SourceIndex guarda a linha de origem.
      Detection.HasMask := False;
      Detection.SourceIndex := A;
      FillClassName(Detection, Context);
      List.Add(Detection);

      if EndToEnd and (Context.MaxDetections > 0) and
         (List.Count >= Context.MaxDetections) then
        Break;
    end;

    Detections := List.ToArray;
  finally
    List.Free;
  end;

  if not EndToEnd then
    Detections := FinalizeDetections(Detections, Context, False);

  // Segunda passada: monta a mascara apenas das deteccoes sobreviventes.
  for I := 0 to High(Detections) do
  begin
    A := Detections[I].SourceIndex;
    if (A < 0) or (A >= View.Anchors) then
      Continue;

    for C := 0 to Proto.Channels - 1 do
      Coefficients[C] := View.Value(A, CoefficientFirstChannel + C);

    Plane := BuildMaskPlane(Proto, Coefficients);
    Detections[I].Mask := CropMaskToBox(Plane, Proto, Detections[I].Box, Context);
    Detections[I].HasMask := Detections[I].Mask.IsValid;
  end;

  Result.Detections := Detections;
end;

initialization
  TDecoderRegistry.RegisterDecoder(vtSegment,
    function: IResultDecoder
    begin
      Result := TSegmentDecoder.Create;
    end);

end.
