unit Vision.Decoder.Pose;

{
  Cabeca de pose: deteccao + keypoints.

  Formatos aceitos (escolhidos pelo shape real do tensor):

    A) end-to-end, [1, 300, 6 + K*D]
       x1, y1, x2, y2, score, class_id, depois K keypoints de D valores.

    B) cru, [1, 4 + nc + K*D, N]
       cx, cy, w, h, score(s) de classe, depois K keypoints de D valores.

  Com D = 3 os valores por keypoint sao x, y, visibilidade.
  K e D vem de kpt_shape nos metadados; sem metadados assume COCO (17x3).
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
  TPoseDecoder = class(TDetectionDecoderBase, IResultDecoder)
  private
    function ReadKeypoints(const View: TPredictionView; Anchor, FirstChannel,
      Count, Dims: Integer; const Context: TDecodeContext): TKeypoints;
    function ResolveKeypointGeometry(const Context: TDecodeContext;
      Channels, BoxAndClassChannels: Integer;
      out Count, Dims: Integer): Boolean;
  public
    function Task: TVisionTask;
    function Describe: string;
    function Decode(const Outputs: TTensorArray;
      const Context: TDecodeContext): TVisionResult;
  end;

implementation

function TPoseDecoder.Task: TVisionTask;
begin
  Result := vtPose;
end;

function TPoseDecoder.Describe: string;
begin
  Result := 'pose (caixa + keypoints; aceita end-to-end e saida crua)';
end;

function TPoseDecoder.ResolveKeypointGeometry(const Context: TDecodeContext;
  Channels, BoxAndClassChannels: Integer; out Count, Dims: Integer): Boolean;
var
  Remaining: Integer;
begin
  Dims := Context.Spec.KeypointDims;
  if Dims <= 0 then
    Dims := 3;

  Count := Context.Spec.KeypointCount;
  Remaining := Channels - BoxAndClassChannels;

  if Remaining <= 0 then
    Exit(False);

  if (Count > 0) and (Count * Dims = Remaining) then
    Exit(True);

  // Metadados ausentes ou inconsistentes: deduz do que sobrou.
  if Remaining mod 3 = 0 then
  begin
    Dims := 3;
    Count := Remaining div 3;
    Exit(True);
  end;

  if Remaining mod 2 = 0 then
  begin
    Dims := 2;
    Count := Remaining div 2;
    Exit(True);
  end;

  Result := False;
end;

function TPoseDecoder.ReadKeypoints(const View: TPredictionView;
  Anchor, FirstChannel, Count, Dims: Integer;
  const Context: TDecodeContext): TKeypoints;
var
  K, Base: Integer;
begin
  SetLength(Result, Count);
  for K := 0 to Count - 1 do
  begin
    Base := FirstChannel + K * Dims;
    Result[K].X := Context.Transform.NetToSourceX(View.Value(Anchor, Base));
    Result[K].Y := Context.Transform.NetToSourceY(View.Value(Anchor, Base + 1));
    if Dims >= 3 then
      Result[K].Score := View.Value(Anchor, Base + 2)
    else
      Result[K].Score := 1;
  end;
end;

function TPoseDecoder.Decode(const Outputs: TTensorArray;
  const Context: TDecodeContext): TVisionResult;
var
  Tensor: TTensor;
  View: TPredictionView;
  List: TList<TDetection>;
  Detection: TDetection;
  A, KeypointCount, KeypointDims: Integer;
  FirstClass, ClassCount, ClassId: Integer;
  HasObjectness, EndToEnd: Boolean;
  Score, Objectness: Single;
  KeypointFirstChannel: Integer;
begin
  Result := Default(TVisionResult);
  Result.Task := vtPose;
  Result.ImageWidth := Context.Transform.SourceWidth;
  Result.ImageHeight := Context.Transform.SourceHeight;

  Tensor := FindPredictionTensor(Outputs);
  View := TPredictionView.FromTensor(Tensor);
  if not View.Valid then
    raise EDecodeError.CreateFmt(
      'Saida de pose com formato inesperado: %s', [Tensor.ShapeText]);

  KeypointCount := Context.Spec.KeypointCount;
  KeypointDims := Context.Spec.KeypointDims;
  if KeypointDims <= 0 then
    KeypointDims := 3;

  EndToEnd := (KeypointCount > 0) and View.LooksDecoded(6 + KeypointCount * KeypointDims);

  if EndToEnd then
  begin
    KeypointFirstChannel := 6;
    FirstClass := 5;
    ClassCount := 1;
    HasObjectness := False;
  end
  else
  begin
    if not ResolveKeypointGeometry(Context, View.Channels,
         4 + Max(1, Context.Spec.ClassCount), KeypointCount, KeypointDims) then
      raise EDecodeError.CreateFmt(
        'Nao foi possivel deduzir o layout de keypoints em %s', [Tensor.ShapeText]);

    ResolveRawClassLayout(View.Channels, KeypointCount * KeypointDims,
      Context.Spec.ClassCount, FirstClass, ClassCount, HasObjectness);

    KeypointFirstChannel := FirstClass + ClassCount;
  end;

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
      Detection.SourceIndex := A;
      Detection.Box := Detection.Box.ClampTo(
        Context.Transform.SourceWidth, Context.Transform.SourceHeight);

      if Detection.Box.IsEmpty then
        Continue;

      Detection.HasKeypoints := KeypointCount > 0;
      if Detection.HasKeypoints then
        Detection.Keypoints := ReadKeypoints(View, A, KeypointFirstChannel,
          KeypointCount, KeypointDims, Context);

      FillClassName(Detection, Context);
      List.Add(Detection);

      if EndToEnd and (Context.MaxDetections > 0) and
         (List.Count >= Context.MaxDetections) then
        Break;
    end;

    if EndToEnd then
      Result.Detections := List.ToArray
    else
      Result.Detections := FinalizeDetections(List.ToArray, Context, False);
  finally
    List.Free;
  end;
end;

initialization
  TDecoderRegistry.RegisterDecoder(vtPose,
    function: IResultDecoder
    begin
      Result := TPoseDecoder.Create;
    end);

end.
