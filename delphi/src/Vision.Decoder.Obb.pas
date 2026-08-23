unit Vision.Decoder.Obb;

{
  Cabeca de caixas orientadas (oriented bounding boxes).

  Formatos aceitos:

    A) end-to-end, [1, 300, 7]
       cx, cy, w, h, score, class_id, angulo

    B) cru, [1, 4 + nc + 1, N]
       cx, cy, w, h, score_classe_0..n, angulo

  O angulo esta em radianos e nao e afetado pelo letterbox, que e isotropico.
  A supressao usa a area de intersecao real entre os quadrilateros.
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
  TObbDecoder = class(TDetectionDecoderBase, IResultDecoder)
  public
    function Task: TVisionTask;
    function Describe: string;
    function Decode(const Outputs: TTensorArray;
      const Context: TDecodeContext): TVisionResult;
  end;

implementation

function TObbDecoder.Task: TVisionTask;
begin
  Result := vtObb;
end;

function TObbDecoder.Describe: string;
begin
  Result := 'caixas orientadas (aceita [1,300,7] end-to-end e [1,5+nc,N] cru)';
end;

function TObbDecoder.Decode(const Outputs: TTensorArray;
  const Context: TDecodeContext): TVisionResult;
var
  Tensor: TTensor;
  View: TPredictionView;
  List: TList<TDetection>;
  Detection: TDetection;
  A, ClassId, FirstClass, ClassCount, AngleChannel: Integer;
  HasObjectness, EndToEnd: Boolean;
  Score, Objectness: Single;
begin
  Result := Default(TVisionResult);
  Result.Task := vtObb;
  Result.ImageWidth := Context.Transform.SourceWidth;
  Result.ImageHeight := Context.Transform.SourceHeight;

  Tensor := FindPredictionTensor(Outputs);
  View := TPredictionView.FromTensor(Tensor);
  if not View.Valid then
    raise EDecodeError.CreateFmt(
      'Saida OBB com formato inesperado: %s', [Tensor.ShapeText]);

  EndToEnd := View.LooksDecoded(7);

  if EndToEnd then
  begin
    FirstClass := 5;
    ClassCount := 1;
    HasObjectness := False;
    AngleChannel := 6;
  end
  else
  begin
    // O ultimo canal e o angulo; os demais seguem o layout de deteccao.
    ResolveRawClassLayout(View.Channels, 1, Context.Spec.ClassCount,
      FirstClass, ClassCount, HasObjectness);
    AngleChannel := View.Channels - 1;
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
      end;

      { Nos dois formatos as quatro primeiras colunas sao cx, cy, w, h em
        pixels do letterbox. Difere das cabecas detect/segment/pose, que
        emitem cantos x1,y1,x2,y2: para uma caixa rotacionada os cantos
        alinhados ao eixo nao descrevem a geometria. }
      Detection.Obb.CX := Context.Transform.NetToSourceX(View.Value(A, 0));
      Detection.Obb.CY := Context.Transform.NetToSourceY(View.Value(A, 1));
      Detection.Obb.W := Context.Transform.NetToSourceLength(View.Value(A, 2));
      Detection.Obb.H := Context.Transform.NetToSourceLength(View.Value(A, 3));

      if (Detection.Obb.W <= 0) or (Detection.Obb.H <= 0) then
        Continue;

      Detection.Obb.Angle := View.Value(A, AngleChannel);
      Detection.HasObb := True;
      Detection.ClassId := ClassId;
      Detection.Score := Score;
      Detection.SourceIndex := A;
      Detection.Box := Detection.Obb.AxisAlignedBounds.ClampTo(
        Context.Transform.SourceWidth, Context.Transform.SourceHeight);

      if Detection.Box.IsEmpty then
        Continue;

      FillClassName(Detection, Context);
      List.Add(Detection);

      if EndToEnd and (Context.MaxDetections > 0) and
         (List.Count >= Context.MaxDetections) then
        Break;
    end;

    if EndToEnd then
      Result.Detections := List.ToArray
    else
      Result.Detections := FinalizeDetections(List.ToArray, Context, True);
  finally
    List.Free;
  end;
end;

initialization
  TDecoderRegistry.RegisterDecoder(vtObb,
    function: IResultDecoder
    begin
      Result := TObbDecoder.Create;
    end);

end.
