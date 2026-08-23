unit Vision.Decoder.Detect;

{
  Cabeca de deteccao.

  Tres formatos sao aceitos e escolhidos pelo shape real do tensor:

    A) YOLO26 end-to-end (NMS-free), [1, 300, 6]
       colunas: x1, y1, x2, y2, score, class_id   (pixels do letterbox)
       ja vem filtrado: nao passa por NMS.

    B) YOLOv8/v11/YOLO26 com end2end=False, [1, 4+nc, 8400]
       colunas: cx, cy, w, h, score_classe_0..n   (pixels do letterbox)

    C) YOLOv5, [1, 25200, 5+nc]
       colunas: cx, cy, w, h, objectness, score_classe_0..n

  B e C passam por NMS.
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
  Vision.Decoder;

type
  TDetectDecoder = class(TDetectionDecoderBase, IResultDecoder)
  private
    function DecodeEndToEnd(const View: TPredictionView;
      const Context: TDecodeContext): TDetections;
    function DecodeRaw(const View: TPredictionView;
      const Context: TDecodeContext): TDetections;
  public
    function Task: TVisionTask;
    function Describe: string;
    function Decode(const Outputs: TTensorArray;
      const Context: TDecodeContext): TVisionResult;
  end;

{ Resolve quantas classes o tensor cru carrega e se ha coluna de objectness.
  Compartilhado com os decoders de pose/segment/obb. }
procedure ResolveRawClassLayout(Channels, ExtraChannels, DeclaredClasses: Integer;
  out FirstClassChannel, ClassCount: Integer; out HasObjectness: Boolean);

implementation

procedure ResolveRawClassLayout(Channels, ExtraChannels, DeclaredClasses: Integer;
  out FirstClassChannel, ClassCount: Integer; out HasObjectness: Boolean);
var
  Available: Integer;
begin
  HasObjectness := False;
  FirstClassChannel := 4;

  Available := Channels - 4 - ExtraChannels;
  if Available <= 0 then
    raise EDecodeError.CreateFmt(
      'Saida com %d canais e pequena demais para 4 de caixa + %d extras',
      [Channels, ExtraChannels]);

  if (DeclaredClasses > 0) and (Available = DeclaredClasses + 1) then
  begin
    // Layout YOLOv5: uma coluna de objectness antes das classes.
    HasObjectness := True;
    FirstClassChannel := 5;
    ClassCount := DeclaredClasses;
    Exit;
  end;

  if (DeclaredClasses > 0) and (Available = DeclaredClasses) then
  begin
    ClassCount := DeclaredClasses;
    Exit;
  end;

  // Sem rotulos confiaveis: assume layout v8 e deduz o numero de classes.
  ClassCount := Available;
end;

{ TDetectDecoder }

function TDetectDecoder.Task: TVisionTask;
begin
  Result := vtDetect;
end;

function TDetectDecoder.Describe: string;
begin
  Result := 'deteccao (aceita [1,300,6] end-to-end, [1,4+nc,N] e [1,N,5+nc])';
end;

function TDetectDecoder.Decode(const Outputs: TTensorArray;
  const Context: TDecodeContext): TVisionResult;
var
  Tensor: TTensor;
  View: TPredictionView;
begin
  Result := Default(TVisionResult);
  Result.Task := vtDetect;
  Result.ImageWidth := Context.Transform.SourceWidth;
  Result.ImageHeight := Context.Transform.SourceHeight;

  Tensor := FindPredictionTensor(Outputs);
  View := TPredictionView.FromTensor(Tensor);

  if not View.Valid then
    raise EDecodeError.CreateFmt(
      'Saida de deteccao com formato inesperado: %s', [Tensor.ShapeText]);

  if View.LooksDecoded(6) then
    Result.Detections := DecodeEndToEnd(View, Context)
  else
    Result.Detections := FinalizeDetections(DecodeRaw(View, Context), Context, False);
end;

function TDetectDecoder.DecodeEndToEnd(const View: TPredictionView;
  const Context: TDecodeContext): TDetections;
var
  List: TList<TDetection>;
  A: Integer;
  Detection: TDetection;
  Score: Single;
begin
  List := TList<TDetection>.Create;
  try
    for A := 0 to View.Anchors - 1 do
    begin
      Score := View.Value(A, 4);
      // As linhas vem ordenadas por score decrescente; a primeira abaixo do
      // limiar encerra a varredura.
      if Score < Context.ConfidenceThreshold then
        Break;

      Detection := Default(TDetection);
      Detection.Score := Score;
      Detection.SourceIndex := A;
      Detection.ClassId := Round(View.Value(A, 5));
      Detection.Box := BoxFromCornerChannels(View, A, Context)
        .ClampTo(Context.Transform.SourceWidth, Context.Transform.SourceHeight);

      if Detection.Box.IsEmpty then
        Continue;

      FillClassName(Detection, Context);
      List.Add(Detection);

      if (Context.MaxDetections > 0) and (List.Count >= Context.MaxDetections) then
        Break;
    end;

    Result := List.ToArray;
  finally
    List.Free;
  end;
end;

function TDetectDecoder.DecodeRaw(const View: TPredictionView;
  const Context: TDecodeContext): TDetections;
var
  List: TList<TDetection>;
  A, FirstClass, ClassCount, ClassId: Integer;
  HasObjectness: Boolean;
  Score, Objectness: Single;
  Detection: TDetection;
begin
  ResolveRawClassLayout(View.Channels, 0, Context.Spec.ClassCount,
    FirstClass, ClassCount, HasObjectness);

  List := TList<TDetection>.Create;
  try
    for A := 0 to View.Anchors - 1 do
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

      Detection := Default(TDetection);
      Detection.ClassId := ClassId;
      Detection.Score := Score;
      Detection.SourceIndex := A;
      Detection.Box := BoxFromCenterChannels(View, A, Context)
        .ClampTo(Context.Transform.SourceWidth, Context.Transform.SourceHeight);

      if Detection.Box.IsEmpty then
        Continue;

      FillClassName(Detection, Context);
      List.Add(Detection);
    end;

    Result := List.ToArray;
  finally
    List.Free;
  end;
end;

initialization
  TDecoderRegistry.RegisterDecoder(vtDetect,
    function: IResultDecoder
    begin
      Result := TDetectDecoder.Create;
    end);

end.
