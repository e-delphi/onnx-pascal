unit Vision.Render;

{
  Desenho do resultado sobre a imagem.

  Separado do predictor de proposito: quem infere nao precisa saber desenhar,
  e quem desenha nao precisa saber de ONNX. O renderizador recebe apenas
  IImage + TVisionResult.
}

interface

uses
  System.SysUtils,
  System.Types,
  System.Math,
  Vcl.Graphics,
  Vision.Types,
  Vision.Image;

type
  IResultRenderer = interface
    ['{3C7F8A16-5D42-4E90-B6C8-9A1D0E5B7F34}']
    function Render(const Source: IImage; const Value: TVisionResult): IImage;
  end;

  TResultRenderer = class(TInterfacedObject, IResultRenderer)
  private
    FKeypointThreshold: Single;
    FMaskThreshold: Byte;
    FMaskAlpha: Single;
    FLineWidth: Integer;
    procedure BlendMasks(const Target: IImage; const Value: TVisionResult);
    procedure DrawBox(Canvas: TCanvas; const Detection: TDetection; Color: TColor);
    procedure DrawObb(Canvas: TCanvas; const Detection: TDetection; Color: TColor);
    procedure DrawKeypoints(Canvas: TCanvas; const Detection: TDetection);
    procedure DrawLabel(Canvas: TCanvas; X, Y: Integer; const Text: string;
      Color: TColor);
  public
    constructor Create(AKeypointThreshold: Single = 0.5;
      AMaskThreshold: Byte = 128; AMaskAlpha: Single = 0.45;
      ALineWidth: Integer = 2);
    function Render(const Source: IImage; const Value: TVisionResult): IImage;
  end;

function ClassColor(ClassId: Integer): TColor;

implementation

const
  { Paleta em RGB; convertida para o formato BGR do TColor em ClassColor. }
  PALETTE: array[0..19] of Cardinal = (
    $FF3838, $FF9D97, $FF701F, $FFB21D, $CFD231,
    $48F90A, $92CC17, $3DDB86, $1A9334, $00D4BB,
    $2C99A8, $00C2FF, $344593, $6473FF, $0018EC,
    $8438FF, $520085, $CB38FF, $FF95C8, $FF37C7);

  { Esqueleto COCO de 17 keypoints, indices base zero. }
  SKELETON: array[0..18, 0..1] of Integer = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6));

function ClassColor(ClassId: Integer): TColor;
var
  Value: Cardinal;
  R, G, B: Byte;
begin
  if ClassId < 0 then
    ClassId := 0;
  Value := PALETTE[ClassId mod Length(PALETTE)];
  R := (Value shr 16) and $FF;
  G := (Value shr 8) and $FF;
  B := Value and $FF;
  Result := TColor(R or (Cardinal(G) shl 8) or (Cardinal(B) shl 16));
end;

{ TResultRenderer }

constructor TResultRenderer.Create(AKeypointThreshold: Single;
  AMaskThreshold: Byte; AMaskAlpha: Single; ALineWidth: Integer);
begin
  inherited Create;
  FKeypointThreshold := AKeypointThreshold;
  FMaskThreshold := AMaskThreshold;
  FMaskAlpha := Min(1.0, Max(0.0, AMaskAlpha));
  FLineWidth := Max(1, ALineWidth);
end;

procedure TResultRenderer.BlendMasks(const Target: IImage;
  const Value: TVisionResult);
var
  I, X, Y, PX, PY: Integer;
  Color: TColor;
  MR, MG, MB: Byte;
  Row: PByte;
  Detection: TDetection;
begin
  for I := 0 to High(Value.Detections) do
  begin
    Detection := Value.Detections[I];
    if (not Detection.HasMask) or (not Detection.Mask.IsValid) then
      Continue;

    Color := ClassColor(Detection.ClassId);
    MR := Byte(Color);
    MG := Byte(Color shr 8);
    MB := Byte(Color shr 16);

    for Y := 0 to Detection.Mask.Height - 1 do
    begin
      PY := Detection.Mask.OffsetY + Y;
      if (PY < 0) or (PY >= Target.Height) then
        Continue;

      Row := Target.RowPtr(PY);
      for X := 0 to Detection.Mask.Width - 1 do
      begin
        if Detection.Mask.ValueAt(X, Y) < FMaskThreshold then
          Continue;

        PX := Detection.Mask.OffsetX + X;
        if (PX < 0) or (PX >= Target.Width) then
          Continue;

        PByte(Row + PX * 3)^ :=
          Byte(Round(PByte(Row + PX * 3)^ * (1 - FMaskAlpha) + MR * FMaskAlpha));
        PByte(Row + PX * 3 + 1)^ :=
          Byte(Round(PByte(Row + PX * 3 + 1)^ * (1 - FMaskAlpha) + MG * FMaskAlpha));
        PByte(Row + PX * 3 + 2)^ :=
          Byte(Round(PByte(Row + PX * 3 + 2)^ * (1 - FMaskAlpha) + MB * FMaskAlpha));
      end;
    end;
  end;
end;

procedure TResultRenderer.DrawBox(Canvas: TCanvas; const Detection: TDetection;
  Color: TColor);
begin
  Canvas.Brush.Style := bsClear;
  Canvas.Pen.Color := Color;
  Canvas.Pen.Width := FLineWidth;
  Canvas.Rectangle(
    Round(Detection.Box.Left), Round(Detection.Box.Top),
    Round(Detection.Box.Right), Round(Detection.Box.Bottom));
end;

procedure TResultRenderer.DrawObb(Canvas: TCanvas; const Detection: TDetection;
  Color: TColor);
var
  Corners: TArray<TPointF>;
  Points: array[0..3] of TPoint;
  I: Integer;
begin
  Corners := Detection.Obb.Corners;
  if Length(Corners) < 4 then
    Exit;

  for I := 0 to 3 do
    Points[I] := Point(Round(Corners[I].X), Round(Corners[I].Y));

  Canvas.Brush.Style := bsClear;
  Canvas.Pen.Color := Color;
  Canvas.Pen.Width := FLineWidth;
  Canvas.Polygon(Points);
end;

procedure TResultRenderer.DrawKeypoints(Canvas: TCanvas;
  const Detection: TDetection);
var
  I, A, B, Radius: Integer;
  Color: TColor;
begin
  if Length(Detection.Keypoints) = 0 then
    Exit;

  Radius := Max(2, FLineWidth + 1);

  // Ligacoes do esqueleto (somente para o layout COCO de 17 pontos).
  if Length(Detection.Keypoints) >= 17 then
  begin
    Canvas.Pen.Width := FLineWidth;
    for I := 0 to High(SKELETON) do
    begin
      A := SKELETON[I, 0];
      B := SKELETON[I, 1];
      if (Detection.Keypoints[A].Score < FKeypointThreshold) or
         (Detection.Keypoints[B].Score < FKeypointThreshold) then
        Continue;

      Canvas.Pen.Color := ClassColor(I);
      Canvas.MoveTo(Round(Detection.Keypoints[A].X), Round(Detection.Keypoints[A].Y));
      Canvas.LineTo(Round(Detection.Keypoints[B].X), Round(Detection.Keypoints[B].Y));
    end;
  end;

  for I := 0 to High(Detection.Keypoints) do
  begin
    if Detection.Keypoints[I].Score < FKeypointThreshold then
      Continue;

    Color := ClassColor(I + 3);
    Canvas.Brush.Style := bsSolid;
    Canvas.Brush.Color := Color;
    Canvas.Pen.Color := Color;
    Canvas.Pen.Width := 1;
    Canvas.Ellipse(
      Round(Detection.Keypoints[I].X) - Radius,
      Round(Detection.Keypoints[I].Y) - Radius,
      Round(Detection.Keypoints[I].X) + Radius,
      Round(Detection.Keypoints[I].Y) + Radius);
  end;

  Canvas.Brush.Style := bsClear;
end;

procedure TResultRenderer.DrawLabel(Canvas: TCanvas; X, Y: Integer;
  const Text: string; Color: TColor);
var
  TextWidth, TextHeight, BoxTop: Integer;
begin
  Canvas.Font.Name := 'Segoe UI';
  Canvas.Font.Size := 9;
  Canvas.Font.Style := [fsBold];

  TextWidth := Canvas.TextWidth(Text) + 6;
  TextHeight := Canvas.TextHeight(Text) + 2;

  BoxTop := Y - TextHeight;
  if BoxTop < 0 then
    BoxTop := Y;

  Canvas.Brush.Style := bsSolid;
  Canvas.Brush.Color := Color;
  Canvas.Pen.Color := Color;
  Canvas.Rectangle(X, BoxTop, X + TextWidth, BoxTop + TextHeight);

  Canvas.Font.Color := clWhite;
  Canvas.Brush.Style := bsClear;
  Canvas.TextOut(X + 3, BoxTop + 1, Text);
end;

function TResultRenderer.Render(const Source: IImage;
  const Value: TVisionResult): IImage;
var
  Working: IImage;
  Bitmap: TBitmap;
  I: Integer;
  Color: TColor;
  Text: string;
  Detection: TDetection;
begin
  if Source = nil then
    raise EImageError.Create('Imagem nula');

  // Copia: nunca desenha sobre a imagem que o chamador passou.
  Working := ResampleBilinear(Source, Source.Width, Source.Height);

  BlendMasks(Working, Value);

  Bitmap := ImageToBitmap(Working);
  try
    for I := 0 to High(Value.Detections) do
    begin
      Detection := Value.Detections[I];
      Color := ClassColor(Detection.ClassId);

      if Detection.HasObb then
        DrawObb(Bitmap.Canvas, Detection, Color)
      else
        DrawBox(Bitmap.Canvas, Detection, Color);

      if Detection.HasKeypoints then
        DrawKeypoints(Bitmap.Canvas, Detection);

      // Uma pagina tem centenas de linhas: rotulo em cada uma cobriria o
      // proprio texto detectado. O relatorio de console traz os scores.
      if Value.Task = vtText then
        Continue;

      Text := Format('%s %.0f%%', [Detection.DisplayName, Detection.Score * 100]);
      DrawLabel(Bitmap.Canvas, Round(Detection.Box.Left),
        Round(Detection.Box.Top), Text, Color);
    end;

    Result := BitmapToImage(Bitmap);
  finally
    Bitmap.Free;
  end;
end;

end.
