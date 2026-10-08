unit Vision.Render.Ocr;

{
  Desenho do resultado de OCR, no formato da visualizacao do PaddleOCR:
  a imagem com as caixas a esquerda e, a direita, um painel branco com o
  texto lido escrito no lugar e no angulo de cada caixa. Comparar os dois
  lados mostra de uma vez o que foi detectado e o que foi lido.
}

interface

uses
  System.SysUtils,
  System.Types,
  System.Math,
  Vcl.Graphics,
  Vision.Types,
  Vision.Image,
  Vision.Render,
  Vision.Ocr;

type
  IOcrRenderer = interface
    ['{F1A6C83E-2D97-4B50-8E14-7C3B9A0D5E26}']
    function Render(const Source: IImage; const Value: TOcrResult): IImage;
  end;

  TOcrRenderer = class(TInterfacedObject, IOcrRenderer)
  private
    FLineWidth: Integer;
    procedure DrawPolygon(Canvas: TCanvas; const Box: TObbBox; OffsetX: Integer;
      Color: TColor);
    procedure DrawText(Canvas: TCanvas; const Line: TOcrLine; OffsetX: Integer);
  public
    constructor Create(ALineWidth: Integer = 2);
    function Render(const Source: IImage; const Value: TOcrResult): IImage;
  end;

implementation

constructor TOcrRenderer.Create(ALineWidth: Integer);
begin
  inherited Create;
  FLineWidth := Max(1, ALineWidth);
end;

procedure TOcrRenderer.DrawPolygon(Canvas: TCanvas; const Box: TObbBox;
  OffsetX: Integer; Color: TColor);
var
  Corners: TArray<TPointF>;
  Points: array[0..3] of TPoint;
  I: Integer;
begin
  Corners := Box.Corners;
  for I := 0 to 3 do
    Points[I] := Point(OffsetX + Round(Corners[I].X), Round(Corners[I].Y));
  Canvas.Brush.Style := bsClear;
  Canvas.Pen.Color := Color;
  Canvas.Pen.Width := FLineWidth;
  Canvas.Polygon(Points);
end;

procedure TOcrRenderer.DrawText(Canvas: TCanvas; const Line: TOcrLine;
  OffsetX: Integer);
var
  Box: TObbBox;
  Corners: TArray<TPointF>;
  Horizontal: Boolean;
  Along, Across, Size, TextWidth: Integer;
begin
  if Line.Text = '' then
    Exit;

  Box := Line.Detection.Obb;
  Corners := Box.Corners;

  { Recorte em pe foi lido deitado (girado 90 graus): o texto acompanha o
    lado maior da caixa. }
  Horizontal := Box.W >= Box.H;
  if Horizontal then
  begin
    Along := Round(Box.W);
    Across := Round(Box.H);
  end
  else
  begin
    Along := Round(Box.H);
    Across := Round(Box.W);
  end;

  // Altura da fonte pela caixa, reduzida ate o texto caber no comprimento.
  Canvas.Font.Name := 'Segoe UI';
  Canvas.Font.Style := [];
  Canvas.Font.Color := clBlack;
  Canvas.Font.Orientation := 0;
  Size := Max(6, Round(Across * 0.8));
  repeat
    Canvas.Font.Height := -Size;
    TextWidth := Canvas.TextWidth(Line.Text);
    if TextWidth <= Along then
      Break;
    Size := Max(6, Floor(Size * Along / TextWidth));
  until Size <= 6;

  Canvas.Brush.Style := bsClear;
  if Horizontal then
  begin
    // Orientation em decimos de grau, anti-horario; Angle e horario.
    Canvas.Font.Orientation := -Round(RadToDeg(Box.Angle) * 10);
    Canvas.TextOut(OffsetX + Round(Corners[0].X), Round(Corners[0].Y), Line.Text);
  end
  else
  begin
    { O recorte em pe gira 90 graus anti-horario antes da leitura, entao o
      texto foi lido de cima para baixo: desce a partir do canto superior
      direito, com o topo das letras para a direita. }
    Canvas.Font.Orientation := -900 - Round(RadToDeg(Box.Angle) * 10);
    Canvas.TextOut(OffsetX + Round(Corners[1].X), Round(Corners[1].Y), Line.Text);
  end;
  Canvas.Font.Orientation := 0;
end;

function TOcrRenderer.Render(const Source: IImage;
  const Value: TOcrResult): IImage;
var
  Bitmap, Left: TBitmap;
  I: Integer;
  Color: TColor;
begin
  if Source = nil then
    raise EImageError.Create('Imagem nula');

  Left := ImageToBitmap(Source);
  Bitmap := TBitmap.Create;
  try
    Bitmap.PixelFormat := pf24bit;
    Bitmap.SetSize(Source.Width * 2, Source.Height);
    Bitmap.Canvas.Brush.Style := bsSolid;
    Bitmap.Canvas.Brush.Color := clWhite;
    Bitmap.Canvas.FillRect(Rect(0, 0, Bitmap.Width, Bitmap.Height));
    Bitmap.Canvas.Draw(0, 0, Left);

    for I := 0 to High(Value.Lines) do
    begin
      Color := ClassColor(I);
      DrawPolygon(Bitmap.Canvas, Value.Lines[I].Detection.Obb, 0, Color);
      DrawPolygon(Bitmap.Canvas, Value.Lines[I].Detection.Obb, Source.Width, Color);
      DrawText(Bitmap.Canvas, Value.Lines[I], Source.Width);
    end;

    Result := BitmapToImage(Bitmap);
  finally
    Bitmap.Free;
    Left.Free;
  end;
end;

end.
