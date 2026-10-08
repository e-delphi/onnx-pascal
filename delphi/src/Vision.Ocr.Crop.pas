unit Vision.Ocr.Crop;

{
  Recorte de uma linha de texto detectada para o reconhecedor.

  Transcreve o CropByPolys (det_box_type "quad") do PaddleX:

    1. os cantos da caixa sao arredondados para pixel e limitados a imagem,
       como sai do detector Paddle;
    2. o retangulo de area minima desses quatro pontos e ordenado como no
       get_minarea_rect_crop (esq-sup, dir-sup, dir-inf, esq-inf);
    3. largura e altura do recorte = int() da maior das arestas opostas;
    4. o retangulo e "desentortado" para um recorte reto, com interpolacao
       bicubica (A = -0,75) e borda replicada - o warpPerspective com
       INTER_CUBIC e BORDER_REPLICATE do original. Entre dois retangulos o
       mapeamento e afim, entao nao ha divisao por W;
    5. recortes em pe (altura/largura >= 1,5) giram 90 graus no sentido
       anti-horario (np.rot90), para o texto vertical ficar deitado.

  A interpolacao e feita em ponto flutuante. Conferido contra o
  cv2.warpPerspective: diferenca maxima de 1 nivel de cinza por canal.
}

interface

uses
  System.SysUtils,
  System.Types,
  System.Math,
  System.Generics.Collections,
  System.Generics.Defaults,
  Vision.Types,
  Vision.Image;

type
  ITextLineCropper = interface
    ['{C2E7A1D4-3F58-4B96-A0C3-6D1E9B47F285}']
    { Nil quando a caixa degenera em menos de 1 pixel. }
    function Crop(const Source: IImage; const Box: TObbBox): IImage;
  end;

  TTextLineCropper = class(TInterfacedObject, ITextLineCropper)
  private
    function OrderedCorners(const Source: IImage; const Box: TObbBox): TArray<TPointF>;
    function Warp(const Source: IImage; const P0, P1, P3: TPointF;
      Width, Height: Integer): IImage;
    function Rotate90CounterClockwise(const Source: IImage): IImage;
  public
    function Crop(const Source: IImage; const Box: TObbBox): IImage;
  end;

implementation

uses
  Vision.Decoder.Text;

const
  { Recortes com altura/largura a partir disto sao texto vertical. }
  VERTICAL_RATIO = 1.5;
  CUBIC_A = -0.75;

{ Pesos da convolucao cubica de Keys para os 4 vizinhos de uma posicao com
  parte fracionaria T, na mesma formula do interpolateCubic do OpenCV. }
procedure CubicWeights(T: Double; out W0, W1, W2, W3: Double);
begin
  W0 := ((CUBIC_A * (T + 1) - 5 * CUBIC_A) * (T + 1) + 8 * CUBIC_A) * (T + 1) - 4 * CUBIC_A;
  W1 := ((CUBIC_A + 2) * T - (CUBIC_A + 3)) * T * T + 1;
  W2 := ((CUBIC_A + 2) * (1 - T) - (CUBIC_A + 3)) * (1 - T) * (1 - T) + 1;
  W3 := 1 - W0 - W1 - W2;
end;

function Distance(const A, B: TPointF): Double;
begin
  Result := Sqrt(Sqr(Double(A.X) - B.X) + Sqr(Double(A.Y) - B.Y));
end;

{ TTextLineCropper }

function TTextLineCropper.OrderedCorners(const Source: IImage;
  const Box: TObbBox): TArray<TPointF>;
var
  Corners, Sorted: TArray<TPointF>;
  I, A, B, C, D: Integer;
begin
  // 1. Cantos inteiros, limitados a imagem (saida do detector Paddle).
  Corners := Box.Corners;
  for I := 0 to 3 do
    Corners[I] := PointF(
      EnsureRange(Round(Corners[I].X), 0, Source.Width),
      EnsureRange(Round(Corners[I].Y), 0, Source.Height));

  // 2. Retangulo minimo desses pontos, ordenado como no PaddleX.
  Sorted := MinAreaRect(Corners).Corners;
  TArray.Sort<TPointF>(Sorted, TComparer<TPointF>.Construct(
    function(const L, R: TPointF): Integer
    begin
      Result := CompareValue(L.X, R.X);
    end));

  if Sorted[1].Y > Sorted[0].Y then
  begin
    A := 0; D := 1;
  end
  else
  begin
    A := 1; D := 0;
  end;
  if Sorted[3].Y > Sorted[2].Y then
  begin
    B := 2; C := 3;
  end
  else
  begin
    B := 3; C := 2;
  end;

  Result := TArray<TPointF>.Create(Sorted[A], Sorted[B], Sorted[C], Sorted[D]);
end;

function TTextLineCropper.Warp(const Source: IImage; const P0, P1, P3: TPointF;
  Width, Height: Integer): IImage;
var
  Target: TRGBImage;
  X, Y, I, J, Channel, IX, IY, SX, SY: Integer;
  EX, EY, FX, FY, SrcX, SrcY, Sum: Double;
  WX, WY: array[0..3] of Double;
  Rows: array[0..3] of PByte;
  Cols: array[0..3] of Integer;
  Dst: PByte;
begin
  Target := TRGBImage.Create(Width, Height);
  Result := Target;

  for Y := 0 to Height - 1 do
  begin
    Dst := Target.RowPtr(Y);
    for X := 0 to Width - 1 do
    begin
      { (0,0) -> P0, (Width,0) -> P1, (0,Height) -> P3. Pixel inteiro e o
        centro, como no warp do OpenCV (sem o deslocamento de meio pixel do
        resize). }
      EX := X / Width;
      EY := Y / Height;
      SrcX := P0.X + (Double(P1.X) - P0.X) * EX + (Double(P3.X) - P0.X) * EY;
      SrcY := P0.Y + (Double(P1.Y) - P0.Y) * EX + (Double(P3.Y) - P0.Y) * EY;

      IX := Floor(SrcX);
      IY := Floor(SrcY);
      FX := SrcX - IX;
      FY := SrcY - IY;
      CubicWeights(FX, WX[0], WX[1], WX[2], WX[3]);
      CubicWeights(FY, WY[0], WY[1], WY[2], WY[3]);

      // BORDER_REPLICATE: vizinhos fora da imagem repetem a borda.
      for I := 0 to 3 do
      begin
        SY := EnsureRange(IY - 1 + I, 0, Source.Height - 1);
        Rows[I] := Source.RowPtr(SY);
        SX := EnsureRange(IX - 1 + I, 0, Source.Width - 1);
        Cols[I] := SX * 3;
      end;

      for Channel := 0 to 2 do
      begin
        Sum := 0;
        for I := 0 to 3 do
          for J := 0 to 3 do
            Sum := Sum + WY[I] * WX[J] * PByte(Rows[I] + Cols[J] + Channel)^;
        Dst^ := Byte(EnsureRange(Round(Sum), 0, 255));
        Inc(Dst);
      end;
    end;
  end;
end;

function TTextLineCropper.Rotate90CounterClockwise(const Source: IImage): IImage;
var
  Target: TRGBImage;
  X, Y: Integer;
  Src, Dst: PByte;
begin
  // np.rot90: destino(y, x) = origem(x, largura - 1 - y).
  Target := TRGBImage.Create(Source.Height, Source.Width);
  Result := Target;
  for Y := 0 to Target.Height - 1 do
  begin
    Dst := Target.RowPtr(Y);
    for X := 0 to Target.Width - 1 do
    begin
      Src := Source.RowPtr(X) + (Source.Width - 1 - Y) * 3;
      Dst^ := Src^;                    Inc(Dst);
      Dst^ := PByte(Src + 1)^;         Inc(Dst);
      Dst^ := PByte(Src + 2)^;         Inc(Dst);
    end;
  end;
end;

function TTextLineCropper.Crop(const Source: IImage; const Box: TObbBox): IImage;
var
  P: TArray<TPointF>;
  Width, Height: Integer;
begin
  if Source = nil then
    raise EImageError.Create('Imagem nula');

  P := OrderedCorners(Source, Box);

  // 3. int() da maior das arestas opostas.
  Width := Trunc(Max(Distance(P[0], P[1]), Distance(P[2], P[3])));
  Height := Trunc(Max(Distance(P[0], P[3]), Distance(P[1], P[2])));
  if (Width < 1) or (Height < 1) then
    Exit(nil);

  // 4. Recorte reto.
  Result := Warp(Source, P[0], P[1], P[3], Width, Height);

  // 5. Texto vertical deita.
  if Height / Width >= VERTICAL_RATIO then
    Result := Rotate90CounterClockwise(Result);
end;

end.
