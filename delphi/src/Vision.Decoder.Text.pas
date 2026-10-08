unit Vision.Decoder.Text;

{
  Deteccao de texto pelo DBNet (Differentiable Binarization), a cabeca dos
  detectores PaddleOCR - PP-OCRv6_medium_det, small, tiny e os v5.

  A rede emite um unico mapa [1, 1, H, W] com a probabilidade de cada pixel
  pertencer a uma linha de texto ENCOLHIDA: o treino reduz cada poligono
  para que linhas vizinhas nao se toquem. O pos-processamento desfaz isso:

    1. binariza o mapa (BinaryThreshold, 0.2 no PP-OCRv6);
    2. separa os componentes conexos (8-vizinhanca);
    3. ajusta a cada um o retangulo de area minima (rotacionado);
    4. pontua pela media do mapa dentro do retangulo e descarta abaixo de
       ConfidenceThreshold (o box_thresh do Paddle, 0.45);
    5. expande o retangulo por d = area * UnclipRatio / perimetro, o
       inverso do encolhimento do treino;
    6. leva os cantos para a imagem original, um fator por eixo.

  E a transcricao do DBPostProcess do PaddleX (box_type "quad", score_mode
  "fast") sem OpenCV nem pyclipper. A mascara do score reproduz a
  rasterizacao do cv2.fillPoly pixel a pixel, porque e ela que define o
  numero publicado. Duas diferencas deliberadas:

    - contornos de buracos internos (RETR_LIST do cv2.findContours) sao
      ignorados; so a borda externa de cada componente vira caixa;
    - o unclip de um retangulo por offset com juntas redondas tem como
      retangulo minimo exatamente o original crescido d de cada lado, entao
      a expansao e feita em forma fechada.

  O resultado vai em TDetection com HasObb: o retangulo final e
  rotacionado, e o renderizador e o relatorio ja sabem lidar com isso.
}

interface

uses
  System.SysUtils,
  System.Types,
  System.Math,
  System.Generics.Collections,
  System.Generics.Defaults,
  ONNX.Types,
  Vision.Types,
  Vision.Model,
  Vision.Preprocess,
  Vision.Decoder;

type
  { Retangulo rotacionado no sistema do mapa: U e o eixo da largura, V o
    da altura, ambos unitarios e perpendiculares. }
  TRotatedRect = record
    Center: TPointF;
    U, V: TPointF;
    Width, Height: Single;
    function ShortSide: Single;
    function Corners: TArray<TPointF>;
    procedure AlignToReadingDirection;
  end;

  TTextDecoder = class(TInterfacedObject, IResultDecoder)
  private
    function FindProbabilityMap(const Outputs: TTensorArray): TTensor;
    function BoxScore(const Map: TArray<Single>; MapWidth, MapHeight: Integer;
      const Corners: TArray<TPointF>): Single;
    function ToSourceObb(const Rect: TRotatedRect; ScaleX, ScaleY: Single): TObbBox;
    procedure SortReadingOrder(var Detections: TDetections);
  public
    function Task: TVisionTask;
    function Describe: string;
    function Decode(const Outputs: TTensorArray;
      const Context: TDecodeContext): TVisionResult;
  end;

{ Geometria exposta para teste. }
function ConvexHull(const Points: TArray<TPointF>): TArray<TPointF>;
function MinAreaRect(const Points: TArray<TPointF>): TRotatedRect;

implementation

const
  { min_size do DBPostProcess: lado menor abaixo disso e ruido. Depois do
    unclip o corte sobe para MIN_SIDE + 2. }
  MIN_SIDE = 3;
  { Tolerancia vertical, em pixels da imagem, para considerar duas caixas
    na mesma linha ao ordenar (a mesma do sorted_boxes do PaddleOCR). }
  SAME_LINE_TOLERANCE = 10;

{ TRotatedRect }

function TRotatedRect.ShortSide: Single;
begin
  Result := Min(Width, Height);
end;

function TRotatedRect.Corners: TArray<TPointF>;
var
  HU, HV: TPointF;
begin
  HU := PointF(U.X * Width * 0.5, U.Y * Width * 0.5);
  HV := PointF(V.X * Height * 0.5, V.Y * Height * 0.5);
  SetLength(Result, 4);
  Result[0] := PointF(Center.X - HU.X - HV.X, Center.Y - HU.Y - HV.Y);
  Result[1] := PointF(Center.X + HU.X - HV.X, Center.Y + HU.Y - HV.Y);
  Result[2] := PointF(Center.X + HU.X + HV.X, Center.Y + HU.Y + HV.Y);
  Result[3] := PointF(Center.X - HU.X + HV.X, Center.Y - HU.Y + HV.Y);
end;

procedure TRotatedRect.AlignToReadingDirection;
var
  Angle, Swap: Single;
  Old: TPointF;
begin
  { O mesmo retangulo admite quatro descricoes (U girado de 90 em 90
    graus). Escolhe a de U mais proximo do eixo X, com V para baixo: assim
    o canto 0 e o superior esquerdo e Width e o sentido de leitura de uma
    linha horizontal. }
  Angle := ArcTan2(U.Y, U.X);

  if (Angle > 3 * Pi / 4) or (Angle <= -3 * Pi / 4) then
  begin
    U := PointF(-U.X, -U.Y);
    V := PointF(-V.X, -V.Y);
  end
  else if Angle > Pi / 4 then
  begin
    // U aponta para baixo: o novo U e o antigo girado -90 graus.
    Old := U;
    U := PointF(-V.X, -V.Y);
    V := Old;
    Swap := Width; Width := Height; Height := Swap;
  end
  else if Angle <= -Pi / 4 then
  begin
    Old := U;
    U := V;
    V := PointF(-Old.X, -Old.Y);
    Swap := Width; Width := Height; Height := Swap;
  end;
end;

{ Geometria }

function Cross(const O, A, B: TPointF): Double;
begin
  Result := (Double(A.X) - O.X) * (Double(B.Y) - O.Y) -
            (Double(A.Y) - O.Y) * (Double(B.X) - O.X);
end;

function ConvexHull(const Points: TArray<TPointF>): TArray<TPointF>;
var
  Sorted: TArray<TPointF>;
  Hull: TArray<TPointF>;
  I, K, LowerSize: Integer;
begin
  // Cadeia monotona de Andrew: O(n log n), casco em sentido anti-horario.
  if Length(Points) < 3 then
    Exit(Copy(Points));

  Sorted := Copy(Points);
  TArray.Sort<TPointF>(Sorted, TComparer<TPointF>.Construct(
    function(const L, R: TPointF): Integer
    begin
      if L.X < R.X then Result := -1
      else if L.X > R.X then Result := 1
      else if L.Y < R.Y then Result := -1
      else if L.Y > R.Y then Result := 1
      else Result := 0;
    end));

  SetLength(Hull, 2 * Length(Sorted));
  K := 0;

  for I := 0 to High(Sorted) do
  begin
    while (K >= 2) and (Cross(Hull[K - 2], Hull[K - 1], Sorted[I]) <= 0) do
      Dec(K);
    Hull[K] := Sorted[I];
    Inc(K);
  end;

  LowerSize := K + 1;
  for I := High(Sorted) - 1 downto 0 do
  begin
    while (K >= LowerSize) and (Cross(Hull[K - 2], Hull[K - 1], Sorted[I]) <= 0) do
      Dec(K);
    Hull[K] := Sorted[I];
    Inc(K);
  end;

  // O ultimo ponto repete o primeiro.
  SetLength(Hull, Max(1, K - 1));
  Result := Hull;
end;

function MinAreaRect(const Points: TArray<TPointF>): TRotatedRect;
var
  Hull: TArray<TPointF>;
  I, J, N: Integer;
  DX, DY, Len, ProjU, ProjV: Double;
  MinU, MaxU, MinV, MaxV, Area, BestArea: Double;
  UX, UY, VX, VY: Double;
begin
  Result := Default(TRotatedRect);
  Result.U := PointF(1, 0);
  Result.V := PointF(0, 1);

  Hull := ConvexHull(Points);
  N := Length(Hull);
  if N = 0 then
    Exit;

  Result.Center := Hull[0];
  if N = 1 then
    Exit;

  { O retangulo minimo tem um lado colinear com alguma aresta do casco
    (Freeman & Shapira). Testa a orientacao de cada aresta: O(n^2), com n
    sendo o numero de vertices do casco - poucas dezenas por linha. }
  BestArea := MaxDouble;
  for I := 0 to N - 1 do
  begin
    J := (I + 1) mod N;
    DX := Double(Hull[J].X) - Hull[I].X;
    DY := Double(Hull[J].Y) - Hull[I].Y;
    Len := Sqrt(DX * DX + DY * DY);
    if Len < 1e-9 then
      Continue;

    UX := DX / Len;  UY := DY / Len;
    VX := -UY;       VY := UX;

    MinU := MaxDouble;  MaxU := -MaxDouble;
    MinV := MaxDouble;  MaxV := -MaxDouble;
    for J := 0 to N - 1 do
    begin
      ProjU := Hull[J].X * UX + Hull[J].Y * UY;
      ProjV := Hull[J].X * VX + Hull[J].Y * VY;
      MinU := Min(MinU, ProjU);  MaxU := Max(MaxU, ProjU);
      MinV := Min(MinV, ProjV);  MaxV := Max(MaxV, ProjV);
    end;

    Area := (MaxU - MinU) * (MaxV - MinV);
    if Area < BestArea then
    begin
      BestArea := Area;
      Result.U := PointF(UX, UY);
      Result.V := PointF(VX, VY);
      Result.Width := MaxU - MinU;
      Result.Height := MaxV - MinV;
      Result.Center := PointF(
        UX * (MinU + MaxU) * 0.5 + VX * (MinV + MaxV) * 0.5,
        UY * (MinU + MaxU) * 0.5 + VY * (MinV + MaxV) * 0.5);
    end;
  end;
end;

{ Rasterizacao }

const
  XY_SHIFT = 16;
  XY_ONE = Int64(1) shl XY_SHIFT;

{ Deslocamento aritmetico: o shr do Delphi e logico e quebraria negativos. }
function FixedToInt(Value: Int64): Int64; inline;
begin
  if Value >= 0 then
    Result := Value shr XY_SHIFT
  else
    Result := -((-Value + XY_ONE - 1) shr XY_SHIFT);
end;

{ cv::LineIterator de 8 vizinhos, da esquerda para a direita. }
procedure DrawLine8(var Mask: TArray<Byte>; Width, Height: Integer;
  P0, P1: TPoint);
var
  DX, DY, StepY, Err, X, Y, I, Swap: Integer;
  Steep, Minor: Boolean;
begin
  if P1.X < P0.X then
  begin
    Swap := P0.X; P0.X := P1.X; P1.X := Swap;
    Swap := P0.Y; P0.Y := P1.Y; P1.Y := Swap;
  end;

  DX := P1.X - P0.X;
  DY := P1.Y - P0.Y;
  StepY := 1;
  if DY < 0 then
  begin
    DY := -DY;
    StepY := -1;
  end;

  Steep := DY > DX;
  if Steep then
  begin
    Swap := DX; DX := DY; DY := Swap;
  end;

  Err := DX - 2 * DY;
  X := P0.X;
  Y := P0.Y;
  for I := 0 to DX do
  begin
    if (X >= 0) and (X < Width) and (Y >= 0) and (Y < Height) then
      Mask[Y * Width + X] := 1;

    Minor := Err < 0;
    Err := Err - 2 * DY;
    if Minor then
      Err := Err + 2 * DX;

    if Steep then
    begin
      Inc(Y, StepY);
      if Minor then Inc(X);
    end
    else
    begin
      Inc(X);
      if Minor then Inc(Y, StepY);
    end;
  end;
end;

{ Reproduz cv2.fillPoly(lineType=8, shift=0) para um poligono convexo:
  contorno tracado pelo LineIterator mais preenchimento por linhas em ponto
  fixo 16.16, onde a linha inferior de cada aresta fica de fora. O score do
  DB e uma media sobre essa mascara, e em linhas de texto finas os pixels de
  borda pesam: um preenchimento "geometricamente correto" desvia o score em
  ate 10 pontos percentuais. Conferido pixel a pixel contra o OpenCV. }
procedure FillPolyMask(var Mask: TArray<Byte>; Width, Height: Integer;
  const Poly: array of TPoint);
type
  TEdge = record
    Y0, Y1: Integer;
    X, DX: Int64;
  end;
var
  Edges: array of TEdge;
  Hits: array of Int64;
  A, B: TPoint;
  I, J, N, Count, Y, YFirst, YLast, X1, X2, X: Integer;
  Temp: Int64;
begin
  N := Length(Poly);
  SetLength(Edges, N);
  Count := 0;
  YFirst := MaxInt;
  YLast := -MaxInt;

  for I := 0 to N - 1 do
  begin
    A := Poly[(I + N - 1) mod N];
    B := Poly[I];
    DrawLine8(Mask, Width, Height, A, B);

    if A.Y = B.Y then
      Continue;

    // div do Delphi trunca para zero, como a divisao inteira do C.
    Edges[Count].DX := ((Int64(B.X) - A.X) shl XY_SHIFT) div (B.Y - A.Y);
    if A.Y < B.Y then
    begin
      Edges[Count].Y0 := A.Y;
      Edges[Count].Y1 := B.Y;
      Edges[Count].X := Int64(A.X) shl XY_SHIFT;
    end
    else
    begin
      Edges[Count].Y0 := B.Y;
      Edges[Count].Y1 := A.Y;
      Edges[Count].X := Int64(B.X) shl XY_SHIFT;
    end;
    YFirst := Min(YFirst, Edges[Count].Y0);
    YLast := Max(YLast, Edges[Count].Y1);
    Inc(Count);
  end;

  SetLength(Hits, Count);
  for Y := Max(YFirst, 0) to Min(YLast, Height) - 1 do
  begin
    // Interseccoes ativas nesta linha, ordenadas (no maximo 4).
    N := 0;
    for I := 0 to Count - 1 do
      if (Edges[I].Y0 <= Y) and (Y < Edges[I].Y1) then
      begin
        Hits[N] := Edges[I].X + Int64(Y - Edges[I].Y0) * Edges[I].DX;
        J := N;
        while (J > 0) and (Hits[J - 1] > Hits[J]) do
        begin
          Temp := Hits[J]; Hits[J] := Hits[J - 1]; Hits[J - 1] := Temp;
          Dec(J);
        end;
        Inc(N);
      end;

    I := 0;
    while I + 1 < N do
    begin
      X1 := Max(0, Integer(FixedToInt(Hits[I] + XY_ONE - 1)));
      X2 := Min(Width - 1, Integer(FixedToInt(Hits[I + 1])));
      for X := X1 to X2 do
        Mask[Y * Width + X] := 1;
      Inc(I, 2);
    end;
  end;
end;

{ Componentes conexos }

type
  { Extremos horizontais de cada linha de um componente: o casco convexo
    dos pixels e o casco desses pontos, entao nao e preciso guardar o
    componente inteiro. }
  TComponentScanner = class
  private
    FMask: TArray<Boolean>;
    FWidth, FHeight: Integer;
    FVisited: TArray<Boolean>;
    FStack: TArray<Integer>;
    FRowMin, FRowMax: TArray<Integer>;
  public
    constructor Create(const Map: TArray<Single>; AWidth, AHeight: Integer;
      Threshold: Single);
    { Percorre a partir de Start e devolve os pontos extremos por linha.
      False se Start nao e pixel de texto ou ja foi visitado. }
    function Extract(Start: Integer; out Extremes: TArray<TPointF>): Boolean;
  end;

constructor TComponentScanner.Create(const Map: TArray<Single>;
  AWidth, AHeight: Integer; Threshold: Single);
var
  I: Integer;
begin
  inherited Create;
  FWidth := AWidth;
  FHeight := AHeight;
  SetLength(FMask, AWidth * AHeight);
  for I := 0 to High(FMask) do
    FMask[I] := Map[I] > Threshold;
  SetLength(FVisited, AWidth * AHeight);
  SetLength(FRowMin, AHeight);
  SetLength(FRowMax, AHeight);
  for I := 0 to AHeight - 1 do
  begin
    FRowMin[I] := MaxInt;
    FRowMax[I] := -1;
  end;
end;

function TComponentScanner.Extract(Start: Integer;
  out Extremes: TArray<TPointF>): Boolean;
var
  Top, Index, X, Y, NX, NY, DX, DY, MinY, MaxY, Count: Integer;
begin
  Extremes := nil;
  if (not FMask[Start]) or FVisited[Start] then
    Exit(False);

  // Flood fill iterativo: componentes de texto passam de 10^5 pixels.
  if Length(FStack) < 64 then
    SetLength(FStack, 1024);
  FVisited[Start] := True;
  FStack[0] := Start;
  Top := 1;
  MinY := Start div FWidth;
  MaxY := MinY;

  while Top > 0 do
  begin
    Dec(Top);
    Index := FStack[Top];
    Y := Index div FWidth;
    X := Index mod FWidth;

    if X < FRowMin[Y] then FRowMin[Y] := X;
    if X > FRowMax[Y] then FRowMax[Y] := X;
    if Y < MinY then MinY := Y;
    if Y > MaxY then MaxY := Y;

    for DY := -1 to 1 do
    begin
      NY := Y + DY;
      if (NY < 0) or (NY >= FHeight) then
        Continue;
      for DX := -1 to 1 do
      begin
        NX := X + DX;
        if (NX < 0) or (NX >= FWidth) then
          Continue;
        Index := NY * FWidth + NX;
        if FMask[Index] and (not FVisited[Index]) then
        begin
          FVisited[Index] := True;
          if Top >= Length(FStack) then
            SetLength(FStack, Length(FStack) * 2);
          FStack[Top] := Index;
          Inc(Top);
        end;
      end;
    end;
  end;

  SetLength(Extremes, 2 * (MaxY - MinY + 1));
  Count := 0;
  for Y := MinY to MaxY do
  begin
    if FRowMax[Y] >= 0 then
    begin
      Extremes[Count] := PointF(FRowMin[Y], Y);
      Inc(Count);
      if FRowMax[Y] <> FRowMin[Y] then
      begin
        Extremes[Count] := PointF(FRowMax[Y], Y);
        Inc(Count);
      end;
    end;
    // Prepara as linhas para o proximo componente.
    FRowMin[Y] := MaxInt;
    FRowMax[Y] := -1;
  end;
  SetLength(Extremes, Count);
  Result := True;
end;

{ TTextDecoder }

function TTextDecoder.Task: TVisionTask;
begin
  Result := vtText;
end;

function TTextDecoder.Describe: string;
begin
  Result := 'DBNet: mapa [1,1,H,W] -> retangulos rotacionados (PaddleOCR)';
end;

function TTextDecoder.FindProbabilityMap(const Outputs: TTensorArray): TTensor;
var
  I: Integer;
begin
  for I := 0 to High(Outputs) do
    if (Outputs[I].Rank = 4) and (Outputs[I].Dim(1) = 1) then
    begin
      if Outputs[I].Dim(0) <> 1 then
        raise EDecodeError.CreateFmt(
          'Somente batch 1 e suportado; recebido shape %s',
          [Outputs[I].ShapeText]);
      Exit(Outputs[I]);
    end;

  raise EDecodeError.Create(
    'Nenhuma saida com formato de mapa de texto [1, 1, H, W] foi encontrada');
end;

function TTextDecoder.BoxScore(const Map: TArray<Single>;
  MapWidth, MapHeight: Integer; const Corners: TArray<TPointF>): Single;
var
  Poly: array[0..3] of TPoint;
  Mask: TArray<Byte>;
  XMin, XMax, YMin, YMax, MaskWidth, MaskHeight, X, Y, I, Row, Count: Integer;
  MinX, MaxX, MinY, MaxY: Single;
  Sum: Double;
begin
  { box_score_fast: media do mapa dentro do quadrilatero, numa janela do
    tamanho do seu envelope. Os vertices viram inteiros por truncamento,
    relativos ao canto da janela, como o astype(np.int32) do original. }
  MinX := Corners[0].X;  MaxX := MinX;
  MinY := Corners[0].Y;  MaxY := MinY;
  for I := 1 to 3 do
  begin
    MinX := Min(MinX, Corners[I].X);  MaxX := Max(MaxX, Corners[I].X);
    MinY := Min(MinY, Corners[I].Y);  MaxY := Max(MaxY, Corners[I].Y);
  end;

  XMin := EnsureRange(Floor(MinX), 0, MapWidth - 1);
  XMax := EnsureRange(Ceil(MaxX), 0, MapWidth - 1);
  YMin := EnsureRange(Floor(MinY), 0, MapHeight - 1);
  YMax := EnsureRange(Ceil(MaxY), 0, MapHeight - 1);

  for I := 0 to 3 do
    Poly[I] := Point(Trunc(Corners[I].X - XMin), Trunc(Corners[I].Y - YMin));

  MaskWidth := XMax - XMin + 1;
  MaskHeight := YMax - YMin + 1;
  SetLength(Mask, MaskWidth * MaskHeight);
  FillPolyMask(Mask, MaskWidth, MaskHeight, Poly);

  Sum := 0;
  Count := 0;
  for Y := 0 to MaskHeight - 1 do
  begin
    Row := (YMin + Y) * MapWidth + XMin;
    for X := 0 to MaskWidth - 1 do
      if Mask[Y * MaskWidth + X] <> 0 then
      begin
        Sum := Sum + Map[Row + X];
        Inc(Count);
      end;
  end;

  if Count = 0 then
    Result := 0
  else
    Result := Sum / Count;
end;

function TTextDecoder.ToSourceObb(const Rect: TRotatedRect;
  ScaleX, ScaleY: Single): TObbBox;
var
  P: TArray<TPointF>;
  I: Integer;
  TopX, TopY, SideX, SideY: Double;
begin
  { A escala pode diferir entre os eixos (cada lado arredonda para
    multiplo de 32 por conta propria). Leva os cantos e remonta o
    retangulo: o desvio de perpendicularidade fica abaixo de 1%. }
  P := Rect.Corners;
  for I := 0 to 3 do
    P[I] := PointF(P[I].X * ScaleX, P[I].Y * ScaleY);

  Result.CX := (P[0].X + P[1].X + P[2].X + P[3].X) * 0.25;
  Result.CY := (P[0].Y + P[1].Y + P[2].Y + P[3].Y) * 0.25;

  // Media das arestas opostas, para nao favorecer nenhum lado.
  TopX := ((P[1].X - P[0].X) + (P[2].X - P[3].X)) * 0.5;
  TopY := ((P[1].Y - P[0].Y) + (P[2].Y - P[3].Y)) * 0.5;
  SideX := ((P[3].X - P[0].X) + (P[2].X - P[1].X)) * 0.5;
  SideY := ((P[3].Y - P[0].Y) + (P[2].Y - P[1].Y)) * 0.5;

  Result.W := Sqrt(TopX * TopX + TopY * TopY);
  Result.H := Sqrt(SideX * SideX + SideY * SideY);
  Result.Angle := ArcTan2(TopY, TopX);
end;

procedure TTextDecoder.SortReadingOrder(var Detections: TDetections);
var
  I, J: Integer;
  Temp: TDetection;
begin
  // Mesma regra do sorted_boxes do PaddleOCR: ordena pelo canto superior
  // e depois corrige a ordem dentro de cada linha visual.
  TArray.Sort<TDetection>(Detections, TComparer<TDetection>.Construct(
    function(const L, R: TDetection): Integer
    begin
      Result := CompareValue(L.Box.Top, R.Box.Top);
      if Result = 0 then
        Result := CompareValue(L.Box.Left, R.Box.Left);
    end));

  for I := 0 to High(Detections) - 1 do
    for J := I downto 0 do
    begin
      if (Abs(Detections[J + 1].Box.Top - Detections[J].Box.Top) < SAME_LINE_TOLERANCE) and
         (Detections[J + 1].Box.Left < Detections[J].Box.Left) then
      begin
        Temp := Detections[J];
        Detections[J] := Detections[J + 1];
        Detections[J + 1] := Temp;
      end
      else
        Break;
    end;
end;

function TTextDecoder.Decode(const Outputs: TTensorArray;
  const Context: TDecodeContext): TVisionResult;
var
  Tensor: TTensor;
  Map: TArray<Single>;
  MapWidth, MapHeight, Index, Candidates: Integer;
  Scanner: TComponentScanner;
  Extremes: TArray<TPointF>;
  Rect: TRotatedRect;
  Score, Distance, ScaleX, ScaleY: Single;
  List: TList<TDetection>;
  Detection: TDetection;
begin
  Result := Default(TVisionResult);
  Result.Task := vtText;
  Result.ImageWidth := Context.Transform.SourceWidth;
  Result.ImageHeight := Context.Transform.SourceHeight;

  Tensor := FindProbabilityMap(Outputs);
  MapHeight := Tensor.DimAsInt(2);
  MapWidth := Tensor.DimAsInt(3);
  Map := Tensor.Data;
  if (MapWidth <= 0) or (MapHeight <= 0) or
     (Length(Map) < MapWidth * MapHeight) then
    raise EDecodeError.CreateFmt('Mapa de texto invalido: %s', [Tensor.ShapeText]);

  // O mapa tem o tamanho da entrada da rede; a origem, o da imagem.
  ScaleX := Context.Transform.SourceWidth / MapWidth;
  ScaleY := Context.Transform.SourceHeight / MapHeight;

  Scanner := TComponentScanner.Create(Map, MapWidth, MapHeight,
    Context.BinaryThreshold);
  List := TList<TDetection>.Create;
  try
    Candidates := 0;
    for Index := 0 to MapWidth * MapHeight - 1 do
    begin
      if not Scanner.Extract(Index, Extremes) then
        Continue;

      // max_candidates do Paddle: limita os componentes examinados.
      Inc(Candidates);
      if (Context.MaxDetections > 0) and (Candidates > Context.MaxDetections) then
        Break;

      Rect := MinAreaRect(Extremes);
      if Rect.ShortSide < MIN_SIDE then
        Continue;

      Score := BoxScore(Map, MapWidth, MapHeight, Rect.Corners);
      if Score < Context.ConfidenceThreshold then
        Continue;

      // Unclip: desfaz o encolhimento aplicado aos rotulos no treino.
      Distance := Rect.Width * Rect.Height * Context.UnclipRatio /
                  (2 * (Rect.Width + Rect.Height));
      Rect.Width := Rect.Width + 2 * Distance;
      Rect.Height := Rect.Height + 2 * Distance;
      if Rect.ShortSide < MIN_SIDE + 2 then
        Continue;

      Rect.AlignToReadingDirection;

      Detection := Default(TDetection);
      Detection.ClassId := 0;
      Detection.ClassName := 'texto';
      Detection.Score := Score;
      Detection.SourceIndex := Index;
      Detection.HasObb := True;
      Detection.Obb := ToSourceObb(Rect, ScaleX, ScaleY);
      Detection.Box := Detection.Obb.AxisAlignedBounds.ClampTo(
        Context.Transform.SourceWidth, Context.Transform.SourceHeight);
      if Detection.Box.IsEmpty then
        Continue;

      List.Add(Detection);
    end;

    Result.Detections := List.ToArray;
    SortReadingOrder(Result.Detections);
  finally
    List.Free;
    Scanner.Free;
  end;
end;

initialization
  TDecoderRegistry.RegisterDecoder(vtText,
    function: IResultDecoder
    begin
      Result := TTextDecoder.Create;
    end);

end.
