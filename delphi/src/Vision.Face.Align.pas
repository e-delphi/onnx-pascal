unit Vision.Face.Align;

{
  Alinhamento facial por 5 pontos.

  O ArcFace nao aceita um recorte qualquer: ele foi treinado sobre faces
  levadas por uma transformada de similaridade a um template canonico de
  112x112. Sem esse passo a acuracia cai muito - e a causa mais comum de
  numeros irreproduziveis com esses modelos.

  A transformada tem 4 graus de liberdade (escala, rotacao, tx, ty), entao
  5 pares de pontos ja sobredeterminam o sistema. A solucao de minimos
  quadrados tem forma fechada (Procrustes), o que dispensa SVD:

      x' = a*x - b*y + tx
      y' = b*x + a*y + ty

      a = sum(u*U + v*V) / sum(u^2 + v^2)
      b = sum(u*V - v*U) / sum(u^2 + v^2)

  com u,v e U,V os pontos de origem e destino centrados na media.

  A forma fechada equivale ao Umeyama via SVD para este caso e tem a
  vantagem de nao poder produzir reflexao, que para rosto nunca e desejada.
}

interface

uses
  System.SysUtils,
  System.Types,
  System.Math,
  Vision.Types,
  Vision.Image;

const
  ARCFACE_SIZE = 112;

type
  EAlignError = class(Exception);

  { Matriz afim 2x3 que leva origem -> destino. }
  TSimilarityTransform = record
    A, B, TX, TY: Single;
    function Apply(const P: TPointF): TPointF;
    function Invert(const P: TPointF): TPointF;
    function Scale: Single;
    function RotationDegrees: Single;
    function IsValid: Boolean;
  end;

  IFaceAligner = interface
    ['{7D4E1B85-3A29-4C60-B8F1-5E2A9C07D643}']
    function OutputSize: Integer;
    function Align(const Source: IImage; const Landmarks: TKeypoints): IImage;
    function LastTransform: TSimilarityTransform;
  end;

  { Alinhador para o template canonico do ArcFace (112x112). }
  TArcFaceAligner = class(TInterfacedObject, IFaceAligner)
  private
    FTemplate: TArray<TPointF>;
    FSize: Integer;
    FLast: TSimilarityTransform;
  public
    constructor Create;
    function OutputSize: Integer;
    function Align(const Source: IImage; const Landmarks: TKeypoints): IImage;
    function LastTransform: TSimilarityTransform;
  end;

{ Ajusta a transformada de similaridade que melhor leva Source em Target. }
function EstimateSimilarity(const Source, Target: TArray<TPointF>): TSimilarityTransform;

{ Reamostra Source aplicando a inversa de Transform, produzindo Width x Height. }
function WarpAffine(const Source: IImage; const Transform: TSimilarityTransform;
  Width, Height: Integer): IImage;

{ Template canonico do ArcFace, em coordenadas de 112x112. }
function ArcFaceTemplate: TArray<TPointF>;

implementation

function ArcFaceTemplate: TArray<TPointF>;
begin
  // olho esquerdo, olho direito, nariz, canto esquerdo e direito da boca
  Result := TArray<TPointF>.Create(
    PointF(38.2946, 51.6963),
    PointF(73.5318, 51.5014),
    PointF(56.0252, 71.7366),
    PointF(41.5493, 92.3655),
    PointF(70.7299, 92.2041));
end;

{ TSimilarityTransform }

function TSimilarityTransform.Apply(const P: TPointF): TPointF;
begin
  Result.X := A * P.X - B * P.Y + TX;
  Result.Y := B * P.X + A * P.Y + TY;
end;

function TSimilarityTransform.Invert(const P: TPointF): TPointF;
var
  Det, DX, DY: Single;
begin
  Det := A * A + B * B;
  if Det = 0 then
    Exit(P);

  DX := P.X - TX;
  DY := P.Y - TY;
  Result.X := (A * DX + B * DY) / Det;
  Result.Y := (A * DY - B * DX) / Det;
end;

function TSimilarityTransform.Scale: Single;
begin
  Result := Sqrt(A * A + B * B);
end;

function TSimilarityTransform.RotationDegrees: Single;
begin
  Result := RadToDeg(ArcTan2(B, A));
end;

function TSimilarityTransform.IsValid: Boolean;
begin
  Result := (A * A + B * B) > 1E-12;
end;

{ Estimativa }

function EstimateSimilarity(const Source, Target: TArray<TPointF>): TSimilarityTransform;
var
  N, I: Integer;
  SumSX, SumSY, SumTX, SumTY: Double;
  MeanSX, MeanSY, MeanTX, MeanTY: Double;
  U, V, BigU, BigV: Double;
  Numerator, Cross, Denominator: Double;
begin
  N := Length(Source);
  if (N < 2) or (Length(Target) <> N) then
    raise EAlignError.CreateFmt(
      'Estimativa precisa de pelo menos 2 pares de pontos; recebidos %d e %d',
      [N, Length(Target)]);

  SumSX := 0; SumSY := 0; SumTX := 0; SumTY := 0;
  for I := 0 to N - 1 do
  begin
    SumSX := SumSX + Source[I].X;
    SumSY := SumSY + Source[I].Y;
    SumTX := SumTX + Target[I].X;
    SumTY := SumTY + Target[I].Y;
  end;

  MeanSX := SumSX / N;
  MeanSY := SumSY / N;
  MeanTX := SumTX / N;
  MeanTY := SumTY / N;

  Numerator := 0;
  Cross := 0;
  Denominator := 0;
  for I := 0 to N - 1 do
  begin
    U := Source[I].X - MeanSX;
    V := Source[I].Y - MeanSY;
    BigU := Target[I].X - MeanTX;
    BigV := Target[I].Y - MeanTY;

    Numerator := Numerator + U * BigU + V * BigV;
    Cross := Cross + U * BigV - V * BigU;
    Denominator := Denominator + U * U + V * V;
  end;

  if Denominator <= 1E-12 then
    raise EAlignError.Create(
      'Pontos de origem degenerados (todos coincidentes)');

  Result.A := Numerator / Denominator;
  Result.B := Cross / Denominator;
  Result.TX := MeanTX - (Result.A * MeanSX - Result.B * MeanSY);
  Result.TY := MeanTY - (Result.B * MeanSX + Result.A * MeanSY);
end;

{ Reamostragem }

function WarpAffine(const Source: IImage; const Transform: TSimilarityTransform;
  Width, Height: Integer): IImage;
var
  Dst: TRGBImage;
  X, Y, C, X0, Y0, X1, Y1: Integer;
  P, S: TPointF;
  FX, FY, Value: Single;
  R0, R1, P00, P10, P01, P11: PByte;
  Out_: PByte;
begin
  if Source = nil then
    raise EAlignError.Create('Imagem nula');
  if not Transform.IsValid then
    raise EAlignError.Create('Transformada degenerada');
  if (Width <= 0) or (Height <= 0) then
    raise EAlignError.CreateFmt('Destino invalido: %dx%d', [Width, Height]);

  Dst := TRGBImage.Create(Width, Height);
  Result := Dst;

  for Y := 0 to Height - 1 do
  begin
    Out_ := Dst.RowPtr(Y);
    for X := 0 to Width - 1 do
    begin
      P := PointF(X, Y);
      S := Transform.Invert(P);

      X0 := Floor(S.X);
      Y0 := Floor(S.Y);
      FX := S.X - X0;
      FY := S.Y - Y0;

      // Fora da imagem de origem fica preto, como no warpAffine da OpenCV
      // com borderValue 0.
      if (X0 < 0) or (Y0 < 0) or
         (X0 >= Source.Width) or (Y0 >= Source.Height) then
      begin
        PByte(Out_ + 0)^ := 0;
        PByte(Out_ + 1)^ := 0;
        PByte(Out_ + 2)^ := 0;
        Inc(Out_, 3);
        Continue;
      end;

      X1 := Min(X0 + 1, Source.Width - 1);
      Y1 := Min(Y0 + 1, Source.Height - 1);

      R0 := Source.RowPtr(Y0);
      R1 := Source.RowPtr(Y1);
      P00 := R0 + X0 * 3;
      P10 := R0 + X1 * 3;
      P01 := R1 + X0 * 3;
      P11 := R1 + X1 * 3;

      for C := 0 to 2 do
      begin
        Value := PByte(P00 + C)^ * (1 - FX) * (1 - FY) +
                 PByte(P10 + C)^ * FX * (1 - FY) +
                 PByte(P01 + C)^ * (1 - FX) * FY +
                 PByte(P11 + C)^ * FX * FY;
        PByte(Out_ + C)^ := Byte(Round(Min(255.0, Max(0.0, Value))));
      end;

      Inc(Out_, 3);
    end;
  end;
end;

{ TArcFaceAligner }

constructor TArcFaceAligner.Create;
begin
  inherited Create;
  FTemplate := ArcFaceTemplate;
  FSize := ARCFACE_SIZE;
end;

function TArcFaceAligner.OutputSize: Integer;
begin
  Result := FSize;
end;

function TArcFaceAligner.LastTransform: TSimilarityTransform;
begin
  Result := FLast;
end;

function TArcFaceAligner.Align(const Source: IImage;
  const Landmarks: TKeypoints): IImage;
var
  Points: TArray<TPointF>;
  I: Integer;
begin
  if Length(Landmarks) < Length(FTemplate) then
    raise EAlignError.CreateFmt(
      'O alinhamento do ArcFace precisa de %d landmarks; recebidos %d',
      [Length(FTemplate), Length(Landmarks)]);

  SetLength(Points, Length(FTemplate));
  for I := 0 to High(Points) do
    Points[I] := PointF(Landmarks[I].X, Landmarks[I].Y);

  FLast := EstimateSimilarity(Points, FTemplate);
  Result := WarpAffine(Source, FLast, FSize, FSize);
end;

end.
