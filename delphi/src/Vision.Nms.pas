unit Vision.Nms;

{
  Supressao de nao-maximos.

  Modelos YOLO26 exportados em modo end-to-end ja saem filtrados e nao passam
  por aqui. Continua sendo necessario para YOLOv5/v8/v11 e para YOLO26
  exportado com end2end=False.
}

interface

uses
  System.SysUtils,
  System.Types,
  System.Math,
  System.Generics.Collections,
  System.Generics.Defaults,
  Vision.Types;

type
  TNmsMode = (nmClassAware, nmClassAgnostic);

{ Ordena por score decrescente (estavel o suficiente para o uso aqui). }
procedure SortDetectionsByScore(var Detections: TDetections);

function ApplyNms(const Detections: TDetections; IoUThreshold: Single;
  Mode: TNmsMode = nmClassAware; MaxOutput: Integer = 0): TDetections;

{ NMS para caixas rotacionadas: usa a area de intersecao real entre os dois
  quadrilateros (Sutherland-Hodgman + shoelace). }
function ApplyRotatedNms(const Detections: TDetections; IoUThreshold: Single;
  Mode: TNmsMode = nmClassAware; MaxOutput: Integer = 0): TDetections;

function PolygonArea(const Points: TArray<TPointF>): Single;
function ConvexPolygonIntersection(const A, B: TArray<TPointF>): TArray<TPointF>;
function RotatedIoU(const A, B: TObbBox): Single;

implementation

procedure SortDetectionsByScore(var Detections: TDetections);
begin
  TArray.Sort<TDetection>(Detections,
    TComparer<TDetection>.Construct(
      function(const L, R: TDetection): Integer
      begin
        if L.Score > R.Score then
          Result := -1
        else if L.Score < R.Score then
          Result := 1
        else
          Result := 0;
      end));
end;

function ApplyNms(const Detections: TDetections; IoUThreshold: Single;
  Mode: TNmsMode; MaxOutput: Integer): TDetections;
var
  Sorted: TDetections;
  Suppressed: TArray<Boolean>;
  Kept: TList<TDetection>;
  I, J: Integer;
begin
  Result := nil;
  if Length(Detections) = 0 then
    Exit;

  Sorted := Copy(Detections);
  SortDetectionsByScore(Sorted);

  SetLength(Suppressed, Length(Sorted));
  Kept := TList<TDetection>.Create;
  try
    for I := 0 to High(Sorted) do
    begin
      if Suppressed[I] then
        Continue;

      Kept.Add(Sorted[I]);
      if (MaxOutput > 0) and (Kept.Count >= MaxOutput) then
        Break;

      for J := I + 1 to High(Sorted) do
      begin
        if Suppressed[J] then
          Continue;
        if (Mode = nmClassAware) and (Sorted[J].ClassId <> Sorted[I].ClassId) then
          Continue;
        if Sorted[I].Box.IoU(Sorted[J].Box) > IoUThreshold then
          Suppressed[J] := True;
      end;
    end;

    Result := Kept.ToArray;
  finally
    Kept.Free;
  end;
end;

function ApplyRotatedNms(const Detections: TDetections; IoUThreshold: Single;
  Mode: TNmsMode; MaxOutput: Integer): TDetections;
var
  Sorted: TDetections;
  Suppressed: TArray<Boolean>;
  Kept: TList<TDetection>;
  I, J: Integer;
begin
  Result := nil;
  if Length(Detections) = 0 then
    Exit;

  Sorted := Copy(Detections);
  SortDetectionsByScore(Sorted);

  SetLength(Suppressed, Length(Sorted));
  Kept := TList<TDetection>.Create;
  try
    for I := 0 to High(Sorted) do
    begin
      if Suppressed[I] then
        Continue;

      Kept.Add(Sorted[I]);
      if (MaxOutput > 0) and (Kept.Count >= MaxOutput) then
        Break;

      for J := I + 1 to High(Sorted) do
      begin
        if Suppressed[J] then
          Continue;
        if (Mode = nmClassAware) and (Sorted[J].ClassId <> Sorted[I].ClassId) then
          Continue;
        // Descarta rapido pelo envelope alinhado antes do calculo caro.
        if Sorted[I].Box.IoU(Sorted[J].Box) <= 0 then
          Continue;
        if RotatedIoU(Sorted[I].Obb, Sorted[J].Obb) > IoUThreshold then
          Suppressed[J] := True;
      end;
    end;

    Result := Kept.ToArray;
  finally
    Kept.Free;
  end;
end;

function PolygonArea(const Points: TArray<TPointF>): Single;
var
  I, J, N: Integer;
  Sum: Double;
begin
  N := Length(Points);
  if N < 3 then
    Exit(0);

  Sum := 0;
  J := N - 1;
  for I := 0 to N - 1 do
  begin
    Sum := Sum + (Points[J].X * Points[I].Y - Points[I].X * Points[J].Y);
    J := I;
  end;
  Result := Abs(Sum) * 0.5;
end;

function ConvexPolygonIntersection(const A, B: TArray<TPointF>): TArray<TPointF>;

  function IsInside(const P, EdgeStart, EdgeEnd: TPointF): Boolean;
  begin
    // >= 0 mantem pontos sobre a aresta; orientacao anti-horaria assumida
    // depois da normalizacao feita abaixo.
    Result := ((EdgeEnd.X - EdgeStart.X) * (P.Y - EdgeStart.Y) -
               (EdgeEnd.Y - EdgeStart.Y) * (P.X - EdgeStart.X)) >= -1E-6;
  end;

  function LineIntersection(const P1, P2, P3, P4: TPointF): TPointF;
  var
    A1, B1, C1, A2, B2, C2, Det: Double;
  begin
    A1 := P2.Y - P1.Y;
    B1 := P1.X - P2.X;
    C1 := A1 * P1.X + B1 * P1.Y;

    A2 := P4.Y - P3.Y;
    B2 := P3.X - P4.X;
    C2 := A2 * P3.X + B2 * P3.Y;

    Det := A1 * B2 - A2 * B1;
    if Abs(Det) < 1E-12 then
      Exit(P2);

    Result.X := (B2 * C1 - B1 * C2) / Det;
    Result.Y := (A1 * C2 - A2 * C1) / Det;
  end;

  function SignedArea(const Poly: TArray<TPointF>): Double;
  var
    I, J, N: Integer;
  begin
    N := Length(Poly);
    Result := 0;
    if N < 3 then
      Exit;
    J := N - 1;
    for I := 0 to N - 1 do
    begin
      Result := Result + (Poly[J].X * Poly[I].Y - Poly[I].X * Poly[J].Y);
      J := I;
    end;
    Result := Result * 0.5;
  end;

  function EnsureCounterClockwise(const Poly: TArray<TPointF>): TArray<TPointF>;
  var
    I, N: Integer;
  begin
    if SignedArea(Poly) >= 0 then
      Exit(Poly);
    N := Length(Poly);
    SetLength(Result, N);
    for I := 0 to N - 1 do
      Result[I] := Poly[N - 1 - I];
  end;

var
  Subject, Clip, Output: TArray<TPointF>;
  I, J, N: Integer;
  Current, Previous, EdgeStart, EdgeEnd: TPointF;
  CurrentInside, PreviousInside: Boolean;
  Buffer: TList<TPointF>;
begin
  Result := nil;
  if (Length(A) < 3) or (Length(B) < 3) then
    Exit;

  Subject := EnsureCounterClockwise(A);
  Clip := EnsureCounterClockwise(B);

  Buffer := TList<TPointF>.Create;
  try
    Output := Subject;

    for I := 0 to High(Clip) do
    begin
      if Length(Output) = 0 then
        Break;

      EdgeStart := Clip[I];
      if I = High(Clip) then
        EdgeEnd := Clip[0]
      else
        EdgeEnd := Clip[I + 1];

      Buffer.Clear;
      N := Length(Output);
      Previous := Output[N - 1];
      PreviousInside := IsInside(Previous, EdgeStart, EdgeEnd);

      for J := 0 to N - 1 do
      begin
        Current := Output[J];
        CurrentInside := IsInside(Current, EdgeStart, EdgeEnd);

        if CurrentInside then
        begin
          if not PreviousInside then
            Buffer.Add(LineIntersection(Previous, Current, EdgeStart, EdgeEnd));
          Buffer.Add(Current);
        end
        else if PreviousInside then
          Buffer.Add(LineIntersection(Previous, Current, EdgeStart, EdgeEnd));

        Previous := Current;
        PreviousInside := CurrentInside;
      end;

      Output := Buffer.ToArray;
    end;

    Result := Output;
  finally
    Buffer.Free;
  end;
end;

function RotatedIoU(const A, B: TObbBox): Single;
var
  Inter: TArray<TPointF>;
  InterArea, Union: Single;
begin
  Inter := ConvexPolygonIntersection(A.Corners, B.Corners);
  InterArea := PolygonArea(Inter);
  if InterArea <= 0 then
    Exit(0);

  Union := A.Area + B.Area - InterArea;
  if Union <= 0 then
    Exit(0);

  Result := InterArea / Union;
end;

end.
