unit Vision.Types;

{
  Vocabulario do dominio de visao computacional.

  Nao depende de ONNX nem de VCL: sao apenas os tipos que a aplicacao,
  os decoders e o renderizador trocam entre si.
}

interface

uses
  System.SysUtils,
  System.Types,
  System.Math;

type
  TVisionTask = (
    vtUnknown,
    vtClassify,   // [1, nc]                     -> rotulo mais provavel
    vtDetect,     // [1, 4+nc, N] ou [1, N, 6]   -> caixas
    vtSegment,    // detect + protos             -> caixas + mascaras
    vtPose,       // detect + keypoints          -> caixas + esqueleto
    vtObb,        // detect + angulo             -> caixas rotacionadas
    vtFace,       // SCRFD: 9 saidas             -> caixas + 5 landmarks
    vtEmbed       // [1, D]                      -> vetor de identidade
  );

  TBoxF = record
    Left, Top, Right, Bottom: Single;
    class function FromLTRB(ALeft, ATop, ARight, ABottom: Single): TBoxF; static;
    class function FromCenter(CX, CY, W, H: Single): TBoxF; static;
    function Width: Single;
    function Height: Single;
    function Area: Single;
    function CenterX: Single;
    function CenterY: Single;
    function Normalized: TBoxF;
    function ClampTo(MaxWidth, MaxHeight: Single): TBoxF;
    function IntersectionArea(const Other: TBoxF): Single;
    function IoU(const Other: TBoxF): Single;
    function IsEmpty: Boolean;
    function ToString: string;
  end;

  TKeypoint = record
    X, Y: Single;
    Score: Single;
  end;
  TKeypoints = TArray<TKeypoint>;

  { Caixa rotacionada no sistema de coordenadas da imagem original.
    Angle em radianos, sentido horario (convencao Ultralytics OBB). }
  TObbBox = record
    CX, CY, W, H, Angle: Single;
    function Corners: TArray<TPointF>;
    function AxisAlignedBounds: TBoxF;
    function Area: Single;
  end;

  { Mascara binaria/probabilistica recortada na caixa da deteccao,
    ja no sistema de coordenadas da imagem original. }
  TMaskData = record
    OffsetX, OffsetY: Integer;
    Width, Height: Integer;
    Values: TArray<Byte>;   // 0..255, probabilidade * 255
    function IsValid: Boolean;
    function ValueAt(X, Y: Integer): Byte;
  end;

  TDetection = record
    ClassId: Integer;
    ClassName: string;
    Score: Single;
    { Linha do tensor de predicao que originou esta deteccao. Sobrevive ao
      NMS, o que permite ao decoder de segmentacao buscar os coeficientes de
      mascara apenas das deteccoes que restaram. }
    SourceIndex: Integer;
    Box: TBoxF;
    HasObb: Boolean;
    Obb: TObbBox;
    HasKeypoints: Boolean;
    Keypoints: TKeypoints;
    HasMask: Boolean;
    Mask: TMaskData;
    function DisplayName: string;
  end;

  TDetections = TArray<TDetection>;

  TClassScore = record
    ClassId: Integer;
    ClassName: string;
    Score: Single;
  end;

  TClassScores = TArray<TClassScore>;

  TVisionResult = record
    Task: TVisionTask;
    ImageWidth: Integer;
    ImageHeight: Integer;
    Detections: TDetections;
    Classes: TClassScores;
    { Preenchido pela tarefa vtEmbed: vetor ja L2-normalizado. }
    Embedding: TArray<Single>;
    PreprocessMs: Double;
    InferenceMs: Double;
    PostprocessMs: Double;
    function TotalMs: Double;
    function Count: Integer;
  end;

function VisionTaskToString(Task: TVisionTask): string;
function StringToVisionTask(const Value: string): TVisionTask;

function Sigmoid(X: Single): Single;
procedure SoftmaxInPlace(var Values: TArray<Single>);
function LooksLikeProbabilityVector(const Values: TArray<Single>): Boolean;

implementation

{ TBoxF }

class function TBoxF.FromLTRB(ALeft, ATop, ARight, ABottom: Single): TBoxF;
begin
  Result.Left := ALeft;
  Result.Top := ATop;
  Result.Right := ARight;
  Result.Bottom := ABottom;
end;

class function TBoxF.FromCenter(CX, CY, W, H: Single): TBoxF;
begin
  Result.Left := CX - W * 0.5;
  Result.Top := CY - H * 0.5;
  Result.Right := CX + W * 0.5;
  Result.Bottom := CY + H * 0.5;
end;

function TBoxF.Width: Single;
begin
  Result := Right - Left;
end;

function TBoxF.Height: Single;
begin
  Result := Bottom - Top;
end;

function TBoxF.Area: Single;
begin
  Result := Max(0.0, Width) * Max(0.0, Height);
end;

function TBoxF.CenterX: Single;
begin
  Result := (Left + Right) * 0.5;
end;

function TBoxF.CenterY: Single;
begin
  Result := (Top + Bottom) * 0.5;
end;

function TBoxF.Normalized: TBoxF;
begin
  Result.Left := Min(Left, Right);
  Result.Right := Max(Left, Right);
  Result.Top := Min(Top, Bottom);
  Result.Bottom := Max(Top, Bottom);
end;

function TBoxF.ClampTo(MaxWidth, MaxHeight: Single): TBoxF;
begin
  Result := Normalized;
  Result.Left := Min(Max(Result.Left, 0.0), MaxWidth);
  Result.Top := Min(Max(Result.Top, 0.0), MaxHeight);
  Result.Right := Min(Max(Result.Right, 0.0), MaxWidth);
  Result.Bottom := Min(Max(Result.Bottom, 0.0), MaxHeight);
end;

function TBoxF.IntersectionArea(const Other: TBoxF): Single;
var
  IL, IT, IR, IB: Single;
begin
  IL := Max(Left, Other.Left);
  IT := Max(Top, Other.Top);
  IR := Min(Right, Other.Right);
  IB := Min(Bottom, Other.Bottom);
  if (IR <= IL) or (IB <= IT) then
    Exit(0);
  Result := (IR - IL) * (IB - IT);
end;

function TBoxF.IoU(const Other: TBoxF): Single;
var
  Inter, Union: Single;
begin
  Inter := IntersectionArea(Other);
  if Inter <= 0 then
    Exit(0);
  Union := Area + Other.Area - Inter;
  if Union <= 0 then
    Exit(0);
  Result := Inter / Union;
end;

function TBoxF.IsEmpty: Boolean;
begin
  Result := (Width <= 0) or (Height <= 0);
end;

function TBoxF.ToString: string;
begin
  Result := Format('(%.1f, %.1f)-(%.1f, %.1f)  %.0fx%.0f',
    [Left, Top, Right, Bottom, Width, Height]);
end;

{ TObbBox }

function TObbBox.Corners: TArray<TPointF>;
var
  C, S, HW, HH: Single;
  DX1, DY1, DX2, DY2: Single;
begin
  C := Cos(Angle);
  S := Sin(Angle);
  HW := W * 0.5;
  HH := H * 0.5;

  DX1 := HW * C;
  DY1 := HW * S;
  DX2 := HH * S;
  DY2 := HH * C;

  SetLength(Result, 4);
  Result[0] := PointF(CX - DX1 + DX2, CY - DY1 - DY2);
  Result[1] := PointF(CX + DX1 + DX2, CY + DY1 - DY2);
  Result[2] := PointF(CX + DX1 - DX2, CY + DY1 + DY2);
  Result[3] := PointF(CX - DX1 - DX2, CY - DY1 + DY2);
end;

function TObbBox.AxisAlignedBounds: TBoxF;
var
  Pts: TArray<TPointF>;
  I: Integer;
begin
  Pts := Corners;
  Result.Left := Pts[0].X;
  Result.Right := Pts[0].X;
  Result.Top := Pts[0].Y;
  Result.Bottom := Pts[0].Y;
  for I := 1 to High(Pts) do
  begin
    Result.Left := Min(Result.Left, Pts[I].X);
    Result.Right := Max(Result.Right, Pts[I].X);
    Result.Top := Min(Result.Top, Pts[I].Y);
    Result.Bottom := Max(Result.Bottom, Pts[I].Y);
  end;
end;

function TObbBox.Area: Single;
begin
  Result := Abs(W) * Abs(H);
end;

{ TMaskData }

function TMaskData.IsValid: Boolean;
begin
  Result := (Width > 0) and (Height > 0) and (Length(Values) = Width * Height);
end;

function TMaskData.ValueAt(X, Y: Integer): Byte;
begin
  if (X < 0) or (Y < 0) or (X >= Width) or (Y >= Height) then
    Exit(0);
  Result := Values[Y * Width + X];
end;

{ TDetection }

function TDetection.DisplayName: string;
begin
  if ClassName <> '' then
    Result := ClassName
  else
    Result := Format('classe_%d', [ClassId]);
end;

{ TVisionResult }

function TVisionResult.TotalMs: Double;
begin
  Result := PreprocessMs + InferenceMs + PostprocessMs;
end;

function TVisionResult.Count: Integer;
begin
  if Task = vtEmbed then
    Result := Length(Embedding)
  else if Task = vtClassify then
    Result := Length(Classes)
  else
    Result := Length(Detections);
end;

{ helpers }

function VisionTaskToString(Task: TVisionTask): string;
begin
  case Task of
    vtClassify: Result := 'classify';
    vtDetect:   Result := 'detect';
    vtSegment:  Result := 'segment';
    vtPose:     Result := 'pose';
    vtObb:      Result := 'obb';
    vtFace:     Result := 'face';
    vtEmbed:    Result := 'embed';
  else
    Result := 'unknown';
  end;
end;

function StringToVisionTask(const Value: string): TVisionTask;
var
  V: string;
begin
  V := LowerCase(Trim(Value));
  if (V = 'classify') or (V = 'cls') or (V = 'classification') then
    Result := vtClassify
  else if (V = 'detect') or (V = 'detection') or (V = 'det') then
    Result := vtDetect
  else if (V = 'segment') or (V = 'seg') or (V = 'segmentation') then
    Result := vtSegment
  else if V = 'pose' then
    Result := vtPose
  else if V = 'obb' then
    Result := vtObb
  else if (V = 'face') or (V = 'facedetect') then
    Result := vtFace
  else if (V = 'embed') or (V = 'embedding') or (V = 'arcface') then
    Result := vtEmbed
  else
    Result := vtUnknown;
end;

function Sigmoid(X: Single): Single;
begin
  if X >= 0 then
    Result := 1 / (1 + Exp(-X))
  else
    Result := Exp(X) / (1 + Exp(X));
end;

procedure SoftmaxInPlace(var Values: TArray<Single>);
var
  I: Integer;
  MaxV, Sum: Double;
begin
  if Length(Values) = 0 then
    Exit;

  MaxV := Values[0];
  for I := 1 to High(Values) do
    if Values[I] > MaxV then
      MaxV := Values[I];

  Sum := 0;
  for I := 0 to High(Values) do
  begin
    Values[I] := Exp(Values[I] - MaxV);
    Sum := Sum + Values[I];
  end;

  if Sum <= 0 then
    Exit;

  for I := 0 to High(Values) do
    Values[I] := Values[I] / Sum;
end;

function LooksLikeProbabilityVector(const Values: TArray<Single>): Boolean;
var
  I: Integer;
  Sum: Double;
begin
  if Length(Values) = 0 then
    Exit(False);

  Sum := 0;
  for I := 0 to High(Values) do
  begin
    if (Values[I] < -0.001) or (Values[I] > 1.001) then
      Exit(False);
    Sum := Sum + Values[I];
  end;

  Result := Abs(Sum - 1.0) < 0.05;
end;

end.
