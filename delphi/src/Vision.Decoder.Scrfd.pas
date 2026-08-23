unit Vision.Decoder.Scrfd;

{
  Cabeca de deteccao facial SCRFD (det_10g do pacote buffalo_l da InsightFace).

  Formato das saidas:

    9 saidas, rank 2, agrupadas em 3 niveis de piramide:

      score  [N, 1]    ja com sigmoid aplicado (faixa 0..1)
      bbox   [N, 4]    distancias l,t,r,b do centro da ancora, em unidades
                       de stride (multiplicar por stride)
      kps    [N, 10]   deslocamentos x,y de 5 pontos a partir do centro da
                       ancora, tambem em unidades de stride

    N = (netW / stride) * (netH / stride) * ancoras, com 2 ancoras por
    celula. Para entrada 640x640: 12800 / 3200 / 800 para strides 8/16/32.

  Os nomes das saidas sao numericos ("448", "471", ...) e nao servem de
  referencia: o agrupamento e feito por shape, e o stride de cada grupo e
  derivado resolvendo N = (netW div s) * (netH div s) * A. Se nenhuma
  combinacao fechar, o decoder levanta erro.

  Os 5 pontos saem na ordem: olho esquerdo, olho direito, nariz, canto
  esquerdo da boca, canto direito da boca - que e exatamente a ordem que o
  template canonico do ArcFace espera.
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

const
  SCRFD_KEYPOINTS = 5;
  SCRFD_STRIDES: array[0..2] of Integer = (8, 16, 32);

type
  TScrfdLevel = record
    Stride: Integer;
    Anchors: Integer;      // ancoras por celula
    GridWidth: Integer;
    GridHeight: Integer;
    Rows: Integer;         // N
    Score: TTensor;
    Box: TTensor;
    Keypoints: TTensor;
    function IsComplete: Boolean;
  end;

  TScrfdDecoder = class(TDetectionDecoderBase, IResultDecoder)
  private
    function CollectLevels(const Outputs: TTensorArray;
      NetWidth, NetHeight: Integer): TArray<TScrfdLevel>;
    function ResolveStride(Rows, NetWidth, NetHeight: Integer;
      out Stride, Anchors, GridWidth, GridHeight: Integer): Boolean;
    procedure DecodeLevel(const Level: TScrfdLevel;
      const Context: TDecodeContext; Target: TList<TDetection>);
  public
    function Task: TVisionTask;
    function Describe: string;
    function Decode(const Outputs: TTensorArray;
      const Context: TDecodeContext): TVisionResult;
  end;

implementation

{ TScrfdLevel }

function TScrfdLevel.IsComplete: Boolean;
begin
  Result := (Rows > 0) and
            (Length(Score.Data) >= Rows) and
            (Length(Box.Data) >= Rows * 4) and
            (Length(Keypoints.Data) >= Rows * SCRFD_KEYPOINTS * 2);
end;

{ TScrfdDecoder }

function TScrfdDecoder.Task: TVisionTask;
begin
  Result := vtFace;
end;

function TScrfdDecoder.Describe: string;
begin
  Result := 'deteccao facial SCRFD (9 saidas, 3 strides, 5 landmarks)';
end;

function TScrfdDecoder.ResolveStride(Rows, NetWidth, NetHeight: Integer;
  out Stride, Anchors, GridWidth, GridHeight: Integer): Boolean;
var
  I, S, GW, GH, Cells, A: Integer;
begin
  for I := Low(SCRFD_STRIDES) to High(SCRFD_STRIDES) do
  begin
    S := SCRFD_STRIDES[I];
    GW := NetWidth div S;
    GH := NetHeight div S;
    Cells := GW * GH;
    if Cells <= 0 then
      Continue;

    if Rows mod Cells <> 0 then
      Continue;

    A := Rows div Cells;
    if (A < 1) or (A > 4) then
      Continue;

    Stride := S;
    Anchors := A;
    GridWidth := GW;
    GridHeight := GH;
    Exit(True);
  end;

  Stride := 0;
  Anchors := 0;
  GridWidth := 0;
  GridHeight := 0;
  Result := False;
end;

function TScrfdDecoder.CollectLevels(const Outputs: TTensorArray;
  NetWidth, NetHeight: Integer): TArray<TScrfdLevel>;
var
  Levels: TList<TScrfdLevel>;
  Level: TScrfdLevel;
  Index, I, Rows, Columns: Integer;
  Stride, Anchors, GridWidth, GridHeight: Integer;

  function IndexOfStride(S: Integer): Integer;
  var
    K: Integer;
  begin
    for K := 0 to Levels.Count - 1 do
      if Levels[K].Stride = S then
        Exit(K);
    Result := -1;
  end;

begin
  Levels := TList<TScrfdLevel>.Create;
  try
    for I := 0 to High(Outputs) do
    begin
      if Outputs[I].Rank <> 2 then
        Continue;

      Rows := Outputs[I].DimAsInt(0);
      Columns := Outputs[I].DimAsInt(1);
      if (Rows <= 0) or (Columns <= 0) then
        Continue;

      if not ResolveStride(Rows, NetWidth, NetHeight,
           Stride, Anchors, GridWidth, GridHeight) then
        raise EDecodeError.CreateFmt(
          'Saida SCRFD com %d linhas nao corresponde a nenhum stride para ' +
          'entrada %dx%d', [Rows, NetWidth, NetHeight]);

      Index := IndexOfStride(Stride);
      if Index < 0 then
      begin
        Level := Default(TScrfdLevel);
        Level.Stride := Stride;
        Level.Anchors := Anchors;
        Level.GridWidth := GridWidth;
        Level.GridHeight := GridHeight;
        Level.Rows := Rows;
        Levels.Add(Level);
        Index := Levels.Count - 1;
      end;

      Level := Levels[Index];
      case Columns of
        1:                    Level.Score := Outputs[I];
        4:                    Level.Box := Outputs[I];
        SCRFD_KEYPOINTS * 2:  Level.Keypoints := Outputs[I];
      else
        raise EDecodeError.CreateFmt(
          'Saida SCRFD com %d colunas nao reconhecida (esperado 1, 4 ou %d)',
          [Columns, SCRFD_KEYPOINTS * 2]);
      end;
      Levels[Index] := Level;
    end;

    for I := 0 to Levels.Count - 1 do
      if not Levels[I].IsComplete then
        raise EDecodeError.CreateFmt(
          'Nivel de stride %d incompleto: faltou score, caixa ou landmarks',
          [Levels[I].Stride]);

    Result := Levels.ToArray;
  finally
    Levels.Free;
  end;
end;

procedure TScrfdDecoder.DecodeLevel(const Level: TScrfdLevel;
  const Context: TDecodeContext; Target: TList<TDetection>);
var
  Row, Cell, GX, GY, K: Integer;
  CenterX, CenterY, Score: Single;
  L, T, R, B: Single;
  Detection: TDetection;
begin
  for Row := 0 to Level.Rows - 1 do
  begin
    Score := Level.Score.Data[Row];
    if Score < Context.ConfidenceThreshold then
      Continue;

    // Ancoras consecutivas compartilham a mesma celula.
    Cell := Row div Level.Anchors;
    GX := Cell mod Level.GridWidth;
    GY := Cell div Level.GridWidth;
    CenterX := GX * Level.Stride;
    CenterY := GY * Level.Stride;

    L := Level.Box.Data[Row * 4 + 0] * Level.Stride;
    T := Level.Box.Data[Row * 4 + 1] * Level.Stride;
    R := Level.Box.Data[Row * 4 + 2] * Level.Stride;
    B := Level.Box.Data[Row * 4 + 3] * Level.Stride;

    Detection := Default(TDetection);
    Detection.ClassId := 0;
    Detection.ClassName := 'face';
    Detection.Score := Score;
    Detection.SourceIndex := Row;
    Detection.Box := TBoxF.FromLTRB(
      Context.Transform.NetToSourceX(CenterX - L),
      Context.Transform.NetToSourceY(CenterY - T),
      Context.Transform.NetToSourceX(CenterX + R),
      Context.Transform.NetToSourceY(CenterY + B));

    if Detection.Box.IsEmpty then
      Continue;

    Detection.HasKeypoints := True;
    SetLength(Detection.Keypoints, SCRFD_KEYPOINTS);
    for K := 0 to SCRFD_KEYPOINTS - 1 do
    begin
      Detection.Keypoints[K].X := Context.Transform.NetToSourceX(
        CenterX + Level.Keypoints.Data[Row * SCRFD_KEYPOINTS * 2 + K * 2] * Level.Stride);
      Detection.Keypoints[K].Y := Context.Transform.NetToSourceY(
        CenterY + Level.Keypoints.Data[Row * SCRFD_KEYPOINTS * 2 + K * 2 + 1] * Level.Stride);
      // O SCRFD nao emite confianca por ponto; usa a da propria deteccao.
      Detection.Keypoints[K].Score := Score;
    end;

    Target.Add(Detection);
  end;
end;

function TScrfdDecoder.Decode(const Outputs: TTensorArray;
  const Context: TDecodeContext): TVisionResult;
var
  Levels: TArray<TScrfdLevel>;
  Candidates: TList<TDetection>;
  I: Integer;
begin
  Result := Default(TVisionResult);
  Result.Task := vtFace;
  Result.ImageWidth := Context.Transform.SourceWidth;
  Result.ImageHeight := Context.Transform.SourceHeight;

  if Length(Outputs) < 3 then
    raise EDecodeError.CreateFmt(
      'SCRFD esperava 9 saidas (3 por stride); recebidas %d', [Length(Outputs)]);

  Levels := CollectLevels(Outputs,
    Context.Transform.NetWidth, Context.Transform.NetHeight);

  Candidates := TList<TDetection>.Create;
  try
    for I := 0 to High(Levels) do
      DecodeLevel(Levels[I], Context, Candidates);

    // A caixa e recortada so depois do NMS: os landmarks podem cair fora da
    // imagem em rostos nas bordas, e recortar antes deslocaria o IoU.
    Result.Detections := FinalizeDetections(Candidates.ToArray, Context, False);
  finally
    Candidates.Free;
  end;

  for I := 0 to High(Result.Detections) do
    Result.Detections[I].Box := Result.Detections[I].Box.ClampTo(
      Context.Transform.SourceWidth, Context.Transform.SourceHeight);
end;

initialization
  TDecoderRegistry.RegisterDecoder(vtFace,
    function: IResultDecoder
    begin
      Result := TScrfdDecoder.Create;
    end);

end.
