unit Vision.Preprocess;

{
  Transformacao imagem -> tensor.

  Cada familia de modelo tem a sua propria convencao de pre-processamento.
  Em vez de espalhar isso pelo programa principal, cada convencao e uma
  classe que implementa IImagePreprocessor (Open/Closed: modelo novo com
  regra nova = classe nova, nenhum arquivo existente muda).

  O preprocessor tambem devolve a transformacao geometrica aplicada, que e
  o que permite ao decoder mapear caixas/keypoints de volta para as
  coordenadas da imagem original.
}

interface

uses
  System.SysUtils,
  System.Math,
  Vision.Image;

type
  EPreprocessError = class(Exception);

  { srcX = (netX - PadX) / Scale }
  TGeometryTransform = record
    Scale: Single;
    PadX: Single;
    PadY: Single;
    SourceWidth: Integer;
    SourceHeight: Integer;
    NetWidth: Integer;
    NetHeight: Integer;
    class function Identity(ASourceWidth, ASourceHeight: Integer): TGeometryTransform; static;
    function NetToSourceX(X: Single): Single;
    function NetToSourceY(Y: Single): Single;
    function NetToSourceLength(L: Single): Single;
    function SourceToNetX(X: Single): Single;
    function SourceToNetY(Y: Single): Single;
  end;

  TPreparedInput = record
    Data: TArray<Single>;
    Shape: TArray<Int64>;
  end;

  IImagePreprocessor = interface
    ['{1E7B4C05-2D68-4A3F-9C1B-6E0A8D5F7C22}']
    function Describe: string;
    { Numero de passadas (test time augmentation). 1 no caso normal. }
    function PassCount: Integer;
    function Prepare(const Image: IImage; Pass, NetWidth, NetHeight: Integer;
      out Transform: TGeometryTransform): TPreparedInput;
  end;

  { Onde a imagem redimensionada e colada na tela da rede.
      laCenter  - convencao YOLO: sobra dividida entre os dois lados;
      laTopLeft - convencao SCRFD/InsightFace: tudo colado em (0,0). }
  TLetterboxAlign = (laCenter, laTopLeft);

  { Convencao YOLO (v5/v8/v11/26): redimensiona mantendo proporcao,
    preenche o resto com cinza 114 e normaliza para 0..1 em RGB NCHW.
    Com AAlign = laTopLeft e AScale/AMean ajustados serve tambem ao SCRFD. }
  TLetterboxPreprocessor = class(TInterfacedObject, IImagePreprocessor)
  private
    FFillR, FFillG, FFillB: Byte;
    FAllowUpscale: Boolean;
    FAlign: TLetterboxAlign;
    FMean: Single;
    FStd: Single;
  public
    constructor Create(AAllowUpscale: Boolean = True;
      AFillR: Byte = 114; AFillG: Byte = 114; AFillB: Byte = 114;
      AAlign: TLetterboxAlign = laCenter;
      AMean: Single = 0; AStd: Single = 255);
    { Pre-ajuste do SCRFD: cola no canto, fundo preto, (x - 127.5) / 128. }
    class function Scrfd: TLetterboxPreprocessor; static;
    function Describe: string;
    function PassCount: Integer;
    function Prepare(const Image: IImage; Pass, NetWidth, NetHeight: Integer;
      out Transform: TGeometryTransform): TPreparedInput;
  end;

  { Convencao de classificadores ImageNet: redimensiona o lado menor,
    recorta no centro e normaliza com media/desvio por canal.

    Com AMultiCrop = True executa 5 passadas (centro + 4 cantos), que e o
    esquema de test time augmentation usado na versao original do projeto. }
  TCropClassifierPreprocessor = class(TInterfacedObject, IImagePreprocessor)
  private
    FMean: array[0..2] of Single;
    FStd: array[0..2] of Single;
    FCropRatio: Single;
    FMultiCrop: Boolean;
  public
    constructor Create(const AMean, AStd: TArray<Single>;
      ACropRatio: Single = 0.875; AMultiCrop: Boolean = False);
    class function ImageNet(AMultiCrop: Boolean = False): TCropClassifierPreprocessor; static;
    class function UnitScale(AMultiCrop: Boolean = False): TCropClassifierPreprocessor; static;
    function Describe: string;
    function PassCount: Integer;
    function Prepare(const Image: IImage; Pass, NetWidth, NetHeight: Integer;
      out Transform: TGeometryTransform): TPreparedInput;
  end;

implementation

{ TGeometryTransform }

class function TGeometryTransform.Identity(ASourceWidth,
  ASourceHeight: Integer): TGeometryTransform;
begin
  Result.Scale := 1;
  Result.PadX := 0;
  Result.PadY := 0;
  Result.SourceWidth := ASourceWidth;
  Result.SourceHeight := ASourceHeight;
  Result.NetWidth := ASourceWidth;
  Result.NetHeight := ASourceHeight;
end;

function TGeometryTransform.NetToSourceX(X: Single): Single;
begin
  if Scale = 0 then
    Exit(X);
  Result := (X - PadX) / Scale;
end;

function TGeometryTransform.NetToSourceY(Y: Single): Single;
begin
  if Scale = 0 then
    Exit(Y);
  Result := (Y - PadY) / Scale;
end;

function TGeometryTransform.NetToSourceLength(L: Single): Single;
begin
  if Scale = 0 then
    Exit(L);
  Result := L / Scale;
end;

function TGeometryTransform.SourceToNetX(X: Single): Single;
begin
  Result := X * Scale + PadX;
end;

function TGeometryTransform.SourceToNetY(Y: Single): Single;
begin
  Result := Y * Scale + PadY;
end;

{ TLetterboxPreprocessor }

constructor TLetterboxPreprocessor.Create(AAllowUpscale: Boolean;
  AFillR, AFillG, AFillB: Byte; AAlign: TLetterboxAlign;
  AMean: Single; AStd: Single);
begin
  inherited Create;
  if AStd = 0 then
    raise EPreprocessError.Create('Desvio padrao nao pode ser zero');
  FAllowUpscale := AAllowUpscale;
  FFillR := AFillR;
  FFillG := AFillG;
  FFillB := AFillB;
  FAlign := AAlign;
  FMean := AMean;
  FStd := AStd;
end;

class function TLetterboxPreprocessor.Scrfd: TLetterboxPreprocessor;
begin
  Result := TLetterboxPreprocessor.Create(True, 0, 0, 0, laTopLeft, 127.5, 128.0);
end;

function TLetterboxPreprocessor.Describe: string;
begin
  if FAlign = laTopLeft then
    Result := 'letterbox canto superior esquerdo'
  else
    Result := 'letterbox centralizado';
  Result := Result + Format(' (fill %d,%d,%d, (x-%.1f)/%.1f, RGB NCHW)',
    [FFillR, FFillG, FFillB, FMean, FStd]);
end;

function TLetterboxPreprocessor.PassCount: Integer;
begin
  Result := 1;
end;

function TLetterboxPreprocessor.Prepare(const Image: IImage;
  Pass, NetWidth, NetHeight: Integer;
  out Transform: TGeometryTransform): TPreparedInput;
var
  Scale: Single;
  NewWidth, NewHeight, OffsetX, OffsetY: Integer;
  Resized: IImage;
  PlaneSize, X, Y, DstX, DstY: Integer;
  Src: PByte;
  RIndex, GIndex, BIndex: Integer;
begin
  if Image = nil then
    raise EPreprocessError.Create('Imagem nula');
  if (NetWidth <= 0) or (NetHeight <= 0) then
    raise EPreprocessError.CreateFmt('Tamanho de rede invalido: %dx%d',
      [NetWidth, NetHeight]);

  Scale := Min(NetWidth / Image.Width, NetHeight / Image.Height);
  if (not FAllowUpscale) and (Scale > 1) then
    Scale := 1;

  NewWidth := Max(1, Round(Image.Width * Scale));
  NewHeight := Max(1, Round(Image.Height * Scale));
  NewWidth := Min(NewWidth, NetWidth);
  NewHeight := Min(NewHeight, NetHeight);

  if FAlign = laTopLeft then
  begin
    OffsetX := 0;
    OffsetY := 0;
  end
  else
  begin
    OffsetX := (NetWidth - NewWidth) div 2;
    OffsetY := (NetHeight - NewHeight) div 2;
  end;

  Transform.Scale := Scale;
  Transform.PadX := OffsetX;
  Transform.PadY := OffsetY;
  Transform.SourceWidth := Image.Width;
  Transform.SourceHeight := Image.Height;
  Transform.NetWidth := NetWidth;
  Transform.NetHeight := NetHeight;

  PlaneSize := NetWidth * NetHeight;
  SetLength(Result.Data, PlaneSize * 3);
  Result.Shape := TArray<Int64>.Create(1, 3, NetHeight, NetWidth);

  // Preenchimento de fundo.
  for Y := 0 to NetHeight - 1 do
    for X := 0 to NetWidth - 1 do
    begin
      RIndex := Y * NetWidth + X;
      Result.Data[RIndex] := (FFillR - FMean) / FStd;
      Result.Data[PlaneSize + RIndex] := (FFillG - FMean) / FStd;
      Result.Data[PlaneSize * 2 + RIndex] := (FFillB - FMean) / FStd;
    end;

  Resized := ResampleBilinear(Image, NewWidth, NewHeight);

  for Y := 0 to NewHeight - 1 do
  begin
    Src := Resized.RowPtr(Y);
    DstY := OffsetY + Y;
    for X := 0 to NewWidth - 1 do
    begin
      DstX := OffsetX + X;
      RIndex := DstY * NetWidth + DstX;
      GIndex := PlaneSize + RIndex;
      BIndex := PlaneSize * 2 + RIndex;

      Result.Data[RIndex] := (Src^ - FMean) / FStd;  Inc(Src);
      Result.Data[GIndex] := (Src^ - FMean) / FStd;  Inc(Src);
      Result.Data[BIndex] := (Src^ - FMean) / FStd;  Inc(Src);
    end;
  end;
end;

{ TCropClassifierPreprocessor }

constructor TCropClassifierPreprocessor.Create(const AMean, AStd: TArray<Single>;
  ACropRatio: Single; AMultiCrop: Boolean);
var
  I: Integer;
begin
  inherited Create;

  if (Length(AMean) <> 3) or (Length(AStd) <> 3) then
    raise EPreprocessError.Create('Media e desvio precisam de 3 canais');

  for I := 0 to 2 do
  begin
    FMean[I] := AMean[I];
    if AStd[I] = 0 then
      raise EPreprocessError.Create('Desvio padrao nao pode ser zero');
    FStd[I] := AStd[I];
  end;

  if (ACropRatio <= 0) or (ACropRatio > 1) then
    raise EPreprocessError.Create('CropRatio precisa estar em (0, 1]');

  FCropRatio := ACropRatio;
  FMultiCrop := AMultiCrop;
end;

class function TCropClassifierPreprocessor.ImageNet(
  AMultiCrop: Boolean): TCropClassifierPreprocessor;
begin
  Result := TCropClassifierPreprocessor.Create(
    TArray<Single>.Create(0.485, 0.456, 0.406),
    TArray<Single>.Create(0.229, 0.224, 0.225),
    0.875, AMultiCrop);
end;

class function TCropClassifierPreprocessor.UnitScale(
  AMultiCrop: Boolean): TCropClassifierPreprocessor;
begin
  Result := TCropClassifierPreprocessor.Create(
    TArray<Single>.Create(0, 0, 0),
    TArray<Single>.Create(1, 1, 1),
    1.0, AMultiCrop);
end;

function TCropClassifierPreprocessor.Describe: string;
begin
  Result := Format('resize+crop (ratio %.3f, mean %.3f/%.3f/%.3f, std %.3f/%.3f/%.3f, passadas %d)',
    [FCropRatio, FMean[0], FMean[1], FMean[2], FStd[0], FStd[1], FStd[2], PassCount]);
end;

function TCropClassifierPreprocessor.PassCount: Integer;
begin
  if FMultiCrop then
    Result := 5
  else
    Result := 1;
end;

function TCropClassifierPreprocessor.Prepare(const Image: IImage;
  Pass, NetWidth, NetHeight: Integer;
  out Transform: TGeometryTransform): TPreparedInput;
var
  ShortSide, ResizeWidth, ResizeHeight: Integer;
  Scale: Single;
  Resized: IImage;
  MaxLeft, MaxTop, CropLeft, CropTop: Integer;
  PlaneSize, X, Y, Index: Integer;
  Src: PByte;
  Value: Single;
begin
  if Image = nil then
    raise EPreprocessError.Create('Imagem nula');
  if (NetWidth <= 0) or (NetHeight <= 0) then
    raise EPreprocessError.CreateFmt('Tamanho de rede invalido: %dx%d',
      [NetWidth, NetHeight]);

  // Lado menor vai para Net/CropRatio; depois recorta Net x Net.
  ShortSide := Max(NetWidth, NetHeight);
  ShortSide := Max(ShortSide, Round(ShortSide / FCropRatio));

  Scale := ShortSide / Min(Image.Width, Image.Height);
  ResizeWidth := Max(NetWidth, Round(Image.Width * Scale));
  ResizeHeight := Max(NetHeight, Round(Image.Height * Scale));

  Resized := ResampleBilinear(Image, ResizeWidth, ResizeHeight);

  MaxLeft := ResizeWidth - NetWidth;
  MaxTop := ResizeHeight - NetHeight;

  // 0 = centro, 1..4 = cantos (usados apenas em multi-crop).
  case Pass of
    1: begin CropLeft := 0;       CropTop := 0;      end;
    2: begin CropLeft := MaxLeft; CropTop := 0;      end;
    3: begin CropLeft := 0;       CropTop := MaxTop; end;
    4: begin CropLeft := MaxLeft; CropTop := MaxTop; end;
  else
    CropLeft := MaxLeft div 2;
    CropTop := MaxTop div 2;
  end;

  CropLeft := Min(Max(CropLeft, 0), MaxLeft);
  CropTop := Min(Max(CropTop, 0), MaxTop);

  Transform.Scale := Scale;
  Transform.PadX := -CropLeft;
  Transform.PadY := -CropTop;
  Transform.SourceWidth := Image.Width;
  Transform.SourceHeight := Image.Height;
  Transform.NetWidth := NetWidth;
  Transform.NetHeight := NetHeight;

  PlaneSize := NetWidth * NetHeight;
  SetLength(Result.Data, PlaneSize * 3);
  Result.Shape := TArray<Int64>.Create(1, 3, NetHeight, NetWidth);

  for Y := 0 to NetHeight - 1 do
  begin
    Src := Resized.RowPtr(CropTop + Y) + NativeInt(CropLeft) * 3;
    for X := 0 to NetWidth - 1 do
    begin
      Index := Y * NetWidth + X;

      Value := Src^ / 255; Inc(Src);
      Result.Data[Index] := (Value - FMean[0]) / FStd[0];

      Value := Src^ / 255; Inc(Src);
      Result.Data[PlaneSize + Index] := (Value - FMean[1]) / FStd[1];

      Value := Src^ / 255; Inc(Src);
      Result.Data[PlaneSize * 2 + Index] := (Value - FMean[2]) / FStd[2];
    end;
  end;
end;

end.
