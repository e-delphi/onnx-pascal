unit Vision.Image;

{
  Buffer de imagem RGB independente da VCL, mais um carregador baseado na VCL.

  O restante do pipeline conversa apenas com IImage. Trocar o carregador
  (GDI+, FFmpeg, camera) e adicionar uma classe nova, nao mexer no pipeline.

  Layout do buffer: RGB entrelacado, 3 bytes por pixel, linha 0 no topo,
  sem padding de linha. TBitmap da VCL usa BGR e a conversao acontece
  apenas nas fronteiras (carregar / desenhar).
}

interface

uses
  System.SysUtils,
  System.Classes,
  System.Types,
  System.Math,
  Vcl.Graphics;

type
  EImageError = class(Exception);

  IImage = interface
    ['{4D6E1A72-9C33-4F58-9D21-3B7E5A0C8E14}']
    function Width: Integer;
    function Height: Integer;
    function RowPtr(Y: Integer): PByte;
    function Data: PByte;
  end;

  TRGBImage = class(TInterfacedObject, IImage)
  private
    FWidth: Integer;
    FHeight: Integer;
    FData: TBytes;
  public
    constructor Create(AWidth, AHeight: Integer);
    function Width: Integer;
    function Height: Integer;
    function RowPtr(Y: Integer): PByte;
    function Data: PByte;
    procedure Fill(R, G, B: Byte);
  end;

  IImageLoader = interface
    ['{8F2C5B90-6A41-4E77-B0D5-1C9F4E2A6B35}']
    function Load(const FileName: string): IImage;
  end;

  { Usa TPicture, portanto aceita BMP, JPG, PNG e GIF. Imagens com canal
    alfa sao compostas sobre fundo branco. }
  TVclImageLoader = class(TInterfacedObject, IImageLoader)
  public
    function Load(const FileName: string): IImage;
  end;

function CreateImage(AWidth, AHeight: Integer): TRGBImage;

{ Reamostragem bilinear. A VCL usa StretchDraw sem interpolacao, o que
  degrada bastante a acuracia do modelo em imagens grandes. }
function ResampleBilinear(const Src: IImage; DstWidth, DstHeight: Integer): IImage;

function ImageToBitmap(const Src: IImage): TBitmap;
function BitmapToImage(Bmp: TBitmap): IImage;
procedure SaveImageToFile(const Src: IImage; const FileName: string);

implementation

uses
  Vcl.Imaging.Jpeg,
  Vcl.Imaging.PngImage,
  Vcl.Imaging.GIFImg;

function CreateImage(AWidth, AHeight: Integer): TRGBImage;
begin
  Result := TRGBImage.Create(AWidth, AHeight);
end;

{ TRGBImage }

constructor TRGBImage.Create(AWidth, AHeight: Integer);
begin
  inherited Create;
  if (AWidth <= 0) or (AHeight <= 0) then
    raise EImageError.CreateFmt('Dimensoes invalidas: %dx%d', [AWidth, AHeight]);
  FWidth := AWidth;
  FHeight := AHeight;
  SetLength(FData, FWidth * FHeight * 3);
end;

function TRGBImage.Width: Integer;
begin
  Result := FWidth;
end;

function TRGBImage.Height: Integer;
begin
  Result := FHeight;
end;

function TRGBImage.Data: PByte;
begin
  Result := PByte(FData);
end;

function TRGBImage.RowPtr(Y: Integer): PByte;
begin
  if (Y < 0) or (Y >= FHeight) then
    raise EImageError.CreateFmt('Linha fora da imagem: %d', [Y]);
  Result := PByte(FData) + NativeInt(Y) * FWidth * 3;
end;

procedure TRGBImage.Fill(R, G, B: Byte);
var
  I, N: Integer;
  P: PByte;
begin
  N := FWidth * FHeight;
  P := PByte(FData);
  for I := 0 to N - 1 do
  begin
    P^ := R; Inc(P);
    P^ := G; Inc(P);
    P^ := B; Inc(P);
  end;
end;

{ TVclImageLoader }

function TVclImageLoader.Load(const FileName: string): IImage;
var
  Picture: TPicture;
  Bmp: TBitmap;
begin
  if not FileExists(FileName) then
    raise EImageError.CreateFmt('Imagem nao encontrada: %s', [FileName]);

  Picture := TPicture.Create;
  try
    try
      Picture.LoadFromFile(FileName);
    except
      on E: Exception do
        raise EImageError.CreateFmt('Nao foi possivel abrir "%s": %s',
          [FileName, E.Message]);
    end;

    if (Picture.Width <= 0) or (Picture.Height <= 0) then
      raise EImageError.CreateFmt('Imagem vazia: %s', [FileName]);

    Bmp := TBitmap.Create;
    try
      Bmp.PixelFormat := pf24bit;
      Bmp.SetSize(Picture.Width, Picture.Height);
      // Fundo branco: PNG/GIF com transparencia nao ficam com lixo.
      Bmp.Canvas.Brush.Color := clWhite;
      Bmp.Canvas.FillRect(Rect(0, 0, Bmp.Width, Bmp.Height));
      Bmp.Canvas.Draw(0, 0, Picture.Graphic);
      Result := BitmapToImage(Bmp);
    finally
      Bmp.Free;
    end;
  finally
    Picture.Free;
  end;
end;

{ Conversoes VCL }

function BitmapToImage(Bmp: TBitmap): IImage;
var
  Img: TRGBImage;
  Y, X: Integer;
  Src, Dst: PByte;
begin
  if Bmp = nil then
    raise EImageError.Create('Bitmap nulo');

  if Bmp.PixelFormat <> pf24bit then
    Bmp.PixelFormat := pf24bit;

  Img := TRGBImage.Create(Bmp.Width, Bmp.Height);
  Result := Img;

  for Y := 0 to Bmp.Height - 1 do
  begin
    Src := PByte(Bmp.ScanLine[Y]);
    Dst := Img.RowPtr(Y);
    for X := 0 to Bmp.Width - 1 do
    begin
      // ScanLine de pf24bit vem em BGR.
      Dst^ := PByte(Src + 2)^; Inc(Dst);  // R
      Dst^ := PByte(Src + 1)^; Inc(Dst);  // G
      Dst^ := Src^;            Inc(Dst);  // B
      Inc(Src, 3);
    end;
  end;
end;

function ImageToBitmap(const Src: IImage): TBitmap;
var
  Y, X: Integer;
  S, D: PByte;
begin
  if Src = nil then
    raise EImageError.Create('Imagem nula');

  Result := TBitmap.Create;
  try
    Result.PixelFormat := pf24bit;
    Result.SetSize(Src.Width, Src.Height);

    for Y := 0 to Src.Height - 1 do
    begin
      S := Src.RowPtr(Y);
      D := PByte(Result.ScanLine[Y]);
      for X := 0 to Src.Width - 1 do
      begin
        PByte(D + 2)^ := S^; Inc(S);  // R
        PByte(D + 1)^ := S^; Inc(S);  // G
        D^            := S^; Inc(S);  // B
        Inc(D, 3);
      end;
    end;
  except
    Result.Free;
    raise;
  end;
end;

procedure SaveImageToFile(const Src: IImage; const FileName: string);
var
  Bmp: TBitmap;
  Png: TPngImage;
  Jpg: TJPEGImage;
  Ext, Folder: string;
begin
  // A pasta de saida e criada aqui para que nenhum chamador precise lembrar.
  Folder := ExtractFilePath(FileName);
  if (Folder <> '') and (not DirectoryExists(Folder)) then
    if not ForceDirectories(Folder) then
      raise EImageError.CreateFmt('Nao foi possivel criar a pasta %s', [Folder]);

  Bmp := ImageToBitmap(Src);
  try
    Ext := LowerCase(ExtractFileExt(FileName));
    if (Ext = '.jpg') or (Ext = '.jpeg') then
    begin
      Jpg := TJPEGImage.Create;
      try
        Jpg.CompressionQuality := 92;
        Jpg.Assign(Bmp);
        Jpg.SaveToFile(FileName);
      finally
        Jpg.Free;
      end;
    end
    else if Ext = '.bmp' then
      Bmp.SaveToFile(FileName)
    else
    begin
      Png := TPngImage.Create;
      try
        Png.Assign(Bmp);
        Png.SaveToFile(FileName);
      finally
        Png.Free;
      end;
    end;
  finally
    Bmp.Free;
  end;
end;

{ Reamostragem }

function ResampleBilinear(const Src: IImage; DstWidth, DstHeight: Integer): IImage;
var
  Dst: TRGBImage;
  X, Y, C: Integer;
  SX, SY, ScaleX, ScaleY: Single;
  X0, Y0, X1, Y1: Integer;
  FX, FY, W00, W10, W01, W11: Single;
  R0, R1: PByte;
  P00, P10, P01, P11: PByte;
  Out_: PByte;
  Value: Single;
begin
  if Src = nil then
    raise EImageError.Create('Imagem nula');
  if (DstWidth <= 0) or (DstHeight <= 0) then
    raise EImageError.CreateFmt('Destino invalido: %dx%d', [DstWidth, DstHeight]);

  Dst := TRGBImage.Create(DstWidth, DstHeight);
  Result := Dst;

  if (Src.Width = DstWidth) and (Src.Height = DstHeight) then
  begin
    Move(Src.Data^, Dst.Data^, NativeInt(DstWidth) * DstHeight * 3);
    Exit;
  end;

  ScaleX := Src.Width / DstWidth;
  ScaleY := Src.Height / DstHeight;

  for Y := 0 to DstHeight - 1 do
  begin
    // Alinhamento pelo centro do pixel (mesma convencao do cv2.resize).
    SY := (Y + 0.5) * ScaleY - 0.5;
    if SY < 0 then
      SY := 0;
    Y0 := Floor(SY);
    if Y0 > Src.Height - 1 then
      Y0 := Src.Height - 1;
    Y1 := Min(Y0 + 1, Src.Height - 1);
    FY := SY - Y0;

    R0 := Src.RowPtr(Y0);
    R1 := Src.RowPtr(Y1);
    Out_ := Dst.RowPtr(Y);

    for X := 0 to DstWidth - 1 do
    begin
      SX := (X + 0.5) * ScaleX - 0.5;
      if SX < 0 then
        SX := 0;
      X0 := Floor(SX);
      if X0 > Src.Width - 1 then
        X0 := Src.Width - 1;
      X1 := Min(X0 + 1, Src.Width - 1);
      FX := SX - X0;

      W00 := (1 - FX) * (1 - FY);
      W10 := FX * (1 - FY);
      W01 := (1 - FX) * FY;
      W11 := FX * FY;

      P00 := R0 + X0 * 3;
      P10 := R0 + X1 * 3;
      P01 := R1 + X0 * 3;
      P11 := R1 + X1 * 3;

      for C := 0 to 2 do
      begin
        Value := PByte(P00 + C)^ * W00 + PByte(P10 + C)^ * W10 +
                 PByte(P01 + C)^ * W01 + PByte(P11 + C)^ * W11;
        PByte(Out_ + C)^ := Byte(Round(Min(255.0, Max(0.0, Value))));
      end;

      Inc(Out_, 3);
    end;
  end;
end;

end.
