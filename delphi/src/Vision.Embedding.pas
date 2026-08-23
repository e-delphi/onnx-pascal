unit Vision.Embedding;

{
  Vetores de identidade e a galeria que os guarda.

  Um embedding e um vetor L2-normalizado; a semelhanca entre dois rostos e o
  cosseno entre eles, que para vetores unitarios e o proprio produto escalar.
  Com buffalo_l / w600k_r50, mesma pessoa fica em torno de 0,78 e pessoas
  diferentes perto de 0,00; o limiar padrao e 0,40.

  A galeria e um arquivo de texto: uma linha por rosto, campos separados por
  TAB. Fica grande (cerca de 4 KB por rosto) mas e inspecionavel, diffavel e
  independente de plataforma, o que importa mais que bytes num cadastro local
  de algumas centenas de faces.
}

interface

uses
  System.SysUtils,
  System.Classes,
  System.Math,
  System.Generics.Collections,
  System.Generics.Defaults;

const
  { Limiar de cosseno acima do qual dois vetores sao tratados como a mesma
    pessoa. Valido para buffalo_l / w600k_r50. }
  DEFAULT_FACE_THRESHOLD = 0.40;

  GALLERY_SIGNATURE = '#ONNXDemo-face-gallery';
  GALLERY_VERSION   = 1;

type
  EGalleryError = class(Exception);

  TEmbedding = record
    Vector: TArray<Single>;
    class function FromArray(const AVector: TArray<Single>): TEmbedding; static;
    class function FromText(const Text: string): TEmbedding; static;
    function Dimension: Integer;
    function IsValid: Boolean;
    function Norm: Single;
    function Normalize: TEmbedding;
    { Cosseno. Normaliza os dois lados por seguranca se algum nao estiver. }
    function Similarity(const Other: TEmbedding): Single;
    function ToText: string;
  end;

  TFaceRecord = record
    Name: string;
    Source: string;
    Embedding: TEmbedding;
  end;

  TGalleryMatch = record
    Name: string;
    Source: string;
    Similarity: Single;
    RecordIndex: Integer;
    function IsSamePerson(Threshold: Single): Boolean;
  end;

  IFaceGallery = interface
    ['{2E9A7C41-8B36-4F52-A0D7-6C1B4E93F805}']
    function Count: Integer;
    function Item(Index: Integer): TFaceRecord;
    function Names: TArray<string>;
    procedure Enroll(const AName, ASource: string; const AEmbedding: TEmbedding);
    function RemoveByName(const AName: string): Integer;
    function Query(const AEmbedding: TEmbedding; TopK: Integer): TArray<TGalleryMatch>;
    procedure LoadFromFile(const FileName: string);
    procedure SaveToFile(const FileName: string);
  end;

  TFaceGallery = class(TInterfacedObject, IFaceGallery)
  private
    FRecords: TList<TFaceRecord>;
    FDimension: Integer;
  public
    constructor Create;
    destructor Destroy; override;

    function Count: Integer;
    function Item(Index: Integer): TFaceRecord;
    function Names: TArray<string>;
    procedure Enroll(const AName, ASource: string; const AEmbedding: TEmbedding);
    function RemoveByName(const AName: string): Integer;
    function Query(const AEmbedding: TEmbedding; TopK: Integer): TArray<TGalleryMatch>;
    procedure LoadFromFile(const FileName: string);
    procedure SaveToFile(const FileName: string);
  end;

implementation

const
  TAB = #9;

function InvariantSettings: TFormatSettings;
begin
  Result := TFormatSettings.Invariant;
end;

{ TEmbedding }

class function TEmbedding.FromArray(const AVector: TArray<Single>): TEmbedding;
begin
  Result.Vector := Copy(AVector);
end;

class function TEmbedding.FromText(const Text: string): TEmbedding;
var
  Parts: TArray<string>;
  I, N: Integer;
  Value: Double;
begin
  Result.Vector := nil;
  Parts := Text.Split([' '], TStringSplitOptions.ExcludeEmpty);
  SetLength(Result.Vector, Length(Parts));

  N := 0;
  for I := 0 to High(Parts) do
    if TryStrToFloat(Parts[I], Value, InvariantSettings) then
    begin
      Result.Vector[N] := Value;
      Inc(N);
    end;

  SetLength(Result.Vector, N);
end;

function TEmbedding.Dimension: Integer;
begin
  Result := Length(Vector);
end;

function TEmbedding.IsValid: Boolean;
begin
  Result := (Length(Vector) > 0) and (Norm > 0);
end;

function TEmbedding.Norm: Single;
var
  I: Integer;
  Sum: Double;
begin
  Sum := 0;
  for I := 0 to High(Vector) do
    Sum := Sum + Double(Vector[I]) * Vector[I];
  Result := Sqrt(Sum);
end;

function TEmbedding.Normalize: TEmbedding;
var
  I: Integer;
  N: Single;
begin
  N := Norm;
  if N <= 0 then
    Exit(Self);

  SetLength(Result.Vector, Length(Vector));
  for I := 0 to High(Vector) do
    Result.Vector[I] := Vector[I] / N;
end;

function TEmbedding.Similarity(const Other: TEmbedding): Single;
var
  A, B: TEmbedding;
  I: Integer;
  Sum: Double;
begin
  if (Length(Vector) = 0) or (Length(Vector) <> Length(Other.Vector)) then
    Exit(0);

  A := Self;
  if Abs(A.Norm - 1) > 1E-3 then
    A := A.Normalize;

  B := Other;
  if Abs(B.Norm - 1) > 1E-3 then
    B := B.Normalize;

  Sum := 0;
  for I := 0 to High(A.Vector) do
    Sum := Sum + Double(A.Vector[I]) * B.Vector[I];

  Result := Min(1, Max(-1, Sum));
end;

function TEmbedding.ToText: string;
var
  Builder: TStringBuilder;
  I: Integer;
begin
  Builder := TStringBuilder.Create;
  try
    for I := 0 to High(Vector) do
    begin
      if I > 0 then
        Builder.Append(' ');
      Builder.Append(FormatFloat('0.######', Vector[I], InvariantSettings));
    end;
    Result := Builder.ToString;
  finally
    Builder.Free;
  end;
end;

{ TGalleryMatch }

function TGalleryMatch.IsSamePerson(Threshold: Single): Boolean;
begin
  Result := Similarity >= Threshold;
end;

{ TFaceGallery }

constructor TFaceGallery.Create;
begin
  inherited Create;
  FRecords := TList<TFaceRecord>.Create;
  FDimension := 0;
end;

destructor TFaceGallery.Destroy;
begin
  FRecords.Free;
  inherited;
end;

function TFaceGallery.Count: Integer;
begin
  Result := FRecords.Count;
end;

function TFaceGallery.Item(Index: Integer): TFaceRecord;
begin
  if (Index < 0) or (Index >= FRecords.Count) then
    raise EGalleryError.CreateFmt('Indice invalido na galeria: %d', [Index]);
  Result := FRecords[Index];
end;

function TFaceGallery.Names: TArray<string>;
var
  Seen: TStringList;
  I: Integer;
begin
  Seen := TStringList.Create;
  try
    Seen.Sorted := True;
    Seen.Duplicates := dupIgnore;
    for I := 0 to FRecords.Count - 1 do
      Seen.Add(FRecords[I].Name);
    Result := Seen.ToStringArray;
  finally
    Seen.Free;
  end;
end;

procedure TFaceGallery.Enroll(const AName, ASource: string;
  const AEmbedding: TEmbedding);
var
  Entry: TFaceRecord;
begin
  if Trim(AName) = '' then
    raise EGalleryError.Create('O nome do cadastro nao pode ser vazio');

  if not AEmbedding.IsValid then
    raise EGalleryError.Create('Embedding invalido (vazio ou norma zero)');

  if (FDimension > 0) and (AEmbedding.Dimension <> FDimension) then
    raise EGalleryError.CreateFmt(
      'Esta galeria guarda vetores de %d dimensoes e o recebido tem %d. ' +
      'Vetores de modelos de reconhecimento diferentes nao sao comparaveis.',
      [FDimension, AEmbedding.Dimension]);

  Entry.Name := StringReplace(Trim(AName), TAB, ' ', [rfReplaceAll]);
  Entry.Source := StringReplace(ASource, TAB, ' ', [rfReplaceAll]);
  Entry.Embedding := AEmbedding.Normalize;

  FRecords.Add(Entry);
  FDimension := Entry.Embedding.Dimension;
end;

function TFaceGallery.RemoveByName(const AName: string): Integer;
var
  I: Integer;
begin
  Result := 0;
  for I := FRecords.Count - 1 downto 0 do
    if SameText(FRecords[I].Name, Trim(AName)) then
    begin
      FRecords.Delete(I);
      Inc(Result);
    end;

  if FRecords.Count = 0 then
    FDimension := 0;
end;

function TFaceGallery.Query(const AEmbedding: TEmbedding;
  TopK: Integer): TArray<TGalleryMatch>;
var
  Matches: TList<TGalleryMatch>;
  Match: TGalleryMatch;
  Probe: TEmbedding;
  I: Integer;
begin
  Result := nil;
  if (FRecords.Count = 0) or (not AEmbedding.IsValid) then
    Exit;

  Probe := AEmbedding.Normalize;

  Matches := TList<TGalleryMatch>.Create;
  try
    for I := 0 to FRecords.Count - 1 do
    begin
      Match.Name := FRecords[I].Name;
      Match.Source := FRecords[I].Source;
      Match.RecordIndex := I;
      Match.Similarity := Probe.Similarity(FRecords[I].Embedding);
      Matches.Add(Match);
    end;

    Matches.Sort(TComparer<TGalleryMatch>.Construct(
      function(const L, R: TGalleryMatch): Integer
      begin
        if L.Similarity > R.Similarity then
          Result := -1
        else if L.Similarity < R.Similarity then
          Result := 1
        else
          Result := 0;
      end));

    while (TopK > 0) and (Matches.Count > TopK) do
      Matches.Delete(Matches.Count - 1);

    Result := Matches.ToArray;
  finally
    Matches.Free;
  end;
end;

procedure TFaceGallery.LoadFromFile(const FileName: string);
var
  Lines: TStringList;
  Parts: TArray<string>;
  Entry: TFaceRecord;
  I: Integer;
begin
  FRecords.Clear;
  FDimension := 0;

  if not FileExists(FileName) then
    Exit;

  Lines := TStringList.Create;
  try
    Lines.LoadFromFile(FileName, TEncoding.UTF8);

    for I := 0 to Lines.Count - 1 do
    begin
      if (Trim(Lines[I]) = '') or Lines[I].StartsWith('#') then
        Continue;

      Parts := Lines[I].Split([TAB]);
      if Length(Parts) < 3 then
        raise EGalleryError.CreateFmt(
          '%s linha %d: esperados 3 campos separados por TAB, encontrados %d',
          [ExtractFileName(FileName), I + 1, Length(Parts)]);

      Entry.Name := Parts[0];
      Entry.Source := Parts[1];
      Entry.Embedding := TEmbedding.FromText(Parts[2]);

      if not Entry.Embedding.IsValid then
        raise EGalleryError.CreateFmt(
          '%s linha %d: vetor invalido para "%s"',
          [ExtractFileName(FileName), I + 1, Entry.Name]);

      if FDimension = 0 then
        FDimension := Entry.Embedding.Dimension
      else if Entry.Embedding.Dimension <> FDimension then
        raise EGalleryError.CreateFmt(
          '%s linha %d: vetor de %d dimensoes numa galeria de %d',
          [ExtractFileName(FileName), I + 1,
           Entry.Embedding.Dimension, FDimension]);

      FRecords.Add(Entry);
    end;
  finally
    Lines.Free;
  end;
end;

procedure TFaceGallery.SaveToFile(const FileName: string);
var
  Lines: TStringList;
  I: Integer;
  Folder: string;
begin
  Lines := TStringList.Create;
  try
    Lines.Add(Format('%s v%d dim=%d faces=%d',
      [GALLERY_SIGNATURE, GALLERY_VERSION, FDimension, FRecords.Count]));
    Lines.Add('# campos por linha, separados por TAB: nome, origem, vetor');

    for I := 0 to FRecords.Count - 1 do
      Lines.Add(FRecords[I].Name + TAB + FRecords[I].Source + TAB +
                FRecords[I].Embedding.ToText);

    Folder := ExtractFilePath(FileName);
    if (Folder <> '') and (not DirectoryExists(Folder)) then
      ForceDirectories(Folder);

    Lines.SaveToFile(FileName, TEncoding.UTF8);
  finally
    Lines.Free;
  end;
end;

end.
