unit Vision.Report.Face;

{
  Apresentacao do modo facial.

  Separado de Vision.Report por Interface Segregation: quem so imprime
  resultado de deteccao nao precisa arrastar metodos de galeria, e vice-versa.
}

interface

uses
  System.SysUtils,
  System.Math,
  Vision.Types,
  Vision.Embedding,
  Vision.Face;

type
  IFaceReporter = interface
    ['{6B0F3D74-2C58-4A19-8E63-1D9A5C07B482}']
    procedure ReportFaces(const FileName: string; const Faces: TDetectedFaces);
    procedure ReportMatches(const Matches: TArray<TGalleryMatch>;
      Threshold: Single);
    procedure ReportComparison(const LeftFile, RightFile: string;
      Similarity, Threshold: Single);
    procedure ReportGallery(const Gallery: IFaceGallery);
    procedure ReportEnrolled(const AName, FileName: string;
      const Face: TDetectedFace; Total: Integer);
  end;

  TFaceConsoleReporter = class(TInterfacedObject, IFaceReporter)
  private
    FVerbose: Boolean;
  public
    constructor Create(AVerbose: Boolean = True);
    procedure ReportFaces(const FileName: string; const Faces: TDetectedFaces);
    procedure ReportMatches(const Matches: TArray<TGalleryMatch>;
      Threshold: Single);
    procedure ReportComparison(const LeftFile, RightFile: string;
      Similarity, Threshold: Single);
    procedure ReportGallery(const Gallery: IFaceGallery);
    procedure ReportEnrolled(const AName, FileName: string;
      const Face: TDetectedFace; Total: Integer);
  end;

implementation

const
  KEYPOINT_NAMES: array[0..4] of string = (
    'olho esq', 'olho dir', 'nariz', 'boca esq', 'boca dir');

constructor TFaceConsoleReporter.Create(AVerbose: Boolean);
begin
  inherited Create;
  FVerbose := AVerbose;
end;

procedure TFaceConsoleReporter.ReportFaces(const FileName: string;
  const Faces: TDetectedFaces);
var
  I, K: Integer;
begin
  Writeln;
  Writeln(Format('--- ROSTOS EM %s ---', [ExtractFileName(FileName)]));

  if Length(Faces) = 0 then
  begin
    Writeln('Nenhum rosto detectado.');
    Exit;
  end;

  for I := 0 to High(Faces) do
  begin
    Writeln(Format('  #%d  score %.2f%%  caixa %s  vetor %dD',
      [I + 1, Faces[I].Detection.Score * 100,
       Faces[I].Detection.Box.ToString,
       Faces[I].Embedding.Dimension]));

    if FVerbose and (Length(Faces[I].Detection.Keypoints) >= 5) then
      for K := 0 to 4 do
        Writeln(Format('       %-9s (%7.1f, %7.1f)',
          [KEYPOINT_NAMES[K],
           Faces[I].Detection.Keypoints[K].X,
           Faces[I].Detection.Keypoints[K].Y]));
  end;
end;

procedure TFaceConsoleReporter.ReportMatches(
  const Matches: TArray<TGalleryMatch>; Threshold: Single);
var
  I: Integer;
  Verdict: string;
begin
  Writeln;
  if Length(Matches) = 0 then
  begin
    Writeln('Galeria vazia: nada com que comparar.');
    Exit;
  end;

  Writeln(Format('Mais parecidos na galeria (limiar %.2f):', [Threshold]));
  for I := 0 to High(Matches) do
  begin
    if Matches[I].IsSamePerson(Threshold) then
      Verdict := 'MESMA PESSOA'
    else
      Verdict := '-';

    Writeln(Format('  #%d  %-24s  cos %7.4f  %s',
      [I + 1, Matches[I].Name, Matches[I].Similarity, Verdict]));

    if FVerbose and (Matches[I].Source <> '') then
      Writeln(Format('       cadastrado de: %s', [Matches[I].Source]));
  end;

  Writeln;
  if Matches[0].IsSamePerson(Threshold) then
    Writeln(Format('Veredito: %s', [Matches[0].Name]))
  else
    Writeln('Veredito: desconhecido (nenhum cadastro acima do limiar)');
end;

procedure TFaceConsoleReporter.ReportComparison(const LeftFile,
  RightFile: string; Similarity, Threshold: Single);
begin
  Writeln;
  Writeln('--- COMPARACAO ---');
  Writeln('  A: ', ExtractFileName(LeftFile));
  Writeln('  B: ', ExtractFileName(RightFile));
  Writeln(Format('  similaridade de cosseno: %.4f  (limiar %.2f)',
    [Similarity, Threshold]));
  Writeln;
  if Similarity >= Threshold then
    Writeln('  Veredito: MESMA PESSOA')
  else
    Writeln('  Veredito: pessoas diferentes');
end;

procedure TFaceConsoleReporter.ReportGallery(const Gallery: IFaceGallery);
var
  I: Integer;
  Item: TFaceRecord;
begin
  Writeln;
  Writeln(Format('--- GALERIA: %d cadastro(s) ---', [Gallery.Count]));

  if Gallery.Count = 0 then
  begin
    Writeln('Vazia. Use --enroll=NOME imagem.jpg para cadastrar.');
    Exit;
  end;

  for I := 0 to Gallery.Count - 1 do
  begin
    Item := Gallery.Item(I);
    Writeln(Format('  %-24s  %dD  %s',
      [Item.Name, Item.Embedding.Dimension, Item.Source]));
  end;
end;

procedure TFaceConsoleReporter.ReportEnrolled(const AName, FileName: string;
  const Face: TDetectedFace; Total: Integer);
begin
  Writeln;
  Writeln(Format('Cadastrado: %s', [AName]));
  Writeln(Format('  origem   : %s', [ExtractFileName(FileName)]));
  Writeln(Format('  rosto    : score %.2f%%  caixa %s',
    [Face.Detection.Score * 100, Face.Detection.Box.ToString]));
  Writeln(Format('  vetor    : %d dimensoes', [Face.Embedding.Dimension]));
  Writeln(Format('  galeria  : %d cadastro(s)', [Total]));
end;

end.
