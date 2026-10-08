unit Vision.Report.Ocr;

{
  Saida textual do modo OCR (IOcrReporter), separada de IResultReporter
  pelo mesmo motivo que IFaceReporter: o resultado tem outra forma.

  O texto lido tem acentos e ate ideogramas: o programa principal poe o
  console em UTF-8 antes de qualquer saida.
}

interface

uses
  System.SysUtils,
  System.Classes,
  System.IOUtils,
  ONNX.Types,
  Vision.Ocr;

type
  IOcrReporter = interface
    ['{5E9C3A84-7B21-4D6F-B0E8-2A4F6C1D9B73}']
    procedure ReportEngine(const Engine: IOcrEngine);
    procedure ReportResult(const Value: TOcrResult);
    procedure SaveText(const Value: TOcrResult; const FileName: string);
  end;

  TOcrConsoleReporter = class(TInterfacedObject, IOcrReporter)
  private
    FVerbose: Boolean;
  public
    constructor Create(AVerbose: Boolean = True);
    procedure ReportEngine(const Engine: IOcrEngine);
    procedure ReportResult(const Value: TOcrResult);
    procedure SaveText(const Value: TOcrResult; const FileName: string);
  end;

implementation

constructor TOcrConsoleReporter.Create(AVerbose: Boolean);
begin
  inherited Create;
  FVerbose := AVerbose;
end;

procedure TOcrConsoleReporter.ReportEngine(const Engine: IOcrEngine);
begin
  Writeln;
  Writeln('--- MODELOS ---');
  Writeln('Detector     : ', Engine.DetectorDescription);
  Writeln('Reconhecedor : ', Engine.RecognizerDescription);
end;

procedure TOcrConsoleReporter.ReportResult(const Value: TOcrResult);
var
  I: Integer;
  Line: TOcrLine;
begin
  Writeln;
  Writeln('--- RESULTADO ---');
  Writeln(Format('Imagem       : %dx%d', [Value.ImageWidth, Value.ImageHeight]));
  Writeln(Format('Tempos       : deteccao %.1f ms | leitura %.1f ms (%d linhas) | total %.1f ms',
    [Value.DetectMs, Value.RecognizeMs, Length(Value.Lines),
     Value.DetectMs + Value.RecognizeMs]));
  Writeln;

  if Length(Value.Lines) = 0 then
  begin
    Writeln('Nenhum texto encontrado.');
    Exit;
  end;

  Writeln(Format('%d linha(s) lida(s), em ordem de leitura:', [Length(Value.Lines)]));
  for I := 0 to High(Value.Lines) do
  begin
    Line := Value.Lines[I];
    if FVerbose then
      Writeln(Format('  #%-3d det %5.1f%%  rec %5.1f%%  %s',
        [I + 1, Line.Detection.Score * 100, Line.TextScore * 100, Line.Text]))
    else
      Writeln('  ', Line.Text);
  end;
end;

procedure TOcrConsoleReporter.SaveText(const Value: TOcrResult;
  const FileName: string);
var
  Encoding: TEncoding;
begin
  if FileName = '' then
    Exit;
  ForceDirectories(ExtractFilePath(FileName));
  // UTF-8 sem BOM: e o que editores e ferramentas de linha de comando esperam.
  Encoding := TUTF8Encoding.Create(False);
  try
    TFile.WriteAllText(FileName, Value.FullText + sLineBreak, Encoding);
  finally
    Encoding.Free;
  end;
end;

end.
