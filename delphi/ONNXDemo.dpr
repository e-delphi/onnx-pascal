// Eduardo - 22/08/2026
// Refatorado para arquitetura em camadas com suporte a multiplos modelos.
program ONNXDemo;

{$APPTYPE CONSOLE}

{$R *.res}

uses
  Winapi.Windows,
  System.SysUtils,
  System.IOUtils,
  ONNX.CApi in 'src\ONNX.CApi.pas',
  ONNX.Types in 'src\ONNX.Types.pas',
  ONNX.Runtime in 'src\ONNX.Runtime.pas',
  ONNX.Session in 'src\ONNX.Session.pas',
  Vision.Types in 'src\Vision.Types.pas',
  Vision.Image in 'src\Vision.Image.pas',
  Vision.Preprocess in 'src\Vision.Preprocess.pas',
  Vision.Nms in 'src\Vision.Nms.pas',
  Vision.Model in 'src\Vision.Model.pas',
  Vision.Decoder in 'src\Vision.Decoder.pas',
  // Cada unit de decoder se registra sozinha no initialization.
  // Incluir a unit aqui e tudo o que e preciso para habilitar a tarefa.
  Vision.Decoder.Classify in 'src\Vision.Decoder.Classify.pas',
  Vision.Decoder.Detect in 'src\Vision.Decoder.Detect.pas',
  Vision.Decoder.Segment in 'src\Vision.Decoder.Segment.pas',
  Vision.Decoder.Pose in 'src\Vision.Decoder.Pose.pas',
  Vision.Decoder.Obb in 'src\Vision.Decoder.Obb.pas',
  Vision.Decoder.Scrfd in 'src\Vision.Decoder.Scrfd.pas',
  Vision.Decoder.Embed in 'src\Vision.Decoder.Embed.pas',
  Vision.Decoder.Text in 'src\Vision.Decoder.Text.pas',
  Vision.Decoder.Ctc in 'src\Vision.Decoder.Ctc.pas',
  Vision.Predictor in 'src\Vision.Predictor.pas',
  Vision.Embedding in 'src\Vision.Embedding.pas',
  Vision.Face.Align in 'src\Vision.Face.Align.pas',
  Vision.Face in 'src\Vision.Face.pas',
  Vision.Ocr.Crop in 'src\Vision.Ocr.Crop.pas',
  Vision.Ocr in 'src\Vision.Ocr.pas',
  Vision.Render in 'src\Vision.Render.pas',
  Vision.Report in 'src\Vision.Report.pas',
  Vision.Report.Face in 'src\Vision.Report.Face.pas',
  Vision.Render.Ocr in 'src\Vision.Render.Ocr.pas',
  Vision.Report.Ocr in 'src\Vision.Report.Ocr.pas',
  App.Options in 'src\App.Options.pas';

{ ---------------------------------------------------------------- inferencia }

procedure RunPredict(const Options: TAppOptions; const Runtime: IONNXRuntime);
var
  Predictor: IVisionPredictor;
  Reporter: IResultReporter;
  Renderer: IResultRenderer;
  Loader: IImageLoader;
  Image, Annotated: IImage;
  Prediction: TVisionResult;
begin
  if not FileExists(Options.ImagePath) then
    raise Exception.CreateFmt('Imagem nao encontrada: %s', [Options.ImagePath]);

  Reporter := TConsoleReporter.Create(Options.Verbose);
  Reporter.ReportRuntime(Runtime);

  Predictor := TVisionPredictorFactory.Build(
    Runtime, Options.ModelPath, Options.Predictor, Options.Session);
  Reporter.ReportModel(Predictor);

  Loader := TVclImageLoader.Create;
  Image := Loader.Load(Options.ImagePath);

  Prediction := Predictor.Predict(Image);
  Reporter.ReportResult(Prediction);

  if Options.Render and (Options.OutputPath <> '') and
     (Prediction.Task <> vtClassify) and (Prediction.Task <> vtEmbed) and
     (Prediction.Task <> vtTextRec) then
  begin
    Renderer := TResultRenderer.Create;
    Annotated := Renderer.Render(Image, Prediction);
    SaveImageToFile(Annotated, Options.OutputPath);
    Writeln;
    Writeln('Imagem anotada: ', Options.OutputPath);
  end;
end;

{ --------------------------------------------------------------------- rostos }

function BuildEncoder(const Options: TAppOptions;
  const Runtime: IONNXRuntime): IFaceEncoder;
begin
  if not FileExists(Options.FaceDetectPath) then
    raise Exception.CreateFmt(
      'Detector de rostos nao encontrado: %s'#13#10 +
      'Extraia buffalo_l.zip da InsightFace e coloque det_10g.onnx ali, ' +
      'ou informe --face-detect=ARQ.', [Options.FaceDetectPath]);

  if not FileExists(Options.FaceEmbedPath) then
    raise Exception.CreateFmt(
      'Modelo de embedding nao encontrado: %s'#13#10 +
      'Coloque w600k_r50.onnx ali, ou informe --face-embed=ARQ.',
      [Options.FaceEmbedPath]);

  Result := TFaceEncoderFactory.Build(Runtime, Options.FaceDetectPath,
    Options.FaceEmbedPath, Options.Face, Options.Session);
end;

procedure SaveAlignedCrops(const Options: TAppOptions;
  const Faces: TDetectedFaces);
var
  I: Integer;
  Target: string;
begin
  if (Options.AlignedDir = '') or (Length(Faces) = 0) then
    Exit;

  if not DirectoryExists(Options.AlignedDir) then
    ForceDirectories(Options.AlignedDir);

  for I := 0 to High(Faces) do
    if Faces[I].Aligned <> nil then
    begin
      Target := TPath.Combine(Options.AlignedDir, Format('face_%.2d.png', [I]));
      SaveImageToFile(Faces[I].Aligned, Target);
      Writeln('  recorte alinhado: ', Target);
    end;
end;

procedure RunFaceEnroll(const Options: TAppOptions; const Runtime: IONNXRuntime);
var
  Encoder: IFaceEncoder;
  Gallery: IFaceGallery;
  Reporter: IFaceReporter;
  Face: TDetectedFace;
begin
  Reporter := TFaceConsoleReporter.Create(Options.Verbose);
  Encoder := BuildEncoder(Options, Runtime);

  Face := Encoder.EncodeSingle(Options.ImagePath);

  Gallery := TFaceGallery.Create;
  Gallery.LoadFromFile(Options.GalleryPath);
  Gallery.Enroll(Options.EnrollName, ExtractFileName(Options.ImagePath),
    Face.Embedding);
  Gallery.SaveToFile(Options.GalleryPath);

  Reporter.ReportEnrolled(Options.EnrollName, Options.ImagePath, Face,
    Gallery.Count);
  Writeln('  arquivo  : ', Options.GalleryPath);
end;

procedure RunFaceQuery(const Options: TAppOptions; const Runtime: IONNXRuntime);
var
  Encoder: IFaceEncoder;
  Gallery: IFaceGallery;
  Reporter: IFaceReporter;
  Faces: TDetectedFaces;
  I: Integer;
begin
  Reporter := TFaceConsoleReporter.Create(Options.Verbose);
  Encoder := BuildEncoder(Options, Runtime);

  Faces := Encoder.EncodeFile(Options.ImagePath);
  Reporter.ReportFaces(Options.ImagePath, Faces);
  SaveAlignedCrops(Options, Faces);

  Gallery := TFaceGallery.Create;
  Gallery.LoadFromFile(Options.GalleryPath);

  for I := 0 to High(Faces) do
  begin
    if Length(Faces) > 1 then
    begin
      Writeln;
      Writeln(Format('=== rosto #%d ===', [I + 1]));
    end;
    Reporter.ReportMatches(
      Gallery.Query(Faces[I].Embedding, Options.Predictor.TopK),
      Options.FaceThreshold);
  end;
end;

procedure RunFaceCompare(const Options: TAppOptions; const Runtime: IONNXRuntime);
var
  Encoder: IFaceEncoder;
  Reporter: IFaceReporter;
  Left, Right: TDetectedFace;
begin
  Reporter := TFaceConsoleReporter.Create(Options.Verbose);
  Encoder := BuildEncoder(Options, Runtime);

  Left := Encoder.EncodeSingle(Options.ImagePath);
  Right := Encoder.EncodeSingle(Options.SecondImagePath);

  Reporter.ReportComparison(Options.ImagePath, Options.SecondImagePath,
    Left.Embedding.Similarity(Right.Embedding), Options.FaceThreshold);
end;

procedure RunGallery(const Options: TAppOptions);
var
  Gallery: IFaceGallery;
  Reporter: IFaceReporter;
  Removed: Integer;
begin
  Reporter := TFaceConsoleReporter.Create(Options.Verbose);

  Gallery := TFaceGallery.Create;
  Gallery.LoadFromFile(Options.GalleryPath);

  if Options.RemoveName <> '' then
  begin
    Removed := Gallery.RemoveByName(Options.RemoveName);
    Gallery.SaveToFile(Options.GalleryPath);
    Writeln(Format('%d cadastro(s) de "%s" removido(s).',
      [Removed, Options.RemoveName]));
  end;

  Reporter.ReportGallery(Gallery);
  Writeln;
  Writeln('Arquivo: ', Options.GalleryPath);
end;

{ ------------------------------------------------------------------------ ocr }

procedure RunOcr(const Options: TAppOptions; const Runtime: IONNXRuntime);
var
  Engine: IOcrEngine;
  Reporter: IOcrReporter;
  Renderer: IOcrRenderer;
  Loader: IImageLoader;
  Image: IImage;
  Value: TOcrResult;
begin
  if not FileExists(Options.ImagePath) then
    raise Exception.CreateFmt('Imagem nao encontrada: %s', [Options.ImagePath]);
  if not FileExists(Options.OcrDetectPath) then
    raise Exception.CreateFmt(
      'Detector de texto nao encontrado: %s'#13#10 +
      'Baixe PP-OCRv6_medium_det (passo 7 do README) ou informe --ocr-det=ARQ.',
      [Options.OcrDetectPath]);
  if not FileExists(Options.OcrRecognizePath) then
    raise Exception.CreateFmt(
      'Reconhecedor nao encontrado: %s'#13#10 +
      'Baixe PP-OCRv6_medium_rec (passo 7 do README) ou informe --ocr-rec=ARQ.',
      [Options.OcrRecognizePath]);

  Writeln('Imagem : ', Options.ImagePath);

  Reporter := TOcrConsoleReporter.Create(Options.Verbose);
  Engine := TOcrEngineFactory.Build(Runtime, Options.OcrDetectPath,
    Options.OcrRecognizePath, Options.OcrDictionaryPath, Options.Predictor,
    Options.Ocr, Options.Session);
  Reporter.ReportEngine(Engine);

  Loader := TVclImageLoader.Create;
  Image := Loader.Load(Options.ImagePath);

  Value := Engine.Read(Image);
  Reporter.ReportResult(Value);

  Reporter.SaveText(Value, Options.OcrTextPath);
  Writeln;
  Writeln('Texto        : ', Options.OcrTextPath);

  if Options.Render and (Options.OutputPath <> '') then
  begin
    Renderer := TOcrRenderer.Create;
    SaveImageToFile(Renderer.Render(Image, Value), Options.OutputPath);
    Writeln('Imagem anotada: ', Options.OutputPath);
  end;
end;

{ ----------------------------------------------------------------------- main }

procedure Run;
var
  Options: TAppOptions;
  Runtime: IONNXRuntime;
begin
  // Texto lido (OCR) tem acentos e ideogramas; o console OEM os estragaria.
  SetConsoleOutputCP(CP_UTF8);
  SetTextCodePage(Output, CP_UTF8);

  Options := TCommandLineParser.Parse;

  if Options.ShowHelp then
  begin
    Writeln(TCommandLineParser.Usage);
    Exit;
  end;

  Writeln('=== ONNX Runtime + Delphi - inferencia de visao ===');
  Writeln;

  // A galeria e o unico modo que nao precisa de runtime nem de modelo.
  if Options.Mode = amGalleryList then
  begin
    RunGallery(Options);
    Exit;
  end;

  Runtime := TONNXRuntime.Create;

  case Options.Mode of
    amPredict:
      begin
        Writeln('Modelo : ', Options.ModelPath);
        Writeln('Imagem : ', Options.ImagePath);
        if Options.Predictor.LabelsPath <> '' then
          Writeln('Labels : ', Options.Predictor.LabelsPath);
        RunPredict(Options, Runtime);
      end;

    amFaceEnroll:
      RunFaceEnroll(Options, Runtime);

    amFaceQuery:
      RunFaceQuery(Options, Runtime);

    amFaceCompare:
      RunFaceCompare(Options, Runtime);

    amOcr:
      RunOcr(Options, Runtime);
  end;

  Writeln;
  Writeln('=== CONCLUIDO ===');
end;

{ A pausa final e decidida direto da linha de comando: se Run falhar antes
  de terminar o parsing, a flag ainda precisa ser respeitada. }
function WaitForEnterRequested: Boolean;
var
  I: Integer;
begin
  for I := 1 to ParamCount do
    if SameText(ParamStr(I), '--no-pause') then
      Exit(False);
  Result := True;
end;

var
  ExitOnError: Boolean;
begin
  ExitOnError := False;

  try
    Run;
  except
    on E: EOptionsError do
    begin
      Writeln;
      Writeln('ERRO DE PARAMETRO: ', E.Message);
      Writeln;
      Writeln(TCommandLineParser.Usage);
      ExitOnError := True;
    end;
    on E: Exception do
    begin
      Writeln;
      Writeln('ERRO: ', E.ClassName, ': ', E.Message);
      ExitOnError := True;
    end;
  end;

  if ExitOnError then
    ExitCode := 1;

  if WaitForEnterRequested then
  begin
    Writeln;
    Writeln('Pressione ENTER para sair...');
    Readln;
  end;
end.
