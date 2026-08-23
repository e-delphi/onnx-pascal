unit App.Options;

{
  Leitura da linha de comando.

  Fica separada para que nem o pipeline nem o main precisem saber como os
  parametros chegaram: o main so recebe um TAppOptions pronto.

  Ha dois modos de operacao. O padrao roda um modelo sobre uma imagem. O modo
  facial encadeia dois modelos (detector + embedding) e opera sobre uma
  galeria, entao nao usa o argumento posicional de modelo - os caminhos vem
  de --face-detect / --face-embed, com padrao na subpasta face/.
}

interface

uses
  System.SysUtils,
  System.IOUtils,
  System.Math,
  System.Generics.Collections,
  ONNX.Types,
  Vision.Types,
  Vision.Predictor,
  Vision.Embedding,
  Vision.Face;

type
  EOptionsError = class(Exception);

  TAppMode = (
    amPredict,        // modelo + imagem (comportamento padrao)
    amFaceEnroll,     // cadastra um rosto na galeria
    amFaceQuery,      // consulta uma foto contra a galeria
    amFaceCompare,    // compara duas fotos diretamente
    amGalleryList     // lista o conteudo da galeria
  );

  TAppOptions = record
    Mode: TAppMode;

    ModelPath: string;
    ImagePath: string;
    SecondImagePath: string;
    OutputPath: string;
    OutputDir: string;

    FaceDetectPath: string;
    FaceEmbedPath: string;
    GalleryPath: string;
    EnrollName: string;
    RemoveName: string;
    AlignedDir: string;
    FaceThreshold: Single;

    Render: Boolean;
    Verbose: Boolean;
    Pause: Boolean;
    ShowHelp: Boolean;
    Threads: Integer;

    Predictor: TPredictorOptions;
    Face: TFaceOptions;
    Session: TSessionConfig;
  end;

  TCommandLineParser = class
  private
    class function BaseDirectory: string; static;
    class function ResolvePath(const Value: string): string; static;
    class function ParseFloat(const Value, Option: string): Single; static;
    class function ParseInt(const Value, Option: string): Integer; static;
    class procedure ApplyDefaults(var Options: TAppOptions); static;
    class procedure Validate(const Options: TAppOptions); static;
  public
    class function Parse: TAppOptions; static;
    class function Usage: string; static;
  end;

implementation

class function TCommandLineParser.BaseDirectory: string;
begin
  Result := ExtractFilePath(ParamStr(0));
end;

class function TCommandLineParser.ResolvePath(const Value: string): string;
begin
  if Value = '' then
    Exit('');

  if TPath.IsPathRooted(Value) then
    Result := Value
  else
    Result := TPath.Combine(BaseDirectory, Value);

  // Normaliza as barras. Um caminho com "/" vindo da linha de comando faz
  // ExtractFilePath truncar no lugar errado, e a imagem anotada acabaria
  // salva na pasta do executavel em vez da pasta da imagem.
  Result := TPath.GetFullPath(Result);
end;

class function TCommandLineParser.ParseFloat(const Value, Option: string): Single;
var
  Parsed: Double;
begin
  // Aceita ponto como separador decimal independente do locale da maquina.
  if not TryStrToFloat(Value, Parsed, TFormatSettings.Invariant) then
    raise EOptionsError.CreateFmt('Valor invalido para %s: "%s"', [Option, Value]);
  Result := Parsed;
end;

class function TCommandLineParser.ParseInt(const Value, Option: string): Integer;
begin
  if not TryStrToInt(Value, Result) then
    raise EOptionsError.CreateFmt('Valor invalido para %s: "%s"', [Option, Value]);
end;

class function TCommandLineParser.Usage: string;
begin
  Result :=
    'Uso: ONNXDemo.exe <modelo.onnx> <imagem> [opcoes]' + sLineBreak +
    '     ONNXDemo.exe --enroll=NOME <imagem> [--gallery=ARQ]' + sLineBreak +
    '     ONNXDemo.exe --query <imagem> [--gallery=ARQ]' + sLineBreak +
    '     ONNXDemo.exe --compare <imagem1> <imagem2>' + sLineBreak +
    '     ONNXDemo.exe --list [--gallery=ARQ]' + sLineBreak +
    sLineBreak +
    'INFERENCIA' + sLineBreak +
    '  Modelo e imagem sao obrigatorios. O argumento terminado em .onnx e' + sLineBreak +
    '  tomado como modelo; o outro, como imagem.' + sLineBreak +
    sLineBreak +
    '  --model=ARQ      modelo .onnx (alternativa ao posicional)' + sLineBreak +
    '  --image=ARQ      imagem de entrada (alternativa ao posicional)' + sLineBreak +
    '  --labels=ARQ     rotulos, um por linha; usado so quando o modelo nao' + sLineBreak +
    '                   traz os nomes nos metadados. Sem esta opcao, procura' + sLineBreak +
    '                   labels.txt na pasta do executavel.' + sLineBreak +
    '  --task=T         forca: classify, detect, segment, pose, obb, face, embed' + sLineBreak +
    '  --conf=N         limiar de confianca (padrao 0.25; 0 em classificacao)' + sLineBreak +
    '  --iou=N          limiar de IoU do NMS (padrao 0.45)' + sLineBreak +
    '  --max-det=N      maximo de deteccoes (padrao 300)' + sLineBreak +
    '  --topk=N         classes listadas em classificacao (padrao 5)' + sLineBreak +
    '  --mask-thr=N     limiar da mascara em segmentacao (padrao 0.5)' + sLineBreak +
    '  --agnostic       NMS sem separar por classe' + sLineBreak +
    '  --multi-crop     classificacao com 5 recortes (centro + cantos)' + sLineBreak +
    '  --norm=MODO      auto | imagenet | unit (normalizacao em classificacao)' + sLineBreak +
    '  --out=ARQ        salva a imagem anotada num caminho especifico' + sLineBreak +
    '  --out-dir=DIR    pasta das saidas geradas (padrao: saida/)' + sLineBreak +
    '  --no-render      nao gera imagem anotada' + sLineBreak +
    sLineBreak +
    'ROSTOS (encadeia detector + embedding; nao usa --model)' + sLineBreak +
    '  --enroll=NOME    cadastra na galeria o maior rosto da imagem' + sLineBreak +
    '  --query          lista os rostos da galeria mais parecidos' + sLineBreak +
    '  --compare        compara duas imagens entre si' + sLineBreak +
    '  --list           lista o conteudo da galeria' + sLineBreak +
    '  --remove=NOME    remove todos os cadastros com esse nome' + sLineBreak +
    '  --gallery=ARQ    arquivo da galeria (padrao: faces.gallery)' + sLineBreak +
    '  --face-detect=A  detector com 5 landmarks (padrao: face/det_10g.onnx)' + sLineBreak +
    '  --face-embed=A   modelo de embedding (padrao: face/w600k_r50.onnx)' + sLineBreak +
    '  --face-thr=N     limiar de cosseno para mesma pessoa (padrao 0.40)' + sLineBreak +
    '  --min-face=N     ignora rostos menores que N pixels (padrao 24)' + sLineBreak +
    '  --save-aligned   salva os recortes 112x112 alinhados (aceita =PASTA)' + sLineBreak +
    sLineBreak +
    'GERAL' + sLineBreak +
    '  --threads=N      threads intra-op do ONNX Runtime' + sLineBreak +
    '  --quiet          reduz a saida no console' + sLineBreak +
    '  --no-pause       nao espera ENTER ao final' + sLineBreak +
    '  --help           mostra esta ajuda' + sLineBreak +
    sLineBreak +
    'Exemplos:' + sLineBreak +
    '  ONNXDemo.exe yolo26n.onnx foto.jpg' + sLineBreak +
    '  ONNXDemo.exe yolo26n-pose.onnx foto.jpg --conf=0.4 --out=pose.png' + sLineBreak +
    '  ONNXDemo.exe --enroll="Eduardo" eduardo1.jpg' + sLineBreak +
    '  ONNXDemo.exe --query desconhecido.jpg --face-thr=0.45' + sLineBreak +
    '  ONNXDemo.exe --compare a.jpg b.jpg';
end;

class procedure TCommandLineParser.ApplyDefaults(var Options: TAppOptions);
begin
  if Options.Mode = amPredict then
  begin
    if Options.ModelPath <> '' then
      Options.ModelPath := ResolvePath(Options.ModelPath);
  end
  else
  begin
    if Options.FaceDetectPath = '' then
      Options.FaceDetectPath := TPath.Combine(BaseDirectory, 'face\det_10g.onnx');
    if Options.FaceEmbedPath = '' then
      Options.FaceEmbedPath := TPath.Combine(BaseDirectory, 'face\w600k_r50.onnx');
    if Options.GalleryPath = '' then
      Options.GalleryPath := TPath.Combine(BaseDirectory, 'faces.gallery');

    Options.FaceDetectPath := ResolvePath(Options.FaceDetectPath);
    Options.FaceEmbedPath := ResolvePath(Options.FaceEmbedPath);
    Options.GalleryPath := ResolvePath(Options.GalleryPath);
  end;

  // Saidas geradas nunca caem junto das entradas: por padrao vao para
  // saida/ ao lado do executavel.
  if Options.OutputDir = '' then
    Options.OutputDir := TPath.Combine(BaseDirectory, 'saida');
  Options.OutputDir := ResolvePath(Options.OutputDir);

  if Options.ImagePath <> '' then
    Options.ImagePath := ResolvePath(Options.ImagePath);
  if Options.SecondImagePath <> '' then
    Options.SecondImagePath := ResolvePath(Options.SecondImagePath);
  if Options.Face.KeepAligned then
  begin
    if Options.AlignedDir = '' then
      Options.AlignedDir := TPath.Combine(Options.OutputDir, 'alinhados');
    Options.AlignedDir := ResolvePath(Options.AlignedDir);
  end;

  if Options.Predictor.LabelsPath = '' then
  begin
    Options.Predictor.LabelsPath := TPath.Combine(BaseDirectory, 'labels.txt');
    if not FileExists(Options.Predictor.LabelsPath) then
      Options.Predictor.LabelsPath := '';
  end
  else
    Options.Predictor.LabelsPath := ResolvePath(Options.Predictor.LabelsPath);

  if Options.OutputPath <> '' then
    Options.OutputPath := ResolvePath(Options.OutputPath)
  else if Options.Render and (Options.Mode = amPredict) and
          (Options.ImagePath <> '') then
    Options.OutputPath := TPath.Combine(Options.OutputDir,
      TPath.GetFileNameWithoutExtension(Options.ImagePath) + '_pred.png');

  Options.Predictor.ConfidenceThreshold :=
    Min(1.0, Max(0.0, Options.Predictor.ConfidenceThreshold));
  Options.Predictor.IoUThreshold :=
    Min(1.0, Max(0.0, Options.Predictor.IoUThreshold));
  Options.Predictor.MaskThreshold :=
    Min(1.0, Max(0.0, Options.Predictor.MaskThreshold));
  Options.Predictor.MaxDetections := Max(1, Options.Predictor.MaxDetections);
  Options.Predictor.TopK := Max(1, Options.Predictor.TopK);
  Options.FaceThreshold := Min(1.0, Max(-1.0, Options.FaceThreshold));
end;

class procedure TCommandLineParser.Validate(const Options: TAppOptions);
begin
  case Options.Mode of
    amPredict:
      begin
        if Options.ModelPath = '' then
          raise EOptionsError.Create(
            'Informe o modelo .onnx (posicional ou --model=ARQ).');
        if Options.ImagePath = '' then
          raise EOptionsError.Create(
            'Informe a imagem de entrada (posicional ou --image=ARQ).');
      end;

    amFaceEnroll:
      begin
        if Trim(Options.EnrollName) = '' then
          raise EOptionsError.Create('--enroll precisa de um nome.');
        if Options.ImagePath = '' then
          raise EOptionsError.Create('Informe a imagem com o rosto a cadastrar.');
      end;

    amFaceQuery:
      if Options.ImagePath = '' then
        raise EOptionsError.Create('Informe a imagem a consultar.');

    amFaceCompare:
      if (Options.ImagePath = '') or (Options.SecondImagePath = '') then
        raise EOptionsError.Create('--compare precisa de duas imagens.');

    amGalleryList:
      ; // so precisa da galeria, que ja tem padrao
  end;
end;

class function TCommandLineParser.Parse: TAppOptions;
var
  Positional: TList<string>;
  I, SeparatorPos: Integer;
  Argument, Name, Value: string;
  Task: TVisionTask;
begin
  Result := Default(TAppOptions);
  Result.Mode := amPredict;
  Result.Predictor := TPredictorOptions.Default;
  Result.Face := TFaceOptions.Default;
  Result.Session := TSessionConfig.Default;
  Result.FaceThreshold := DEFAULT_FACE_THRESHOLD;
  Result.Render := True;
  Result.Verbose := True;
  Result.Pause := True;

  // Sem nenhum argumento nao ha o que adivinhar: mostra a ajuda.
  if ParamCount = 0 then
  begin
    Result.ShowHelp := True;
    Exit;
  end;

  Positional := TList<string>.Create;
  try
    for I := 1 to ParamCount do
    begin
      Argument := ParamStr(I);

      if not Argument.StartsWith('--') then
      begin
        Positional.Add(Argument);
        Continue;
      end;

      SeparatorPos := Argument.IndexOf('=');
      if SeparatorPos > 0 then
      begin
        Name := LowerCase(Argument.Substring(2, SeparatorPos - 2));
        Value := Argument.Substring(SeparatorPos + 1);
      end
      else
      begin
        Name := LowerCase(Argument.Substring(2));
        Value := '';
      end;

      if Name = 'help' then
        Result.ShowHelp := True
      else if Name = 'model' then
        Result.ModelPath := Value
      else if Name = 'image' then
        Result.ImagePath := Value
      else if Name = 'labels' then
        Result.Predictor.LabelsPath := Value
      else if Name = 'out' then
        Result.OutputPath := Value
      else if Name = 'out-dir' then
        Result.OutputDir := Value
      else if Name = 'no-render' then
        Result.Render := False
      else if Name = 'quiet' then
        Result.Verbose := False
      else if Name = 'no-pause' then
        Result.Pause := False
      else if Name = 'agnostic' then
        Result.Predictor.ClassAgnosticNms := True
      else if Name = 'multi-crop' then
        Result.Predictor.MultiCropClassify := True
      else if Name = 'enroll' then
      begin
        Result.Mode := amFaceEnroll;
        Result.EnrollName := Value;
      end
      else if Name = 'query' then
        Result.Mode := amFaceQuery
      else if Name = 'compare' then
        Result.Mode := amFaceCompare
      else if Name = 'list' then
        Result.Mode := amGalleryList
      else if Name = 'remove' then
      begin
        Result.Mode := amGalleryList;
        Result.RemoveName := Value;
      end
      else if Name = 'gallery' then
        Result.GalleryPath := Value
      else if Name = 'face-detect' then
        Result.FaceDetectPath := Value
      else if Name = 'face-embed' then
        Result.FaceEmbedPath := Value
      else if Name = 'save-aligned' then
      begin
        // Sem valor, cai na subpasta padrao de saida.
        Result.AlignedDir := Value;
        Result.Face.KeepAligned := True;
      end
      else if Name = 'face-thr' then
        Result.FaceThreshold := ParseFloat(Value, '--face-thr')
      else if Name = 'min-face' then
        Result.Face.MinFaceSize := ParseInt(Value, '--min-face')
      else if Name = 'conf' then
      begin
        Result.Predictor.ConfidenceThreshold := ParseFloat(Value, '--conf');
        Result.Predictor.ConfidenceWasSet := True;
        Result.Face.DetectThreshold := Result.Predictor.ConfidenceThreshold;
      end
      else if Name = 'iou' then
      begin
        Result.Predictor.IoUThreshold := ParseFloat(Value, '--iou');
        Result.Face.NmsThreshold := Result.Predictor.IoUThreshold;
      end
      else if Name = 'mask-thr' then
        Result.Predictor.MaskThreshold := ParseFloat(Value, '--mask-thr')
      else if Name = 'max-det' then
        Result.Predictor.MaxDetections := ParseInt(Value, '--max-det')
      else if Name = 'topk' then
        Result.Predictor.TopK := ParseInt(Value, '--topk')
      else if Name = 'threads' then
      begin
        Result.Threads := ParseInt(Value, '--threads');
        Result.Session.IntraOpThreads := Result.Threads;
      end
      else if Name = 'norm' then
      begin
        if SameText(Value, 'imagenet') then
          Result.Predictor.Normalization := ncImageNet
        else if SameText(Value, 'unit') then
          Result.Predictor.Normalization := ncUnit
        else if SameText(Value, 'auto') then
          Result.Predictor.Normalization := ncAuto
        else
          raise EOptionsError.CreateFmt('Valor invalido para --norm: "%s"', [Value]);
      end
      else if Name = 'task' then
      begin
        Task := StringToVisionTask(Value);
        if Task = vtUnknown then
          raise EOptionsError.CreateFmt('Tarefa desconhecida: "%s"', [Value]);
        Result.Predictor.TaskOverride := Task;
      end
      else
        raise EOptionsError.CreateFmt('Opcao desconhecida: %s', [Argument]);
    end;

    // Posicionais: .onnx vira modelo, o resto vira imagem (duas em --compare).
    for I := 0 to Positional.Count - 1 do
    begin
      if SameText(ExtractFileExt(Positional[I]), '.onnx') then
      begin
        if Result.ModelPath = '' then
          Result.ModelPath := Positional[I];
      end
      else if Result.ImagePath = '' then
        Result.ImagePath := Positional[I]
      else if Result.SecondImagePath = '' then
        Result.SecondImagePath := Positional[I];
    end;
  finally
    Positional.Free;
  end;

  if Result.ShowHelp then
    Exit;

  ApplyDefaults(Result);
  Validate(Result);
end;

end.
