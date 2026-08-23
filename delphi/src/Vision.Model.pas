unit Vision.Model;

{
  Descricao do modelo, montada a partir de tres fontes, nessa ordem de
  prioridade:

    1. metadados embutidos no .onnx (Ultralytics grava task, names, imgsz,
       kpt_shape, stride);
    2. formato dos inputs/outputs declarados pelo grafo;
    3. arquivo de rotulos opcional ao lado do modelo.

  E isso que permite o mesmo executavel rodar SqueezeNet, YOLOv8, YOLO11 e
  YOLO26 (detect / segment / pose / obb) sem recompilar nada.
}

interface

uses
  System.SysUtils,
  System.Classes,
  System.IOUtils,
  System.Math,
  System.Generics.Collections,
  ONNX.CApi,
  ONNX.Types,
  Vision.Types;

type
  EModelSpecError = class(Exception);

  TModelSpec = record
    Task: TVisionTask;
    TaskFromMetadata: Boolean;
    ClassNames: TArray<string>;
    InputName: string;
    InputWidth: Integer;
    InputHeight: Integer;
    DynamicInputSize: Boolean;
    KeypointCount: Integer;
    KeypointDims: Integer;
    ProtoChannels: Integer;
    ProducerName: string;
    ModelName: string;
    IsUltralytics: Boolean;
    function ClassCount: Integer;
    function ClassNameOf(Index: Integer): string;
    function Summary: string;
  end;

  TModelSpecReader = class
  private
    class function InferTaskFromOutputs(const Session: IONNXSession;
      const Spec: TModelSpec): TVisionTask; static;
    class procedure ReadInputGeometry(const Session: IONNXSession;
      var Spec: TModelSpec); static;
    class procedure ReadProtoChannels(const Session: IONNXSession;
      var Spec: TModelSpec); static;
  public
    class function Read(const Session: IONNXSession;
      const FallbackLabels: TArray<string>): TModelSpec; static;
  end;

{ Utilitarios de parsing dos metadados Ultralytics. }
function ParseNamesMap(const Text: string): TArray<string>;
function ParseIntList(const Text: string): TArray<Integer>;
function LoadLabelsFile(const FileName: string): TArray<string>;

implementation

{ TModelSpec }

function TModelSpec.ClassCount: Integer;
begin
  Result := Length(ClassNames);
end;

function TModelSpec.ClassNameOf(Index: Integer): string;
begin
  if (Index >= 0) and (Index < Length(ClassNames)) and (ClassNames[Index] <> '') then
    Result := ClassNames[Index]
  else
    Result := Format('classe_%d', [Index]);
end;

function TModelSpec.Summary: string;
begin
  Result := Format('tarefa=%s  entrada=%dx%d  classes=%d',
    [VisionTaskToString(Task), InputWidth, InputHeight, ClassCount]);
  if Task = vtPose then
    Result := Result + Format('  keypoints=%dx%d', [KeypointCount, KeypointDims]);
  if ProtoChannels > 0 then
    Result := Result + Format('  protos=%d', [ProtoChannels]);
end;

{ Parsing }

function ParseNamesMap(const Text: string): TArray<string>;
var
  Pairs: TDictionary<Integer, string>;
  Builder: TStringBuilder;
  I, N, KeyStart, Key, MaxKey: Integer;
  Quote: Char;
  Value: string;
  Pair: TPair<Integer, string>;
begin
  Result := nil;
  N := Length(Text);
  if N = 0 then
    Exit;

  Pairs := TDictionary<Integer, string>.Create;
  try
    I := 1;
    while I <= N do
    begin
      if not CharInSet(Text[I], ['0'..'9']) then
      begin
        Inc(I);
        Continue;
      end;

      KeyStart := I;
      while (I <= N) and CharInSet(Text[I], ['0'..'9']) do
        Inc(I);

      if not TryStrToInt(Copy(Text, KeyStart, I - KeyStart), Key) then
        Continue;

      // chave pode vir entre aspas: {"0": "person"}
      if (I <= N) and CharInSet(Text[I], ['''', '"']) then
        Inc(I);
      while (I <= N) and (Text[I] = ' ') do
        Inc(I);

      if (I > N) or (Text[I] <> ':') then
        Continue;

      Inc(I);
      while (I <= N) and (Text[I] = ' ') do
        Inc(I);

      if (I > N) or (not CharInSet(Text[I], ['''', '"'])) then
        Continue;

      Quote := Text[I];
      Inc(I);

      Builder := TStringBuilder.Create;
      try
        while (I <= N) and (Text[I] <> Quote) do
        begin
          if (Text[I] = '\') and (I < N) then
            Inc(I);
          Builder.Append(Text[I]);
          Inc(I);
        end;
        Value := Builder.ToString;
      finally
        Builder.Free;
      end;

      if I <= N then
        Inc(I); // aspas de fechamento

      Pairs.AddOrSetValue(Key, Value);
    end;

    if Pairs.Count = 0 then
      Exit;

    MaxKey := 0;
    for Pair in Pairs do
      MaxKey := Max(MaxKey, Pair.Key);

    SetLength(Result, MaxKey + 1);
    for Pair in Pairs do
      Result[Pair.Key] := Pair.Value;
  finally
    Pairs.Free;
  end;
end;

function ParseIntList(const Text: string): TArray<Integer>;
var
  I, N, Start, Value: Integer;
  Negative: Boolean;
  List: TList<Integer>;
begin
  Result := nil;
  N := Length(Text);
  if N = 0 then
    Exit;

  List := TList<Integer>.Create;
  try
    I := 1;
    while I <= N do
    begin
      Negative := False;
      if (Text[I] = '-') and (I < N) and CharInSet(Text[I + 1], ['0'..'9']) then
      begin
        Negative := True;
        Inc(I);
      end;

      if CharInSet(Text[I], ['0'..'9']) then
      begin
        Start := I;
        while (I <= N) and CharInSet(Text[I], ['0'..'9']) do
          Inc(I);
        if TryStrToInt(Copy(Text, Start, I - Start), Value) then
        begin
          if Negative then
            Value := -Value;
          List.Add(Value);
        end;
        // ignora parte fracionaria de eventuais floats
        if (I <= N) and (Text[I] = '.') then
        begin
          Inc(I);
          while (I <= N) and CharInSet(Text[I], ['0'..'9']) do
            Inc(I);
        end;
      end
      else
        Inc(I);
    end;
    Result := List.ToArray;
  finally
    List.Free;
  end;
end;

function LoadLabelsFile(const FileName: string): TArray<string>;
var
  Lines: TArray<string>;
  I: Integer;
begin
  Result := nil;
  if (FileName = '') or (not FileExists(FileName)) then
    Exit;

  try
    Lines := TFile.ReadAllLines(FileName, TEncoding.UTF8);
  except
    Lines := TFile.ReadAllLines(FileName);
  end;

  SetLength(Result, Length(Lines));
  for I := 0 to High(Lines) do
    Result[I] := Trim(Lines[I]);
end;

{ TModelSpecReader }

class procedure TModelSpecReader.ReadInputGeometry(const Session: IONNXSession;
  var Spec: TModelSpec);
var
  Info: TTensorInfo;
begin
  if Session.InputCount = 0 then
    raise EModelSpecError.Create('O modelo nao declara nenhuma entrada');

  Info := Session.InputInfo(0);
  Spec.InputName := Info.Name;

  if Info.ElementType <> ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT then
    raise EModelSpecError.CreateFmt(
      'Este pipeline espera entrada FLOAT32; o modelo declara %s',
      [Info.ToString]);

  if Info.Rank <> 4 then
    raise EModelSpecError.CreateFmt(
      'Esperado tensor de entrada NCHW com 4 dimensoes; recebido %s',
      [Info.ShapeText]);

  // NCHW: [batch, canais, altura, largura]
  Spec.InputHeight := Integer(Info.Dim(2));
  Spec.InputWidth := Integer(Info.Dim(3));
  Spec.DynamicInputSize := (Spec.InputWidth <= 0) or (Spec.InputHeight <= 0);
end;

class procedure TModelSpecReader.ReadProtoChannels(const Session: IONNXSession;
  var Spec: TModelSpec);
var
  I: Integer;
  Info: TTensorInfo;
begin
  Spec.ProtoChannels := 0;
  for I := 0 to Session.OutputCount - 1 do
  begin
    Info := Session.OutputInfo(I);
    if Info.Rank = 4 then
    begin
      Spec.ProtoChannels := Integer(Info.Dim(1));
      Exit;
    end;
  end;
end;

class function TModelSpecReader.InferTaskFromOutputs(const Session: IONNXSession;
  const Spec: TModelSpec): TVisionTask;
var
  I, Rank3Count, Rank4Count, Rank2Count: Integer;
  Info, Main: TTensorInfo;
  Channels, Extra, NC: Int64;
begin
  Rank2Count := 0;
  Rank3Count := 0;
  Rank4Count := 0;
  Main := Default(TTensorInfo);

  for I := 0 to Session.OutputCount - 1 do
  begin
    Info := Session.OutputInfo(I);
    case Info.Rank of
      2: begin Inc(Rank2Count); if Rank3Count = 0 then Main := Info; end;
      3: begin Inc(Rank3Count); Main := Info; end;
      4: Inc(Rank4Count);
    end;
  end;

  if (Rank3Count = 0) and (Rank2Count > 0) then
    Exit(vtClassify);

  if Rank3Count = 0 then
    Exit(vtUnknown);

  if Rank4Count > 0 then
    Exit(vtSegment);

  // Sem metadados a distincao detect/pose/obb depende do numero de canais.
  // Exports com eixos dinamicos declaram -1: nesse caso nada a inferir aqui,
  // o decoder resolve o layout com o shape real devolvido pelo Run.
  if (Main.Dim(1) <= 0) or (Main.Dim(2) <= 0) then
    Exit(vtDetect);

  Channels := Min(Main.Dim(1), Main.Dim(2));
  NC := Spec.ClassCount;

  if (NC > 0) and (Channels > 0) then
  begin
    if Channels = 4 + NC then
      Exit(vtDetect);
    if Channels = 5 + NC then
      Exit(vtDetect);           // estilo v5, com coluna de objectness

    Extra := Channels - 4 - NC;
    if (Extra > 0) and (Extra mod 3 = 0) and (Extra >= 9) then
      Exit(vtPose);
    if Extra = 1 then
      Exit(vtObb);
  end;

  // Saida ja decodificada (NMS-free): [1, 300, 6|7|38|57]
  if Channels = 6 then
    Exit(vtDetect);
  if Channels = 7 then
    Exit(vtObb);

  Result := vtDetect;
end;

class function TModelSpecReader.Read(const Session: IONNXSession;
  const FallbackLabels: TArray<string>): TModelSpec;
var
  MetaTask, MetaNames, MetaImgsz, MetaKpt: string;
  Sizes, Kpt: TArray<Integer>;
begin
  Result := Default(TModelSpec);
  Result.Task := vtUnknown;
  Result.KeypointCount := 0;
  Result.KeypointDims := 3;

  ReadInputGeometry(Session, Result);
  ReadProtoChannels(Session, Result);

  Result.ProducerName := Session.ProducerName;
  Result.ModelName := Session.GraphName;

  MetaTask := Session.MetadataValue('task');
  MetaNames := Session.MetadataValue('names');
  MetaImgsz := Session.MetadataValue('imgsz');
  MetaKpt := Session.MetadataValue('kpt_shape');

  Result.IsUltralytics :=
    (MetaTask <> '') or (MetaNames <> '') or
    (Pos('ultralytics', LowerCase(Result.ProducerName)) > 0) or
    (Pos('ultralytics', LowerCase(Session.Description)) > 0);

  if MetaNames <> '' then
    Result.ClassNames := ParseNamesMap(MetaNames);

  if (Length(Result.ClassNames) = 0) and (Length(FallbackLabels) > 0) then
    Result.ClassNames := FallbackLabels;

  if MetaTask <> '' then
  begin
    Result.Task := StringToVisionTask(MetaTask);
    Result.TaskFromMetadata := Result.Task <> vtUnknown;
  end;

  if MetaKpt <> '' then
  begin
    Kpt := ParseIntList(MetaKpt);
    if Length(Kpt) >= 1 then
      Result.KeypointCount := Kpt[0];
    if Length(Kpt) >= 2 then
      Result.KeypointDims := Kpt[1];
  end;

  if Result.DynamicInputSize then
  begin
    if MetaImgsz <> '' then
    begin
      Sizes := ParseIntList(MetaImgsz);
      if Length(Sizes) >= 2 then
      begin
        Result.InputHeight := Sizes[0];
        Result.InputWidth := Sizes[1];
      end
      else if Length(Sizes) = 1 then
      begin
        Result.InputHeight := Sizes[0];
        Result.InputWidth := Sizes[0];
      end;
    end;

    // Ainda dinamico: assume o padrao da familia YOLO.
    if Result.InputWidth <= 0 then
      Result.InputWidth := 640;
    if Result.InputHeight <= 0 then
      Result.InputHeight := 640;
  end;

  if Result.Task = vtUnknown then
    Result.Task := InferTaskFromOutputs(Session, Result);

  if (Result.Task = vtPose) and (Result.KeypointCount <= 0) then
  begin
    // COCO-pose e o padrao de fato quando kpt_shape nao esta nos metadados.
    Result.KeypointCount := 17;
    Result.KeypointDims := 3;
  end;
end;

end.
