unit ONNX.Session;

{
  Sessao de inferencia.

  Responsabilidade unica: executar um grafo ONNX e expor os metadados dele.
  Nao sabe nada sobre imagens, YOLO, caixas ou classes.

  Suporta N inputs e N outputs (segmentacao devolve 2 tensores). Le outputs
  FLOAT32, FLOAT16, DOUBLE, INT8/16/32/64, UINT8/16/32 e BOOL, convertendo
  tudo para Single. Expoe os metadados customizados do modelo, onde a
  Ultralytics grava task, names, imgsz, kpt_shape e stride. Todos os handles
  ORT sao liberados mesmo se o Run falhar no meio.
}

interface

uses
  System.SysUtils,
  System.Classes,
  ONNX.CApi,
  ONNX.Types;

type
  TONNXSession = class(TInterfacedObject, IONNXSession)
  private
    FCore: IOrtCore;
    FModelPath: string;
    FSession: POrtSession;
    FOptions: POrtSessionOptions;
    FInputInfos: TArray<TTensorInfo>;
    FOutputInfos: TArray<TTensorInfo>;

    procedure ApplyConfig(const Config: TSessionConfig);
    procedure CacheIoInfo;
    function ReadTensorInfo(Index: Integer; IsInput: Boolean): TTensorInfo;
    function ReadOutputValue(Value: POrtValue; const AName: string): TTensor;
    function MetadataHandle: POrtModelMetadata;
  public
    constructor Create(const ACore: IOrtCore; const AModelPath: string;
      const Config: TSessionConfig);
    destructor Destroy; override;

    function ModelPath: string;
    function InputCount: Integer;
    function OutputCount: Integer;
    function InputInfo(Index: Integer): TTensorInfo;
    function OutputInfo(Index: Integer): TTensorInfo;
    function InputNames: TArray<string>;
    function OutputNames: TArray<string>;

    function MetadataKeys: TArray<string>;
    function MetadataValue(const Key: string): string;
    function ProducerName: string;
    function GraphName: string;
    function Description: string;

    function Run(const Inputs: TTensorArray): TTensorArray; overload;
    function Run(const Inputs: TTensorArray;
      const RequestedOutputs: TArray<string>): TTensorArray; overload;
  end;

implementation

{ TONNXSession }

constructor TONNXSession.Create(const ACore: IOrtCore; const AModelPath: string;
  const Config: TSessionConfig);
begin
  inherited Create;

  if ACore = nil then
    raise EONNXError.Create('Runtime ONNX invalido');

  if not FileExists(AModelPath) then
    raise EONNXError.CreateFmt('Modelo ONNX nao encontrado: %s', [AModelPath]);

  FCore := ACore;
  FModelPath := AModelPath;

  FCore.Check(FCore.Api^.CreateSessionOptions(FOptions), 'CreateSessionOptions');
  ApplyConfig(Config);

  FCore.Check(
    FCore.Api^.CreateSession(FCore.Env, PWideChar(FModelPath), FOptions, FSession),
    'CreateSession');

  CacheIoInfo;
end;

destructor TONNXSession.Destroy;
begin
  if FCore <> nil then
  begin
    if FSession <> nil then
      FCore.Api^.ReleaseSession(FSession);
    if FOptions <> nil then
      FCore.Api^.ReleaseSessionOptions(FOptions);
  end;

  FSession := nil;
  FOptions := nil;
  FCore := nil;

  inherited;
end;

procedure TONNXSession.ApplyConfig(const Config: TSessionConfig);
begin
  FCore.Check(
    FCore.Api^.SetGraphOptimizationLevel(FOptions, Config.GraphOptimizationLevel),
    'SetSessionGraphOptimizationLevel');

  FCore.Check(
    FCore.Api^.SetExecutionMode(FOptions, Config.ExecutionMode),
    'SetSessionExecutionMode');

  if Config.IntraOpThreads > 0 then
    FCore.Check(
      FCore.Api^.SetIntraOpNumThreads(FOptions, Config.IntraOpThreads),
      'SetIntraOpNumThreads');

  if Config.InterOpThreads > 0 then
    FCore.Check(
      FCore.Api^.SetInterOpNumThreads(FOptions, Config.InterOpThreads),
      'SetInterOpNumThreads');
end;

procedure TONNXSession.CacheIoInfo;
var
  I, N: Integer;
  Count: TOrtSize;
begin
  Count := 0;
  FCore.Check(FCore.Api^.SessionGetInputCount(FSession, Count), 'SessionGetInputCount');
  N := Integer(Count);
  SetLength(FInputInfos, N);
  for I := 0 to N - 1 do
    FInputInfos[I] := ReadTensorInfo(I, True);

  Count := 0;
  FCore.Check(FCore.Api^.SessionGetOutputCount(FSession, Count), 'SessionGetOutputCount');
  N := Integer(Count);
  SetLength(FOutputInfos, N);
  for I := 0 to N - 1 do
    FOutputInfos[I] := ReadTensorInfo(I, False);
end;

function TONNXSession.ReadTensorInfo(Index: Integer; IsInput: Boolean): TTensorInfo;
var
  TypeInfo: POrtTypeInfo;
  TensorInfo: POrtTensorTypeAndShapeInfo;
  DimCount: TOrtSize;
  NamePtr: PAnsiChar;
begin
  Result := Default(TTensorInfo);

  NamePtr := nil;
  if IsInput then
    FCore.Check(
      FCore.Api^.SessionGetInputName(FSession, Index, FCore.Allocator, NamePtr),
      'SessionGetInputName')
  else
    FCore.Check(
      FCore.Api^.SessionGetOutputName(FSession, Index, FCore.Allocator, NamePtr),
      'SessionGetOutputName');
  Result.Name := FCore.ConsumeString(NamePtr);

  TypeInfo := nil;
  if IsInput then
    FCore.Check(
      FCore.Api^.SessionGetInputTypeInfo(FSession, Index, TypeInfo),
      'SessionGetInputTypeInfo')
  else
    FCore.Check(
      FCore.Api^.SessionGetOutputTypeInfo(FSession, Index, TypeInfo),
      'SessionGetOutputTypeInfo');

  try
    TensorInfo := nil;
    FCore.Check(
      FCore.Api^.CastTypeInfoToTensorInfo(TypeInfo, TensorInfo),
      'CastTypeInfoToTensorInfo');

    // Outputs que nao sao tensores (mapas, sequencias) devolvem nil.
    if TensorInfo = nil then
    begin
      Result.ElementType := ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      Exit;
    end;

    FCore.Check(
      FCore.Api^.GetTensorElementType(TensorInfo, Result.ElementType),
      'GetTensorElementType');

    DimCount := 0;
    FCore.Check(
      FCore.Api^.GetDimensionsCount(TensorInfo, DimCount),
      'GetDimensionsCount');

    SetLength(Result.Shape, Integer(DimCount));
    if DimCount > 0 then
      FCore.Check(
        FCore.Api^.GetDimensions(TensorInfo, @Result.Shape[0], DimCount),
        'GetDimensions');
  finally
    if TypeInfo <> nil then
      FCore.Api^.ReleaseTypeInfo(TypeInfo);
  end;
end;

function TONNXSession.ModelPath: string;
begin
  Result := FModelPath;
end;

function TONNXSession.InputCount: Integer;
begin
  Result := Length(FInputInfos);
end;

function TONNXSession.OutputCount: Integer;
begin
  Result := Length(FOutputInfos);
end;

function TONNXSession.InputInfo(Index: Integer): TTensorInfo;
begin
  if (Index < 0) or (Index >= Length(FInputInfos)) then
    raise EONNXError.CreateFmt('Indice de input invalido: %d', [Index]);
  Result := FInputInfos[Index];
end;

function TONNXSession.OutputInfo(Index: Integer): TTensorInfo;
begin
  if (Index < 0) or (Index >= Length(FOutputInfos)) then
    raise EONNXError.CreateFmt('Indice de output invalido: %d', [Index]);
  Result := FOutputInfos[Index];
end;

function TONNXSession.InputNames: TArray<string>;
var
  I: Integer;
begin
  SetLength(Result, Length(FInputInfos));
  for I := 0 to High(FInputInfos) do
    Result[I] := FInputInfos[I].Name;
end;

function TONNXSession.OutputNames: TArray<string>;
var
  I: Integer;
begin
  SetLength(Result, Length(FOutputInfos));
  for I := 0 to High(FOutputInfos) do
    Result[I] := FOutputInfos[I].Name;
end;

function TONNXSession.MetadataHandle: POrtModelMetadata;
begin
  Result := nil;
  FCore.Check(
    FCore.Api^.SessionGetModelMetadata(FSession, Result),
    'SessionGetModelMetadata');
end;

function TONNXSession.MetadataKeys: TArray<string>;
var
  Metadata: POrtModelMetadata;
  Keys: POrtCharPtr;
  KeyCount: Int64;
  Cursor: POrtCharPtr;
  I: Integer;
begin
  Result := nil;
  Metadata := MetadataHandle;
  if Metadata = nil then
    Exit;
  try
    Keys := nil;
    KeyCount := 0;
    FCore.Check(
      FCore.Api^.MetadataGetKeys(Metadata, FCore.Allocator, Keys, KeyCount),
      'ModelMetadataGetCustomMetadataMapKeys');

    if (Keys = nil) or (KeyCount <= 0) then
      Exit;

    try
      SetLength(Result, Integer(KeyCount));
      Cursor := Keys;
      for I := 0 to Integer(KeyCount) - 1 do
      begin
        Result[I] := FCore.ConsumeString(Cursor^);
        Inc(Cursor);
      end;
    finally
      FCore.Api^.AllocatorFree(FCore.Allocator, Keys);
    end;
  finally
    FCore.Api^.ReleaseModelMetadata(Metadata);
  end;
end;

function TONNXSession.MetadataValue(const Key: string): string;
var
  Metadata: POrtModelMetadata;
  KeyA: AnsiString;
  ValuePtr: PAnsiChar;
begin
  Result := '';
  Metadata := MetadataHandle;
  if Metadata = nil then
    Exit;
  try
    KeyA := AnsiString(Key);
    ValuePtr := nil;
    FCore.Check(
      FCore.Api^.MetadataLookup(Metadata, FCore.Allocator, PAnsiChar(KeyA), ValuePtr),
      'ModelMetadataLookupCustomMetadataMap');
    Result := FCore.ConsumeString(ValuePtr);
  finally
    FCore.Api^.ReleaseModelMetadata(Metadata);
  end;
end;

function TONNXSession.ProducerName: string;
var
  Metadata: POrtModelMetadata;
  ValuePtr: PAnsiChar;
begin
  Result := '';
  Metadata := MetadataHandle;
  if Metadata = nil then
    Exit;
  try
    ValuePtr := nil;
    FCore.Check(
      FCore.Api^.MetadataGetProducerName(Metadata, FCore.Allocator, ValuePtr),
      'ModelMetadataGetProducerName');
    Result := FCore.ConsumeString(ValuePtr);
  finally
    FCore.Api^.ReleaseModelMetadata(Metadata);
  end;
end;

function TONNXSession.GraphName: string;
var
  Metadata: POrtModelMetadata;
  ValuePtr: PAnsiChar;
begin
  Result := '';
  Metadata := MetadataHandle;
  if Metadata = nil then
    Exit;
  try
    ValuePtr := nil;
    FCore.Check(
      FCore.Api^.MetadataGetGraphName(Metadata, FCore.Allocator, ValuePtr),
      'ModelMetadataGetGraphName');
    Result := FCore.ConsumeString(ValuePtr);
  finally
    FCore.Api^.ReleaseModelMetadata(Metadata);
  end;
end;

function TONNXSession.Description: string;
var
  Metadata: POrtModelMetadata;
  ValuePtr: PAnsiChar;
begin
  Result := '';
  Metadata := MetadataHandle;
  if Metadata = nil then
    Exit;
  try
    ValuePtr := nil;
    FCore.Check(
      FCore.Api^.MetadataGetDescription(Metadata, FCore.Allocator, ValuePtr),
      'ModelMetadataGetDescription');
    Result := FCore.ConsumeString(ValuePtr);
  finally
    FCore.Api^.ReleaseModelMetadata(Metadata);
  end;
end;

function TONNXSession.Run(const Inputs: TTensorArray): TTensorArray;
begin
  Result := Run(Inputs, OutputNames);
end;

function TONNXSession.Run(const Inputs: TTensorArray;
  const RequestedOutputs: TArray<string>): TTensorArray;
var
  MemoryInfo: POrtMemoryInfo;
  RunOptions: POrtRunOptions;
  InputNamesA: TArray<AnsiString>;
  OutputNamesA: TArray<AnsiString>;
  InputNamePtrs: TArray<PAnsiChar>;
  OutputNamePtrs: TArray<PAnsiChar>;
  InputValues: TArray<POrtValue>;
  OutputValues: TArray<POrtValue>;
  NIn, NOut, I: Integer;
begin
  Result := nil;

  NIn := Length(Inputs);
  NOut := Length(RequestedOutputs);

  if NIn = 0 then
    raise EONNXError.Create('Run exige pelo menos um tensor de entrada');
  if NOut = 0 then
    raise EONNXError.Create('Run exige pelo menos um output solicitado');

  for I := 0 to NIn - 1 do
  begin
    if Length(Inputs[I].Data) = 0 then
      raise EONNXError.CreateFmt('Tensor de entrada "%s" esta vazio', [Inputs[I].Name]);
    if Length(Inputs[I].Shape) = 0 then
      raise EONNXError.CreateFmt('Tensor de entrada "%s" nao tem shape', [Inputs[I].Name]);
  end;

  SetLength(InputNamesA, NIn);
  SetLength(InputNamePtrs, NIn);
  SetLength(InputValues, NIn);
  SetLength(OutputNamesA, NOut);
  SetLength(OutputNamePtrs, NOut);
  SetLength(OutputValues, NOut);

  for I := 0 to NIn - 1 do
  begin
    InputNamesA[I] := AnsiString(Inputs[I].Name);
    InputNamePtrs[I] := PAnsiChar(InputNamesA[I]);
    InputValues[I] := nil;
  end;

  for I := 0 to NOut - 1 do
  begin
    OutputNamesA[I] := AnsiString(RequestedOutputs[I]);
    OutputNamePtrs[I] := PAnsiChar(OutputNamesA[I]);
    OutputValues[I] := nil;
  end;

  MemoryInfo := nil;
  RunOptions := nil;

  FCore.Check(
    FCore.Api^.CreateCpuMemoryInfo(ORT_DEVICE_ALLOCATOR, ORT_MEM_TYPE_DEFAULT, MemoryInfo),
    'CreateCpuMemoryInfo');
  try
    FCore.Check(FCore.Api^.CreateRunOptions(RunOptions), 'CreateRunOptions');
    try
      try
        for I := 0 to NIn - 1 do
          FCore.Check(
            FCore.Api^.CreateTensorWithDataAsOrtValue(
              MemoryInfo,
              @Inputs[I].Data[0],
              TOrtSize(Length(Inputs[I].Data)) * SizeOf(Single),
              @Inputs[I].Shape[0],
              Length(Inputs[I].Shape),
              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
              InputValues[I]),
            'CreateTensorWithDataAsOrtValue');

        FCore.Check(
          FCore.Api^.Run(
            FSession,
            RunOptions,
            @InputNamePtrs[0],
            @InputValues[0],
            NIn,
            @OutputNamePtrs[0],
            NOut,
            @OutputValues[0]),
          'Run');

        SetLength(Result, NOut);
        for I := 0 to NOut - 1 do
        begin
          if OutputValues[I] = nil then
            raise EONNXError.CreateFmt(
              'ONNX Runtime nao devolveu o output "%s"', [RequestedOutputs[I]]);
          Result[I] := ReadOutputValue(OutputValues[I], RequestedOutputs[I]);
        end;
      finally
        for I := 0 to NOut - 1 do
          if OutputValues[I] <> nil then
            FCore.Api^.ReleaseValue(OutputValues[I]);

        for I := 0 to NIn - 1 do
          if InputValues[I] <> nil then
            FCore.Api^.ReleaseValue(InputValues[I]);
      end;
    finally
      FCore.Api^.ReleaseRunOptions(RunOptions);
    end;
  finally
    FCore.Api^.ReleaseMemoryInfo(MemoryInfo);
  end;
end;

function TONNXSession.ReadOutputValue(Value: POrtValue; const AName: string): TTensor;
var
  ShapeInfo: POrtTensorTypeAndShapeInfo;
  DimCount: TOrtSize;
  ElementCount: TOrtSize;
  ElementType: Integer;
  Raw: Pointer;
  Count, I: Integer;
begin
  Result := Default(TTensor);
  Result.Name := AName;

  ShapeInfo := nil;
  FCore.Check(
    FCore.Api^.GetTensorTypeAndShape(Value, ShapeInfo),
    'GetTensorTypeAndShape');
  try
    DimCount := 0;
    FCore.Check(FCore.Api^.GetDimensionsCount(ShapeInfo, DimCount), 'GetDimensionsCount');

    SetLength(Result.Shape, Integer(DimCount));
    if DimCount > 0 then
      FCore.Check(
        FCore.Api^.GetDimensions(ShapeInfo, @Result.Shape[0], DimCount),
        'GetDimensions');

    ElementType := ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    FCore.Check(
      FCore.Api^.GetTensorElementType(ShapeInfo, ElementType),
      'GetTensorElementType');

    ElementCount := 0;
    FCore.Check(
      FCore.Api^.GetTensorShapeElementCount(ShapeInfo, ElementCount),
      'GetTensorShapeElementCount');
    Count := Integer(ElementCount);
  finally
    FCore.Api^.ReleaseTensorTypeAndShapeInfo(ShapeInfo);
  end;

  SetLength(Result.Data, Count);
  if Count = 0 then
    Exit;

  Raw := nil;
  FCore.Check(FCore.Api^.GetTensorMutableData(Value, Raw), 'GetTensorMutableData');
  if Raw = nil then
    raise EONNXError.CreateFmt('Output "%s" nao expos dados', [AName]);

  case ElementType of
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      Move(Raw^, Result.Data[0], Count * SizeOf(Single));

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      for I := 0 to Count - 1 do
        Result.Data[I] := PDouble(PByte(Raw) + I * SizeOf(Double))^;

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      for I := 0 to Count - 1 do
        Result.Data[I] := HalfToSingle(PWord(PByte(Raw) + I * SizeOf(Word))^);

    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      for I := 0 to Count - 1 do
        Result.Data[I] := PInt64(PByte(Raw) + I * SizeOf(Int64))^;

    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      for I := 0 to Count - 1 do
        Result.Data[I] := PInteger(PByte(Raw) + I * SizeOf(Integer))^;

    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      for I := 0 to Count - 1 do
        Result.Data[I] := PCardinal(PByte(Raw) + I * SizeOf(Cardinal))^;

    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      for I := 0 to Count - 1 do
        Result.Data[I] := PSmallInt(PByte(Raw) + I * SizeOf(SmallInt))^;

    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      for I := 0 to Count - 1 do
        Result.Data[I] := PWord(PByte(Raw) + I * SizeOf(Word))^;

    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      for I := 0 to Count - 1 do
        Result.Data[I] := PShortInt(PByte(Raw) + I)^;

    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      for I := 0 to Count - 1 do
        Result.Data[I] := PByte(PByte(Raw) + I)^;
  else
    raise EONNXError.CreateFmt(
      'Output "%s" tem tipo nao suportado: %s',
      [AName, OrtElementTypeName(ElementType)]);
  end;
end;

end.
