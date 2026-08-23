unit ONNX.CApi;

{
  Binding de baixo nivel para a ONNX Runtime C API 1.28.x (ORT_API_VERSION = 28).

  Responsabilidade unica: descrever a ABI da DLL.
  Nao aloca nada, nao levanta excecoes, nao conhece imagens nem modelos.

  A struct OrtApi do header e composta apenas por ponteiros de funcao, um por
  slot, na ordem de declaracao. Os indices abaixo foram extraidos diretamente
  de onnxruntime_c_api.h (1.28.1) e o total de slots (424) confere com o
  tamanho do array declarado aqui.
}

interface

const
  ORT_API_VERSION    = 28;
  ORT_API_SLOT_COUNT = 424;

  ORT_LOGGING_LEVEL_VERBOSE = 0;
  ORT_LOGGING_LEVEL_INFO    = 1;
  ORT_LOGGING_LEVEL_WARNING = 2;
  ORT_LOGGING_LEVEL_ERROR   = 3;
  ORT_LOGGING_LEVEL_FATAL   = 4;

  // GraphOptimizationLevel
  ORT_DISABLE_ALL     = 0;
  ORT_ENABLE_BASIC    = 1;
  ORT_ENABLE_EXTENDED = 2;
  ORT_ENABLE_LAYOUT   = 3;
  ORT_ENABLE_ALL      = 99;

  // ExecutionMode
  ORT_SEQUENTIAL = 0;
  ORT_PARALLEL   = 1;

  // OrtAllocatorType
  ORT_INVALID_ALLOCATOR   = -1;
  ORT_DEVICE_ALLOCATOR    = 0;
  ORT_ARENA_ALLOCATOR     = 1;
  ORT_READ_ONLY_ALLOCATOR = 2;

  // OrtMemType
  ORT_MEM_TYPE_CPU_INPUT  = -2;
  ORT_MEM_TYPE_CPU_OUTPUT = -1;
  ORT_MEM_TYPE_DEFAULT    = 0;

  // ONNXTensorElementDataType
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED  = 0;
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT      = 1;
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8      = 2;
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8       = 3;
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16     = 4;
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16      = 5;
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32      = 6;
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64      = 7;
  ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING     = 8;
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL       = 9;
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16    = 10;
  ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE     = 11;
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32     = 12;
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64     = 13;
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64  = 14;
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 = 15;
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16   = 16;

  { Indices dos slots dentro de struct OrtApi (onnxruntime_c_api.h 1.28.1). }
  ORT_SLOT_GET_ERROR_MESSAGE          = 2;
  ORT_SLOT_CREATE_ENV                 = 3;
  ORT_SLOT_CREATE_SESSION             = 7;
  ORT_SLOT_RUN                        = 9;
  ORT_SLOT_CREATE_SESSION_OPTIONS     = 10;
  ORT_SLOT_SET_EXECUTION_MODE         = 13;
  ORT_SLOT_SET_GRAPH_OPT_LEVEL        = 23;
  ORT_SLOT_SET_INTRA_OP_THREADS       = 24;
  ORT_SLOT_SET_INTER_OP_THREADS       = 25;
  ORT_SLOT_SESSION_GET_INPUT_COUNT    = 30;
  ORT_SLOT_SESSION_GET_OUTPUT_COUNT   = 31;
  ORT_SLOT_SESSION_GET_INPUT_TYPE     = 33;
  ORT_SLOT_SESSION_GET_OUTPUT_TYPE    = 34;
  ORT_SLOT_SESSION_GET_INPUT_NAME     = 36;
  ORT_SLOT_SESSION_GET_OUTPUT_NAME    = 37;
  ORT_SLOT_CREATE_RUN_OPTIONS         = 39;
  ORT_SLOT_CREATE_TENSOR_WITH_DATA    = 49;
  ORT_SLOT_GET_TENSOR_MUTABLE_DATA    = 51;
  ORT_SLOT_CAST_TYPEINFO_TO_TENSOR    = 55;
  ORT_SLOT_GET_TENSOR_ELEMENT_TYPE    = 60;
  ORT_SLOT_GET_DIMENSIONS_COUNT       = 61;
  ORT_SLOT_GET_DIMENSIONS             = 62;
  ORT_SLOT_GET_TENSOR_ELEMENT_COUNT   = 64;
  ORT_SLOT_GET_TENSOR_TYPE_AND_SHAPE  = 65;
  ORT_SLOT_CREATE_CPU_MEMORY_INFO     = 69;
  ORT_SLOT_ALLOCATOR_FREE             = 76;
  ORT_SLOT_GET_DEFAULT_ALLOCATOR      = 78;
  ORT_SLOT_RELEASE_ENV                = 92;
  ORT_SLOT_RELEASE_STATUS             = 93;
  ORT_SLOT_RELEASE_MEMORY_INFO        = 94;
  ORT_SLOT_RELEASE_SESSION            = 95;
  ORT_SLOT_RELEASE_VALUE              = 96;
  ORT_SLOT_RELEASE_RUN_OPTIONS        = 97;
  ORT_SLOT_RELEASE_TYPE_INFO          = 98;
  ORT_SLOT_RELEASE_TENSOR_INFO        = 99;
  ORT_SLOT_RELEASE_SESSION_OPTIONS    = 100;
  ORT_SLOT_SESSION_GET_METADATA       = 111;
  ORT_SLOT_METADATA_GET_PRODUCER      = 112;
  ORT_SLOT_METADATA_GET_GRAPH_NAME    = 113;
  ORT_SLOT_METADATA_GET_DESCRIPTION   = 115;
  ORT_SLOT_METADATA_LOOKUP            = 116;
  ORT_SLOT_RELEASE_MODEL_METADATA     = 118;
  ORT_SLOT_METADATA_GET_KEYS          = 123;
  ORT_SLOT_GET_AVAILABLE_PROVIDERS    = 125;
  ORT_SLOT_RELEASE_AVAILABLE_PROVIDERS = 126;
  ORT_SLOT_ADD_SESSION_CONFIG_ENTRY   = 130;

type
  TOrtSize = NativeUInt;

  POrtEnv                    = Pointer;
  POrtStatus                 = Pointer;
  POrtSession                = Pointer;
  POrtSessionOptions         = Pointer;
  POrtRunOptions             = Pointer;
  POrtValue                  = Pointer;
  POrtMemoryInfo             = Pointer;
  POrtAllocator              = Pointer;
  POrtTypeInfo               = Pointer;
  POrtTensorTypeAndShapeInfo = Pointer;
  POrtModelMetadata          = Pointer;

  POrtValuePtr = ^POrtValue;
  POrtCharPtr  = ^PAnsiChar;

  { Assinaturas. Em Win64 stdcall mapeia para a convencao nativa x64. }
  TOrtGetApi           = function(Version: Cardinal): Pointer; stdcall;
  TOrtGetVersionString = function: PAnsiChar; stdcall;
  TOrtGetApiBase       = function: Pointer; stdcall;

  TOrtGetErrorMessage = function(Status: POrtStatus): PAnsiChar; stdcall;

  TOrtCreateEnv = function(LoggingLevel: Integer; LogId: PAnsiChar;
    out Env: POrtEnv): POrtStatus; stdcall;

  TOrtCreateSession = function(Env: POrtEnv; ModelPath: PWideChar;
    Options: POrtSessionOptions; out Session: POrtSession): POrtStatus; stdcall;

  TOrtRun = function(Session: POrtSession; RunOptions: POrtRunOptions;
    InputNames: POrtCharPtr; Inputs: POrtValuePtr; InputCount: TOrtSize;
    OutputNames: POrtCharPtr; OutputCount: TOrtSize;
    Outputs: POrtValuePtr): POrtStatus; stdcall;

  TOrtCreateSessionOptions = function(out Options: POrtSessionOptions): POrtStatus; stdcall;
  TOrtSetSessionInt = function(Options: POrtSessionOptions; Value: Integer): POrtStatus; stdcall;
  TOrtAddSessionConfigEntry = function(Options: POrtSessionOptions;
    Key: PAnsiChar; Value: PAnsiChar): POrtStatus; stdcall;

  TOrtSessionGetCount = function(Session: POrtSession; out Count: TOrtSize): POrtStatus; stdcall;
  TOrtSessionGetName = function(Session: POrtSession; Index: TOrtSize;
    Allocator: POrtAllocator; out Value: PAnsiChar): POrtStatus; stdcall;
  TOrtSessionGetTypeInfo = function(Session: POrtSession; Index: TOrtSize;
    out TypeInfo: POrtTypeInfo): POrtStatus; stdcall;

  TOrtCastTypeInfoToTensorInfo = function(TypeInfo: POrtTypeInfo;
    out TensorInfo: POrtTensorTypeAndShapeInfo): POrtStatus; stdcall;
  TOrtGetTensorElementType = function(Info: POrtTensorTypeAndShapeInfo;
    out DataType: Integer): POrtStatus; stdcall;
  TOrtGetDimensionsCount = function(Info: POrtTensorTypeAndShapeInfo;
    out Count: TOrtSize): POrtStatus; stdcall;
  TOrtGetDimensions = function(Info: POrtTensorTypeAndShapeInfo;
    Dimensions: PInt64; DimensionCount: TOrtSize): POrtStatus; stdcall;
  TOrtGetTensorShapeElementCount = function(Info: POrtTensorTypeAndShapeInfo;
    out Count: TOrtSize): POrtStatus; stdcall;
  TOrtGetTensorTypeAndShape = function(Value: POrtValue;
    out Info: POrtTensorTypeAndShapeInfo): POrtStatus; stdcall;
  TOrtGetTensorMutableData = function(Value: POrtValue; out Data: Pointer): POrtStatus; stdcall;

  TOrtCreateTensorWithDataAsOrtValue = function(Info: POrtMemoryInfo;
    Data: Pointer; DataLen: TOrtSize; Shape: PInt64; ShapeLen: TOrtSize;
    DataType: Integer; out Value: POrtValue): POrtStatus; stdcall;

  TOrtCreateCpuMemoryInfo = function(AllocatorType: Integer; MemType: Integer;
    out Info: POrtMemoryInfo): POrtStatus; stdcall;
  TOrtGetAllocatorWithDefaultOptions = function(out Allocator: POrtAllocator): POrtStatus; stdcall;
  TOrtAllocatorFree = function(Allocator: POrtAllocator; P: Pointer): POrtStatus; stdcall;

  TOrtCreateRunOptions = function(out Options: POrtRunOptions): POrtStatus; stdcall;

  TOrtSessionGetModelMetadata = function(Session: POrtSession;
    out Metadata: POrtModelMetadata): POrtStatus; stdcall;
  TOrtMetadataGetString = function(Metadata: POrtModelMetadata;
    Allocator: POrtAllocator; out Value: PAnsiChar): POrtStatus; stdcall;
  TOrtMetadataLookup = function(Metadata: POrtModelMetadata;
    Allocator: POrtAllocator; Key: PAnsiChar; out Value: PAnsiChar): POrtStatus; stdcall;
  TOrtMetadataGetKeys = function(Metadata: POrtModelMetadata;
    Allocator: POrtAllocator; out Keys: POrtCharPtr; out KeyCount: Int64): POrtStatus; stdcall;

  TOrtGetAvailableProviders = function(out Providers: POrtCharPtr;
    out ProviderCount: Integer): POrtStatus; stdcall;
  TOrtReleaseAvailableProviders = function(Providers: POrtCharPtr;
    ProviderCount: Integer): POrtStatus; stdcall;

  { Todas as funcoes ReleaseXxx do header tem a mesma forma: void(handle). }
  TOrtRelease = procedure(Handle: Pointer); stdcall;

  POrtApiSlots = ^TOrtApiSlots;
  TOrtApiSlots = packed record
    Items: array[0..ORT_API_SLOT_COUNT - 1] of Pointer;
  end;

  TOrtApiBase = packed record
    GetApi: TOrtGetApi;
    GetVersionString: TOrtGetVersionString;
  end;
  POrtApiBase = ^TOrtApiBase;

  { Tabela tipada. Resolvida uma unica vez no carregamento da DLL, o que
    elimina os casts espalhados pelo codigo e torna erros de assinatura
    visiveis em tempo de compilacao. }
  POrtApi = ^TOrtApi;
  TOrtApi = record
    GetErrorMessage: TOrtGetErrorMessage;
    CreateEnv: TOrtCreateEnv;
    CreateSession: TOrtCreateSession;
    Run: TOrtRun;
    CreateSessionOptions: TOrtCreateSessionOptions;
    SetExecutionMode: TOrtSetSessionInt;
    SetGraphOptimizationLevel: TOrtSetSessionInt;
    SetIntraOpNumThreads: TOrtSetSessionInt;
    SetInterOpNumThreads: TOrtSetSessionInt;
    AddSessionConfigEntry: TOrtAddSessionConfigEntry;
    SessionGetInputCount: TOrtSessionGetCount;
    SessionGetOutputCount: TOrtSessionGetCount;
    SessionGetInputTypeInfo: TOrtSessionGetTypeInfo;
    SessionGetOutputTypeInfo: TOrtSessionGetTypeInfo;
    SessionGetInputName: TOrtSessionGetName;
    SessionGetOutputName: TOrtSessionGetName;
    CreateRunOptions: TOrtCreateRunOptions;
    CreateTensorWithDataAsOrtValue: TOrtCreateTensorWithDataAsOrtValue;
    GetTensorMutableData: TOrtGetTensorMutableData;
    CastTypeInfoToTensorInfo: TOrtCastTypeInfoToTensorInfo;
    GetTensorElementType: TOrtGetTensorElementType;
    GetDimensionsCount: TOrtGetDimensionsCount;
    GetDimensions: TOrtGetDimensions;
    GetTensorShapeElementCount: TOrtGetTensorShapeElementCount;
    GetTensorTypeAndShape: TOrtGetTensorTypeAndShape;
    CreateCpuMemoryInfo: TOrtCreateCpuMemoryInfo;
    GetAllocatorWithDefaultOptions: TOrtGetAllocatorWithDefaultOptions;
    AllocatorFree: TOrtAllocatorFree;
    SessionGetModelMetadata: TOrtSessionGetModelMetadata;
    MetadataGetProducerName: TOrtMetadataGetString;
    MetadataGetGraphName: TOrtMetadataGetString;
    MetadataGetDescription: TOrtMetadataGetString;
    MetadataLookup: TOrtMetadataLookup;
    MetadataGetKeys: TOrtMetadataGetKeys;
    GetAvailableProviders: TOrtGetAvailableProviders;
    ReleaseAvailableProviders: TOrtReleaseAvailableProviders;

    ReleaseEnv: TOrtRelease;
    ReleaseStatus: TOrtRelease;
    ReleaseMemoryInfo: TOrtRelease;
    ReleaseSession: TOrtRelease;
    ReleaseValue: TOrtRelease;
    ReleaseRunOptions: TOrtRelease;
    ReleaseTypeInfo: TOrtRelease;
    ReleaseTensorTypeAndShapeInfo: TOrtRelease;
    ReleaseSessionOptions: TOrtRelease;
    ReleaseModelMetadata: TOrtRelease;
  end;

{ Preenche a tabela tipada a partir do array cru de slots devolvido por
  OrtApiBase.GetApi(ORT_API_VERSION). }
procedure BindOrtApi(Slots: POrtApiSlots; var Api: TOrtApi);

function OrtElementTypeName(ElementType: Integer): string;
function OrtElementSize(ElementType: Integer): Integer;

implementation

uses
  System.SysUtils;

procedure BindOrtApi(Slots: POrtApiSlots; var Api: TOrtApi);

  function S(Index: Integer): Pointer;
  begin
    Result := Slots^.Items[Index];
  end;

begin
  Api.GetErrorMessage                := TOrtGetErrorMessage(S(ORT_SLOT_GET_ERROR_MESSAGE));
  Api.CreateEnv                      := TOrtCreateEnv(S(ORT_SLOT_CREATE_ENV));
  Api.CreateSession                  := TOrtCreateSession(S(ORT_SLOT_CREATE_SESSION));
  Api.Run                            := TOrtRun(S(ORT_SLOT_RUN));
  Api.CreateSessionOptions           := TOrtCreateSessionOptions(S(ORT_SLOT_CREATE_SESSION_OPTIONS));
  Api.SetExecutionMode               := TOrtSetSessionInt(S(ORT_SLOT_SET_EXECUTION_MODE));
  Api.SetGraphOptimizationLevel      := TOrtSetSessionInt(S(ORT_SLOT_SET_GRAPH_OPT_LEVEL));
  Api.SetIntraOpNumThreads           := TOrtSetSessionInt(S(ORT_SLOT_SET_INTRA_OP_THREADS));
  Api.SetInterOpNumThreads           := TOrtSetSessionInt(S(ORT_SLOT_SET_INTER_OP_THREADS));
  Api.AddSessionConfigEntry          := TOrtAddSessionConfigEntry(S(ORT_SLOT_ADD_SESSION_CONFIG_ENTRY));
  Api.SessionGetInputCount           := TOrtSessionGetCount(S(ORT_SLOT_SESSION_GET_INPUT_COUNT));
  Api.SessionGetOutputCount          := TOrtSessionGetCount(S(ORT_SLOT_SESSION_GET_OUTPUT_COUNT));
  Api.SessionGetInputTypeInfo        := TOrtSessionGetTypeInfo(S(ORT_SLOT_SESSION_GET_INPUT_TYPE));
  Api.SessionGetOutputTypeInfo       := TOrtSessionGetTypeInfo(S(ORT_SLOT_SESSION_GET_OUTPUT_TYPE));
  Api.SessionGetInputName            := TOrtSessionGetName(S(ORT_SLOT_SESSION_GET_INPUT_NAME));
  Api.SessionGetOutputName           := TOrtSessionGetName(S(ORT_SLOT_SESSION_GET_OUTPUT_NAME));
  Api.CreateRunOptions               := TOrtCreateRunOptions(S(ORT_SLOT_CREATE_RUN_OPTIONS));
  Api.CreateTensorWithDataAsOrtValue := TOrtCreateTensorWithDataAsOrtValue(S(ORT_SLOT_CREATE_TENSOR_WITH_DATA));
  Api.GetTensorMutableData           := TOrtGetTensorMutableData(S(ORT_SLOT_GET_TENSOR_MUTABLE_DATA));
  Api.CastTypeInfoToTensorInfo       := TOrtCastTypeInfoToTensorInfo(S(ORT_SLOT_CAST_TYPEINFO_TO_TENSOR));
  Api.GetTensorElementType           := TOrtGetTensorElementType(S(ORT_SLOT_GET_TENSOR_ELEMENT_TYPE));
  Api.GetDimensionsCount             := TOrtGetDimensionsCount(S(ORT_SLOT_GET_DIMENSIONS_COUNT));
  Api.GetDimensions                  := TOrtGetDimensions(S(ORT_SLOT_GET_DIMENSIONS));
  Api.GetTensorShapeElementCount     := TOrtGetTensorShapeElementCount(S(ORT_SLOT_GET_TENSOR_ELEMENT_COUNT));
  Api.GetTensorTypeAndShape          := TOrtGetTensorTypeAndShape(S(ORT_SLOT_GET_TENSOR_TYPE_AND_SHAPE));
  Api.CreateCpuMemoryInfo            := TOrtCreateCpuMemoryInfo(S(ORT_SLOT_CREATE_CPU_MEMORY_INFO));
  Api.GetAllocatorWithDefaultOptions := TOrtGetAllocatorWithDefaultOptions(S(ORT_SLOT_GET_DEFAULT_ALLOCATOR));
  Api.AllocatorFree                  := TOrtAllocatorFree(S(ORT_SLOT_ALLOCATOR_FREE));
  Api.SessionGetModelMetadata        := TOrtSessionGetModelMetadata(S(ORT_SLOT_SESSION_GET_METADATA));
  Api.MetadataGetProducerName        := TOrtMetadataGetString(S(ORT_SLOT_METADATA_GET_PRODUCER));
  Api.MetadataGetGraphName           := TOrtMetadataGetString(S(ORT_SLOT_METADATA_GET_GRAPH_NAME));
  Api.MetadataGetDescription         := TOrtMetadataGetString(S(ORT_SLOT_METADATA_GET_DESCRIPTION));
  Api.MetadataLookup                 := TOrtMetadataLookup(S(ORT_SLOT_METADATA_LOOKUP));
  Api.MetadataGetKeys                := TOrtMetadataGetKeys(S(ORT_SLOT_METADATA_GET_KEYS));
  Api.GetAvailableProviders          := TOrtGetAvailableProviders(S(ORT_SLOT_GET_AVAILABLE_PROVIDERS));
  Api.ReleaseAvailableProviders      := TOrtReleaseAvailableProviders(S(ORT_SLOT_RELEASE_AVAILABLE_PROVIDERS));

  Api.ReleaseEnv                     := TOrtRelease(S(ORT_SLOT_RELEASE_ENV));
  Api.ReleaseStatus                  := TOrtRelease(S(ORT_SLOT_RELEASE_STATUS));
  Api.ReleaseMemoryInfo              := TOrtRelease(S(ORT_SLOT_RELEASE_MEMORY_INFO));
  Api.ReleaseSession                 := TOrtRelease(S(ORT_SLOT_RELEASE_SESSION));
  Api.ReleaseValue                   := TOrtRelease(S(ORT_SLOT_RELEASE_VALUE));
  Api.ReleaseRunOptions              := TOrtRelease(S(ORT_SLOT_RELEASE_RUN_OPTIONS));
  Api.ReleaseTypeInfo                := TOrtRelease(S(ORT_SLOT_RELEASE_TYPE_INFO));
  Api.ReleaseTensorTypeAndShapeInfo  := TOrtRelease(S(ORT_SLOT_RELEASE_TENSOR_INFO));
  Api.ReleaseSessionOptions          := TOrtRelease(S(ORT_SLOT_RELEASE_SESSION_OPTIONS));
  Api.ReleaseModelMetadata           := TOrtRelease(S(ORT_SLOT_RELEASE_MODEL_METADATA));
end;

function OrtElementTypeName(ElementType: Integer): string;
begin
  case ElementType of
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:      Result := 'FLOAT32';
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:      Result := 'UINT8';
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:       Result := 'INT8';
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:     Result := 'UINT16';
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:      Result := 'INT16';
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:      Result := 'INT32';
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:      Result := 'INT64';
    ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:     Result := 'STRING';
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:       Result := 'BOOL';
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:    Result := 'FLOAT16';
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:     Result := 'DOUBLE';
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:     Result := 'UINT32';
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:     Result := 'UINT64';
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:   Result := 'BFLOAT16';
  else
    Result := Format('TYPE_%d', [ElementType]);
  end;
end;

function OrtElementSize(ElementType: Integer): Integer;
begin
  case ElementType of
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:       Result := 1;

    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:   Result := 2;

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:     Result := 4;

    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:     Result := 8;
  else
    Result := 0;
  end;
end;

end.
