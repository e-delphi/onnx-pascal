unit ONNX.Runtime;

{
  Dono do ciclo de vida da DLL, da tabela de API e do OrtEnv.

  Implementa duas interfaces distintas de proposito:
    IONNXRuntime - o que a aplicacao usa (versao, providers, criar sessao);
    IOrtCore     - o que a sessao precisa (api, env, allocator, check).

  A sessao guarda uma referencia a IOrtCore, entao o runtime nunca e destruido
  antes das sessoes que dependem dele.
}

interface

uses
  Winapi.Windows,
  System.SysUtils,
  ONNX.CApi,
  ONNX.Types;

type
  TONNXRuntime = class(TInterfacedObject, IONNXRuntime, IOrtCore)
  private
    FDllPath: string;
    FDll: HMODULE;
    FApiBase: POrtApiBase;
    FApi: TOrtApi;
    FEnv: POrtEnv;
    FAllocator: POrtAllocator;
    FLoggingLevel: Integer;
    procedure LoadLibraryAndApi;
  public
    constructor Create(const ADllPath: string = 'onnxruntime.dll';
      ALoggingLevel: Integer = ORT_LOGGING_LEVEL_WARNING);
    destructor Destroy; override;

    // IONNXRuntime
    function Version: string;
    function AvailableProviders: TArray<string>;
    function CreateSession(const ModelPath: string): IONNXSession; overload;
    function CreateSession(const ModelPath: string;
      const Config: TSessionConfig): IONNXSession; overload;

    // IOrtCore
    function Api: POrtApi;
    function Env: POrtEnv;
    function Allocator: POrtAllocator;
    procedure Check(Status: POrtStatus; const Operation: string);
    function ConsumeString(P: PAnsiChar): string;
  end;

implementation

uses
  ONNX.Session;

{ TONNXRuntime }

constructor TONNXRuntime.Create(const ADllPath: string; ALoggingLevel: Integer);
begin
  inherited Create;
  FDllPath := ADllPath;
  FLoggingLevel := ALoggingLevel;
  LoadLibraryAndApi;

  Check(FApi.CreateEnv(FLoggingLevel, 'DelphiONNX', FEnv), 'CreateEnv');
  Check(FApi.GetAllocatorWithDefaultOptions(FAllocator), 'GetAllocatorWithDefaultOptions');
end;

destructor TONNXRuntime.Destroy;
begin
  if (FEnv <> nil) and Assigned(FApi.ReleaseEnv) then
    FApi.ReleaseEnv(FEnv);

  FEnv := nil;
  FAllocator := nil;
  FApiBase := nil;
  FApi := Default(TOrtApi);

  if FDll <> 0 then
  begin
    FreeLibrary(FDll);
    FDll := 0;
  end;

  inherited;
end;

procedure TONNXRuntime.LoadLibraryAndApi;
var
  GetApiBase: TOrtGetApiBase;
  Slots: POrtApiSlots;
begin
  FDll := LoadLibrary(PChar(FDllPath));
  if FDll = 0 then
    raise EONNXError.CreateFmt(
      'Nao foi possivel carregar %s (erro Win32 %d). ' +
      'Copie onnxruntime.dll para a pasta do executavel.',
      [FDllPath, GetLastError]);

  GetApiBase := TOrtGetApiBase(GetProcAddress(FDll, 'OrtGetApiBase'));
  if not Assigned(GetApiBase) then
    raise EONNXError.Create('OrtGetApiBase nao encontrado em ' + FDllPath);

  FApiBase := POrtApiBase(GetApiBase());
  if FApiBase = nil then
    raise EONNXError.Create('OrtGetApiBase devolveu nil');

  Slots := POrtApiSlots(FApiBase^.GetApi(ORT_API_VERSION));
  if Slots = nil then
    raise EONNXError.CreateFmt(
      'A DLL nao suporta a ONNX Runtime C API versao %d (DLL: %s).',
      [ORT_API_VERSION, string(AnsiString(FApiBase^.GetVersionString))]);

  BindOrtApi(Slots, FApi);

  if not Assigned(FApi.CreateEnv) or not Assigned(FApi.Run) then
    raise EONNXError.Create('Tabela da OrtApi incompleta: DLL incompativel.');
end;

function TONNXRuntime.Api: POrtApi;
begin
  Result := @FApi;
end;

function TONNXRuntime.Env: POrtEnv;
begin
  Result := FEnv;
end;

function TONNXRuntime.Allocator: POrtAllocator;
begin
  Result := FAllocator;
end;

procedure TONNXRuntime.Check(Status: POrtStatus; const Operation: string);
var
  Msg: PAnsiChar;
  Text: string;
begin
  if Status = nil then
    Exit;

  Text := 'erro desconhecido';
  if Assigned(FApi.GetErrorMessage) then
  begin
    Msg := FApi.GetErrorMessage(Status);
    if Msg <> nil then
      Text := string(AnsiString(Msg));
  end;

  if Assigned(FApi.ReleaseStatus) then
    FApi.ReleaseStatus(Status);

  raise EONNXError.CreateFmt('%s falhou: %s', [Operation, Text]);
end;

function TONNXRuntime.ConsumeString(P: PAnsiChar): string;
begin
  if P = nil then
    Exit('');
  try
    Result := string(AnsiString(P));
  finally
    FApi.AllocatorFree(FAllocator, P);
  end;
end;

function TONNXRuntime.Version: string;
var
  P: PAnsiChar;
begin
  Result := '';
  if FApiBase = nil then
    Exit;
  P := FApiBase^.GetVersionString;
  if P <> nil then
    Result := string(AnsiString(P));
end;

function TONNXRuntime.AvailableProviders: TArray<string>;
var
  Providers: POrtCharPtr;
  Count: Integer;
  Cursor: POrtCharPtr;
  I: Integer;
begin
  Result := nil;
  Providers := nil;
  Count := 0;

  if not Assigned(FApi.GetAvailableProviders) then
    Exit;

  Check(FApi.GetAvailableProviders(Providers, Count), 'GetAvailableProviders');
  if (Providers = nil) or (Count <= 0) then
    Exit;
  try
    SetLength(Result, Count);
    Cursor := Providers;
    for I := 0 to Count - 1 do
    begin
      if Cursor^ <> nil then
        Result[I] := string(AnsiString(Cursor^));
      Inc(Cursor);
    end;
  finally
    FApi.ReleaseAvailableProviders(Providers, Count);
  end;
end;

function TONNXRuntime.CreateSession(const ModelPath: string): IONNXSession;
begin
  Result := CreateSession(ModelPath, TSessionConfig.Default);
end;

function TONNXRuntime.CreateSession(const ModelPath: string;
  const Config: TSessionConfig): IONNXSession;
begin
  Result := TONNXSession.Create(Self as IOrtCore, ModelPath, Config);
end;

end.
