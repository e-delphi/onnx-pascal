unit ONNX.Types;

{
  Abstracoes da camada ONNX.

  Este unit existe para quebrar a dependencia circular entre runtime e sessao
  e para que as camadas de cima (visao) dependam de interfaces, nunca das
  classes concretas (Dependency Inversion).
}

interface

uses
  System.SysUtils,
  ONNX.CApi;

type
  EONNXError = class(Exception);

  { Descricao estatica de um input/output declarado pelo modelo. }
  TTensorInfo = record
    Name: string;
    ElementType: Integer;
    Shape: TArray<Int64>;
    function Rank: Integer;
    function Dim(Index: Integer): Int64;
    function HasDynamicDim: Boolean;
    function ShapeText: string;
    function ToString: string;
  end;

  { Tensor float com nome e shape. Toda a E/S da sessao passa por aqui:
    tipos inteiros/half do modelo sao convertidos para Single na leitura. }
  TTensor = record
    Name: string;
    Shape: TArray<Int64>;
    Data: TArray<Single>;
    class function Create(const AName: string; const AShape: TArray<Int64>;
      const AData: TArray<Single>): TTensor; static;
    function Rank: Integer;
    function Dim(Index: Integer): Int64;
    function DimAsInt(Index: Integer): Integer;
    function ElementCount: Integer;
    function ShapeText: string;
  end;

  TTensorArray = TArray<TTensor>;

  TSessionConfig = record
    GraphOptimizationLevel: Integer;
    ExecutionMode: Integer;
    IntraOpThreads: Integer;
    InterOpThreads: Integer;
    class function Default: TSessionConfig; static;
  end;

  IONNXSession = interface
    ['{2C0A6E31-4B77-4B0E-9D0A-0D9B6F1D5A11}']
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

  IONNXRuntime = interface
    ['{6F5D9A44-3E1C-4C58-8F2B-7A6E4C93B022}']
    function Version: string;
    function AvailableProviders: TArray<string>;
    function CreateSession(const ModelPath: string): IONNXSession; overload;
    function CreateSession(const ModelPath: string;
      const Config: TSessionConfig): IONNXSession; overload;
  end;

  { Servicos internos que a sessao precisa do runtime. Nao faz parte da API
    publica: mantido separado de IONNXRuntime por Interface Segregation. }
  IOrtCore = interface
    ['{9B3F2D18-77A5-42D6-B1E4-5C8D0F6A7E33}']
    function Api: POrtApi;
    function Env: POrtEnv;
    function Allocator: POrtAllocator;
    procedure Check(Status: POrtStatus; const Operation: string);
    { Copia a string alocada pelo ORT e devolve a memoria ao alocador. }
    function ConsumeString(P: PAnsiChar): string;
  end;

{ Converte IEEE-754 half precision para Single. }
function HalfToSingle(Value: Word): Single;

implementation

{ TTensorInfo }

function TTensorInfo.Rank: Integer;
begin
  Result := Length(Shape);
end;

function TTensorInfo.Dim(Index: Integer): Int64;
begin
  if (Index >= 0) and (Index < Length(Shape)) then
    Result := Shape[Index]
  else
    Result := -1;
end;

function TTensorInfo.HasDynamicDim: Boolean;
var
  I: Integer;
begin
  for I := 0 to High(Shape) do
    if Shape[I] < 0 then
      Exit(True);
  Result := False;
end;

function TTensorInfo.ShapeText: string;
var
  I: Integer;
begin
  Result := '[';
  for I := 0 to High(Shape) do
  begin
    if I > 0 then
      Result := Result + ', ';
    if Shape[I] < 0 then
      Result := Result + '?'
    else
      Result := Result + IntToStr(Shape[I]);
  end;
  Result := Result + ']';
end;

function TTensorInfo.ToString: string;
begin
  Result := Format('%s: %s %s', [Name, OrtElementTypeName(ElementType), ShapeText]);
end;

{ TTensor }

class function TTensor.Create(const AName: string; const AShape: TArray<Int64>;
  const AData: TArray<Single>): TTensor;
begin
  Result.Name := AName;
  Result.Shape := AShape;
  Result.Data := AData;
end;

function TTensor.Rank: Integer;
begin
  Result := Length(Shape);
end;

function TTensor.Dim(Index: Integer): Int64;
begin
  if (Index >= 0) and (Index < Length(Shape)) then
    Result := Shape[Index]
  else
    Result := 0;
end;

function TTensor.DimAsInt(Index: Integer): Integer;
begin
  Result := Integer(Dim(Index));
end;

function TTensor.ElementCount: Integer;
begin
  Result := Length(Data);
end;

function TTensor.ShapeText: string;
var
  I: Integer;
begin
  Result := '[';
  for I := 0 to High(Shape) do
  begin
    if I > 0 then
      Result := Result + ', ';
    Result := Result + IntToStr(Shape[I]);
  end;
  Result := Result + ']';
end;

{ TSessionConfig }

class function TSessionConfig.Default: TSessionConfig;
begin
  Result.GraphOptimizationLevel := ORT_ENABLE_ALL;
  Result.ExecutionMode := ORT_SEQUENTIAL;
  Result.IntraOpThreads := 0; // 0 = deixa o ORT decidir
  Result.InterOpThreads := 0;
end;

{ HalfToSingle }

function HalfToSingle(Value: Word): Single;
var
  Sign: Cardinal;
  Exponent: Integer;
  Mantissa: Cardinal;
  Bits: Cardinal;
begin
  Sign := (Value shr 15) and 1;
  Exponent := (Value shr 10) and $1F;
  Mantissa := Value and $3FF;

  if Exponent = 0 then
  begin
    if Mantissa = 0 then
      Bits := Sign shl 31
    else
    begin
      // Subnormal: normaliza deslocando ate o bit implicito aparecer.
      Exponent := -1;
      repeat
        Inc(Exponent);
        Mantissa := Mantissa shl 1;
      until (Mantissa and $400) <> 0;
      Mantissa := Mantissa and $3FF;
      Bits := (Sign shl 31) or (Cardinal(127 - 15 - Exponent) shl 23) or (Mantissa shl 13);
    end;
  end
  else if Exponent = $1F then
    Bits := (Sign shl 31) or (Cardinal($FF) shl 23) or (Mantissa shl 13)
  else
    Bits := (Sign shl 31) or (Cardinal(Exponent - 15 + 127) shl 23) or (Mantissa shl 13);

  Result := PSingle(@Bits)^;
end;

end.
