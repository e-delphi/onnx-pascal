unit Vision.Decoder.Embed;

{
  Cabeca de embedding (ArcFace / w600k_r50 e afins).

  Saida [1, D] - tipicamente D = 512. O vetor cru nao e unitario; a
  comparacao por cosseno exige normalizacao L2, feita aqui para que ninguem
  mais precise lembrar disso.

  Nao ha inferencia de tarefa possivel a partir do shape: um [1, 512] e
  indistinguivel de um classificador de 512 classes. Quem monta o pipeline
  informa a tarefa explicitamente (--task=embed ou via TFaceEncoder).
}

interface

uses
  System.SysUtils,
  System.Math,
  ONNX.Types,
  Vision.Types,
  Vision.Model,
  Vision.Preprocess,
  Vision.Decoder;

type
  TEmbeddingDecoder = class(TInterfacedObject, IResultDecoder)
  public
    function Task: TVisionTask;
    function Describe: string;
    function Decode(const Outputs: TTensorArray;
      const Context: TDecodeContext): TVisionResult;
  end;

implementation

function TEmbeddingDecoder.Task: TVisionTask;
begin
  Result := vtEmbed;
end;

function TEmbeddingDecoder.Describe: string;
begin
  Result := 'embedding ([1, D] com normalizacao L2)';
end;

function TEmbeddingDecoder.Decode(const Outputs: TTensorArray;
  const Context: TDecodeContext): TVisionResult;
var
  Values: TArray<Single>;
  I: Integer;
  Sum: Double;
  Norm: Single;
begin
  Result := Default(TVisionResult);
  Result.Task := vtEmbed;
  Result.ImageWidth := Context.Transform.SourceWidth;
  Result.ImageHeight := Context.Transform.SourceHeight;

  if Length(Outputs) = 0 then
    raise EDecodeError.Create('O modelo nao devolveu nenhuma saida');

  Values := Copy(Outputs[0].Data);
  if Length(Values) = 0 then
    raise EDecodeError.Create('Saida de embedding vazia');

  Sum := 0;
  for I := 0 to High(Values) do
    Sum := Sum + Double(Values[I]) * Values[I];

  Norm := Sqrt(Sum);
  if Norm <= 0 then
    raise EDecodeError.Create(
      'Embedding com norma zero: o recorte provavelmente esta em branco');

  for I := 0 to High(Values) do
    Values[I] := Values[I] / Norm;

  Result.Embedding := Values;
end;

initialization
  TDecoderRegistry.RegisterDecoder(vtEmbed,
    function: IResultDecoder
    begin
      Result := TEmbeddingDecoder.Create;
    end);

end.
