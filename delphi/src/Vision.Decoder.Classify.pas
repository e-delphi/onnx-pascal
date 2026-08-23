unit Vision.Decoder.Classify;

{
  Cabeca de classificacao: saida [1, nc] com um score por classe.

  Cobre tanto modelos que ja emitem probabilidades (YOLO-cls faz softmax no
  proprio grafo) quanto modelos que emitem logits crus (SqueezeNet, ResNet
  do ONNX Model Zoo). A deteccao e feita pelo proprio vetor.
}

interface

uses
  System.SysUtils,
  System.Math,
  System.Generics.Collections,
  ONNX.Types,
  Vision.Types,
  Vision.Model,
  Vision.Preprocess,
  Vision.Decoder;

type
  TClassifyDecoder = class(TInterfacedObject, IResultDecoder)
  public
    function Task: TVisionTask;
    function Describe: string;
    function Decode(const Outputs: TTensorArray;
      const Context: TDecodeContext): TVisionResult;
  end;

implementation

function TClassifyDecoder.Task: TVisionTask;
begin
  Result := vtClassify;
end;

function TClassifyDecoder.Describe: string;
begin
  Result := 'classificacao ([1, nc], softmax aplicado se a saida for logits)';
end;

function TClassifyDecoder.Decode(const Outputs: TTensorArray;
  const Context: TDecodeContext): TVisionResult;
var
  Scores: TArray<Single>;
  Used: TArray<Boolean>;
  I, K, BestIndex, Limit: Integer;
  BestScore: Single;
  Item: TClassScore;
  Items: TList<TClassScore>;
begin
  Result := Default(TVisionResult);
  Result.Task := vtClassify;
  Result.ImageWidth := Context.Transform.SourceWidth;
  Result.ImageHeight := Context.Transform.SourceHeight;

  if Length(Outputs) = 0 then
    raise EDecodeError.Create('O modelo nao devolveu nenhuma saida');

  Scores := Copy(Outputs[0].Data);
  if Length(Scores) = 0 then
    raise EDecodeError.Create('Saida de classificacao vazia');

  if not LooksLikeProbabilityVector(Scores) then
    SoftmaxInPlace(Scores);

  Limit := Context.TopK;
  if Limit <= 0 then
    Limit := 5;
  Limit := Min(Limit, Length(Scores));

  SetLength(Used, Length(Scores));
  Items := TList<TClassScore>.Create;
  try
    for K := 0 to Limit - 1 do
    begin
      BestIndex := -1;
      BestScore := -MaxSingle;

      for I := 0 to High(Scores) do
        if (not Used[I]) and (Scores[I] > BestScore) then
        begin
          BestScore := Scores[I];
          BestIndex := I;
        end;

      if BestIndex < 0 then
        Break;

      Used[BestIndex] := True;

      if BestScore < Context.ConfidenceThreshold then
        Break;

      Item.ClassId := BestIndex;
      Item.ClassName := Context.Spec.ClassNameOf(BestIndex);
      Item.Score := BestScore;
      Items.Add(Item);
    end;

    Result.Classes := Items.ToArray;
  finally
    Items.Free;
  end;
end;

initialization
  TDecoderRegistry.RegisterDecoder(vtClassify,
    function: IResultDecoder
    begin
      Result := TClassifyDecoder.Create;
    end);

end.
