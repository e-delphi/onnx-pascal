unit Vision.Decoder.Ctc;

{
  Leitura de uma linha de texto ja recortada: decodificacao CTC
  (Connectionist Temporal Classification), a cabeca dos reconhecedores
  PaddleOCR - PP-OCRv6_medium_rec, small, tiny e os v5.

  A rede emite [1, T, C]: para cada uma das T colunas da imagem (largura/8),
  uma distribuicao ja normalizada (softmax) sobre C classes, onde

    classe 0         = "branco" (nenhum caractere nesta coluna)
    classes 1..C-2   = o dicionario, na ordem do character_dict
    classe C-1       = espaco (use_space_char do PaddleOCR)

  A decodificacao gulosa pega a classe mais provavel de cada coluna, junta
  repeticoes consecutivas e descarta o branco - "hh_e_ll_ll_o" vira "hello".
  O score e a media das probabilidades das colunas que viraram caractere,
  exatamente como o CTCLabelDecode do PaddleX.

  O dicionario chega como Spec.ClassNames (via --labels ou o inference.yml
  ao lado do modelo), entao o decoder nao conhece arquivo nenhum.
}

interface

uses
  System.SysUtils,
  System.Math,
  ONNX.Types,
  Vision.Types,
  Vision.Model,
  Vision.Decoder;

type
  TCtcDecoder = class(TInterfacedObject, IResultDecoder)
  private
    function FindSequenceTensor(const Outputs: TTensorArray): TTensor;
  public
    function Task: TVisionTask;
    function Describe: string;
    function Decode(const Outputs: TTensorArray;
      const Context: TDecodeContext): TVisionResult;
  end;

{ Decodifica o item Index de uma saida CTC [N, Steps, Classes] (Data
  achatado). Classes tem que ser Length(Dictionary) + 2. Exposta para a
  leitura em lote do Vision.Ocr. }
procedure CtcGreedyDecode(const Data: TArray<Single>; Index, Steps,
  Classes: Integer; const Dictionary: TArray<string>; out Text: string;
  out Score: Single);

implementation

procedure CtcGreedyDecode(const Data: TArray<Single>; Index, Steps,
  Classes: Integer; const Dictionary: TArray<string>; out Text: string;
  out Score: Single);
var
  Base, T, C, Best, Previous, Kept: Integer;
  BestProb, ProbSum: Double;
  Builder: TStringBuilder;
begin
  Base := Index * Steps * Classes;
  Builder := TStringBuilder.Create;
  try
    Previous := -1;
    Kept := 0;
    ProbSum := 0;

    for T := 0 to Steps - 1 do
    begin
      Best := 0;
      BestProb := Data[Base + T * Classes];
      for C := 1 to Classes - 1 do
        if Data[Base + T * Classes + C] > BestProb then
        begin
          Best := C;
          BestProb := Data[Base + T * Classes + C];
        end;

      if (Best <> 0) and (Best <> Previous) then
      begin
        if Best = Classes - 1 then
          Builder.Append(' ')
        else
          Builder.Append(Dictionary[Best - 1]);
        ProbSum := ProbSum + BestProb;
        Inc(Kept);
      end;
      Previous := Best;
    end;

    Text := Builder.ToString;
    if Kept > 0 then
      Score := ProbSum / Kept
    else
      Score := 0;
  finally
    Builder.Free;
  end;
end;

function TCtcDecoder.Task: TVisionTask;
begin
  Result := vtTextRec;
end;

function TCtcDecoder.Describe: string;
begin
  Result := 'CTC guloso: [1,T,C] -> texto (0 = branco, ultimo = espaco)';
end;

function TCtcDecoder.FindSequenceTensor(const Outputs: TTensorArray): TTensor;
var
  I: Integer;
begin
  for I := 0 to High(Outputs) do
    if Outputs[I].Rank = 3 then
    begin
      if Outputs[I].Dim(0) <> 1 then
        raise EDecodeError.CreateFmt(
          'Somente batch 1 e suportado; recebido shape %s',
          [Outputs[I].ShapeText]);
      Exit(Outputs[I]);
    end;

  raise EDecodeError.Create(
    'Nenhuma saida com formato de sequencia CTC [1, T, C] foi encontrada');
end;

function TCtcDecoder.Decode(const Outputs: TTensorArray;
  const Context: TDecodeContext): TVisionResult;
var
  Tensor: TTensor;
  Steps, Classes, Dictionary: Integer;
begin
  Result := Default(TVisionResult);
  Result.Task := vtTextRec;
  Result.ImageWidth := Context.Transform.SourceWidth;
  Result.ImageHeight := Context.Transform.SourceHeight;

  Tensor := FindSequenceTensor(Outputs);
  Steps := Tensor.DimAsInt(1);
  Classes := Tensor.DimAsInt(2);
  if (Steps <= 0) or (Classes <= 0) or
     (Length(Tensor.Data) < Steps * Classes) then
    raise EDecodeError.CreateFmt('Saida CTC invalida: %s', [Tensor.ShapeText]);

  // O tamanho do dicionario tem que fechar com a saida: branco + N + espaco.
  Dictionary := Context.Spec.ClassCount;
  if Dictionary + 2 <> Classes then
    raise EDecodeError.CreateFmt(
      'Dicionario com %d caracteres nao combina com a saida %s (esperado %d). ' +
      'Use o inference.yml do mesmo modelo em --labels.',
      [Dictionary, Tensor.ShapeText, Classes - 2]);

  CtcGreedyDecode(Tensor.Data, 0, Steps, Classes, Context.Spec.ClassNames,
    Result.Text, Result.TextScore);
end;

initialization
  TDecoderRegistry.RegisterDecoder(vtTextRec,
    function: IResultDecoder
    begin
      Result := TCtcDecoder.Create;
    end);

end.
