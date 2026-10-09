#!/usr/bin/env bash
# Bateria de regressao do ONNXDemo.
# Uso: ./run_tests.sh          (a partir da pasta delphi/)
set -u
cd "$(dirname "$0")/bin" || exit 1

EXE=./ONNXDemo.exe
PASS=0; FAIL=0

run() {
  local nome="$1"; shift
  local esperado="$1"; shift
  echo "─────────────────────────────────────────────────────────────"
  echo "▶ $nome"
  local saida
  saida=$("$EXE" "$@" --no-pause --quiet 2>&1)
  if echo "$saida" | grep -qE "$esperado"; then
    echo "  ✔ PASSOU"
    echo "$saida" | grep -E "^  #|Tempos|saidas|Output 0" | head -6 | sed 's/^/    /'
    PASS=$((PASS+1))
  else
    echo "  x FALHOU (esperava /$esperado/)"
    echo "$saida" | tail -6 | sed 's/^/    /'
    FAIL=$((FAIL+1))
  fi
}

echo "═════════════════════════════════════════════════════════════"
echo " Regressao ONNXDemo"
echo "═════════════════════════════════════════════════════════════"

run "detect   / bus.jpg"    "bus.*9[0-9],"      yolo/x/yolo26x.onnx      imagem/bus.jpg    --out=saida/t_detect.png
run "segment  / bus.jpg"    "mascara: [0-9]+x"  yolo/x/yolo26x-seg.onnx  imagem/bus.jpg    --out=saida/t_seg.png
run "pose     / zidane.jpg" "keypoints: [0-9]+" yolo/x/yolo26x-pose.onnx imagem/zidane.jpg --out=saida/t_pose.png
run "obb      / boats.jpg"  "orientada: centro" yolo/x/yolo26x-obb.onnx  imagem/boats.jpg  --out=saida/t_obb.png
run "classify / zidane.jpg" "suit"              yolo/l/yolo26l-cls.onnx  imagem/zidane.jpg --no-render
run "squeezenet / dog.jpg"  "Samoyed"           squeezenet/squeezenet1_1.onnx imagem/dog.jpg --labels=squeezenet/labels.txt --no-render
run "5-crop   / dog.jpg"    "Samoyed"           squeezenet/squeezenet1_1.onnx imagem/dog.jpg --labels=squeezenet/labels.txt --multi-crop --no-render
run "texto    / bus.jpg"    "8 linha\(s\) de texto" ocr/PP-OCRv6_medium_det.onnx imagem/bus.jpg --out=saida/t_texto.png
run "texto    / documento"  "98 linha\(s\) de texto" ocr/PP-OCRv6_medium_det.onnx imagem/ocr_exemplo.png --out=saida/t_documento.png
run "ocr      / bus.jpg"    "emisiones"          --ocr imagem/bus.jpg
run "ocr      / documento"  "Markov Entropy Decomposition" --ocr imagem/ocr_exemplo.png

# GPU (DirectML). Sem a DLL do Windows ML o programa avisa e roda na CPU,
# entao estes casos tambem passam numa maquina sem GPU.
run "gpu detect / bus.jpg"  "bus.*9[0-9],"      yolo/x/yolo26x.onnx      imagem/bus.jpg    --gpu --out=saida/t_gpu_detect.png
run "gpu ocr  / documento"  "Markov Entropy Decomposition" --ocr imagem/ocr_exemplo.png --gpu

run "face     / obama.jpg"  "score [0-9]+"       --query imagem/obama.jpg
echo "─────────────────────────────────────────────────────────────"
echo "  Erros esperados (devem falhar com mensagem limpa, sem crash):"
for args in "yolo/x/yolo26x.onnx imagem/naoexiste.jpg" "yolo/x/nada.onnx imagem/bus.jpg" "imagem/bus.jpg" "yolo/x/yolo26x.onnx imagem/bus.jpg --conf=abc"; do
  msg=$($EXE $args --no-pause 2>&1 | grep -E "^ERRO" | head -1)
  if [ -n "$msg" ]; then echo "    ✔ $msg"; PASS=$((PASS+1)); else echo "    ✗ sem erro para: $args"; FAIL=$((FAIL+1)); fi
done

echo "═════════════════════════════════════════════════════════════"
echo "  $PASS passaram, $FAIL falharam"
echo "═════════════════════════════════════════════════════════════"
[ "$FAIL" -eq 0 ]
