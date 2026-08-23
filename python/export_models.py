"""
Baixa os pesos oficiais da Ultralytics e exporta para ONNX.

Nao existe .onnx oficial de YOLO26: a Ultralytics distribui apenas .pt.
Exportar aqui mantem a procedencia rastreavel e os argumentos iguais para
todos os modelos.

Os pesos sao baixados pela propria ultralytics a partir do release oficial
github.com/ultralytics/assets. O sha256 de cada .pt e impresso para registro.
"""
import hashlib
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"
DEST = HERE.parent / "delphi" / "bin" / "yolo"

MODELS = [
    "yolo26x.pt",       # detect
    "yolo26x-seg.pt",   # segment
    "yolo26x-pose.pt",  # pose
    "yolo26x-obb.pt",   # obb
    "yolo26l-cls.pt",   # classify
]
OPSET = 12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(onnx_path: Path) -> str:
    import onnx
    model = onnx.load(str(onnx_path), load_external_data=False)

    def shape_of(vi):
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(str(d.dim_value) if d.dim_value else (d.dim_param or "?"))
        return f"{vi.name}[{', '.join(dims)}]"

    meta = {p.key: p.value for p in model.metadata_props}
    outs = " ".join(shape_of(o) for o in model.graph.output)
    return (
        f"    task={meta.get('task','?')}  imgsz={meta.get('imgsz','?')}  "
        f"end2end={meta.get('end2end','?')}  ultralytics={meta.get('version','?')}\n"
        f"    entrada : {shape_of(model.graph.input[0])}\n"
        f"    saidas  : {outs}"
    )


def main() -> int:
    WEIGHTS.mkdir(exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    # ultralytics baixa os pesos no diretorio corrente
    os.chdir(WEIGHTS)

    from ultralytics import YOLO

    results = []
    for name in MODELS:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        pt = WEIGHTS / name

        model = YOLO(name)  # baixa do release oficial se faltar
        print(f"  sha256(.pt) = {sha256(pt)}", flush=True)

        exported = Path(model.export(format="onnx", opset=OPSET, simplify=True))
        final = DEST / exported.name
        shutil.copy2(exported, final)

        print(f"  -> {final}")
        print(describe(final), flush=True)
        results.append(final.name)

    print(f"\n{'=' * 70}\n{len(results)} modelos exportados para {DEST}\n{'=' * 70}")
    for r in results:
        print("  " + r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
