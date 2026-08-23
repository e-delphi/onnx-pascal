"""
Exporta variantes com end2end=False.

Todo export oficial de YOLO26 sai NMS-free ([1,300,C]), entao o caminho de
decode cru + NMS do ONNXDemo nunca era exercitado. Estes modelos existem
exatamente para cobrir esse caminho:

    detect  -> [1, 4+nc, N]        -> ApplyNms
    segment -> [1, 4+nc+32, N]     -> ApplyNms
    pose    -> [1, 4+1+51, N]      -> ApplyNms
    obb     -> [1, 4+nc+1, N]      -> ApplyRotatedNms (intersecao de poligonos)

Usa os modelos nano: sao pequenos e rapidos, e o formato da saida nao depende
do tamanho do modelo.
"""
import hashlib, os, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"
DEST = HERE.parent / "delphi" / "bin" / "yolo" / "raw"
MODELS = ["yolo26n.pt", "yolo26n-seg.pt", "yolo26n-pose.pt", "yolo26n-obb.pt"]
OPSET = 12


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def describe(path: Path) -> str:
    import onnx
    m = onnx.load(str(path), load_external_data=False)

    def sh(vi):
        d = [str(x.dim_value) if x.dim_value else (x.dim_param or "?")
             for x in vi.type.tensor_type.shape.dim]
        return f"{vi.name}[{', '.join(d)}]"

    meta = {p.key: p.value for p in m.metadata_props}
    return (f"    end2end={meta.get('end2end','?')}  task={meta.get('task','?')}\n"
            f"    entrada : {sh(m.graph.input[0])}\n"
            f"    saidas  : {' '.join(sh(o) for o in m.graph.output)}")


def main() -> int:
    WEIGHTS.mkdir(exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    os.chdir(WEIGHTS)
    from ultralytics import YOLO

    for name in MODELS:
        print(f"\n{'='*70}\n{name}  (end2end=False)\n{'='*70}", flush=True)
        model = YOLO(name)
        print(f"  sha256(.pt) = {sha256(WEIGHTS / name)}", flush=True)
        out = Path(model.export(format="onnx", opset=OPSET, simplify=True, end2end=False))
        final = DEST / out.name
        shutil.copy2(out, final)
        print(f"  -> {final}")
        print(describe(final), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
