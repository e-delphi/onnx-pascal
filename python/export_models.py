"""
Baixa os pesos oficiais da Ultralytics e exporta para ONNX.

Nao existe .onnx oficial de YOLO26: a Ultralytics distribui apenas .pt.
Exportar aqui mantem a procedencia rastreavel e os argumentos iguais para
todos os modelos.

Gera a matriz tamanho x tarefa, uma pasta por tamanho:

    delphi/bin/yolo/n/yolo26n.onnx, yolo26n-seg.onnx, ...
    delphi/bin/yolo/s/...
    delphi/bin/yolo/m/...
    delphi/bin/yolo/l/...
    delphi/bin/yolo/x/...

Os pesos sao baixados pela propria ultralytics a partir do release oficial
github.com/ultralytics/assets. O sha256 de cada .pt e impresso para registro.

Uso:
    python export_models.py                 # matriz completa (25 modelos)
    python export_models.py --sizes n,s      # so alguns tamanhos
    python export_models.py --tasks detect,pose
"""
import argparse
import hashlib
import os
import shutil
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"
DEST_ROOT = HERE.parent / "delphi" / "bin" / "yolo"

SIZES = ["n", "s", "m", "l", "x"]

# sufixo do nome do arquivo -> nome da tarefa
TASKS = {
    "": "detect",
    "-seg": "segment",
    "-pose": "pose",
    "-obb": "obb",
    "-cls": "classify",
}

OPSET = 12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(onnx_path: Path) -> tuple[str, str, str]:
    """Devolve (imgsz, entrada, saidas) lidos do proprio .onnx."""
    import onnx
    model = onnx.load(str(onnx_path), load_external_data=False)

    def shape_of(vi):
        dims = [str(d.dim_value) if d.dim_value else (d.dim_param or "?")
                for d in vi.type.tensor_type.shape.dim]
        return f"[{', '.join(dims)}]"

    meta = {p.key: p.value for p in model.metadata_props}
    inputs = shape_of(model.graph.input[0])
    outputs = " + ".join(shape_of(o) for o in model.graph.output)
    return meta.get("imgsz", "?"), inputs, outputs


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default=",".join(SIZES),
                    help="tamanhos separados por virgula (padrao: n,s,m,l,x)")
    ap.add_argument("--tasks", default=",".join(TASKS.values()),
                    help="tarefas separadas por virgula (padrao: todas)")
    ap.add_argument("--opset", type=int, default=OPSET)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
    suffixes = [sfx for sfx, name in TASKS.items() if name in wanted]

    unknown = wanted - set(TASKS.values())
    if unknown:
        print(f"Tarefa desconhecida: {', '.join(sorted(unknown))}", file=sys.stderr)
        print(f"Validas: {', '.join(TASKS.values())}", file=sys.stderr)
        return 2

    WEIGHTS.mkdir(exist_ok=True)
    os.chdir(WEIGHTS)  # a ultralytics baixa os pesos no diretorio corrente

    from ultralytics import YOLO

    total = len(sizes) * len(suffixes)
    done, failed, rows = 0, [], []

    for size in sizes:
        dest = DEST_ROOT / size
        dest.mkdir(parents=True, exist_ok=True)

        for suffix in suffixes:
            name = f"yolo26{size}{suffix}.pt"
            done += 1
            print(f"\n{'=' * 72}\n[{done}/{total}] {name}\n{'=' * 72}", flush=True)

            try:
                model = YOLO(name)
                digest = sha256(WEIGHTS / name)
                print(f"  sha256(.pt) = {digest}", flush=True)

                exported = Path(model.export(format="onnx", opset=args.opset,
                                             simplify=True))
                final = dest / exported.name
                shutil.copy2(exported, final)

                imgsz, inputs, outputs = describe(final)
                size_mb = final.stat().st_size / 1e6
                print(f"  -> {final}  ({size_mb:.1f} MB)")
                print(f"     imgsz={imgsz}  entrada={inputs}")
                print(f"     saidas={outputs}", flush=True)

                rows.append((size, TASKS[suffix], name, final.name, size_mb,
                             inputs, outputs, digest))
            except Exception as exc:
                failed.append((name, str(exc)))
                print(f"  FALHOU: {exc}", file=sys.stderr)
                traceback.print_exc(limit=1)

    print(f"\n{'=' * 72}")
    print(f"{len(rows)} de {total} modelos exportados para {DEST_ROOT}")
    print(f"{'=' * 72}\n")

    print(f"{'tam':<4} {'tarefa':<9} {'arquivo':<24} {'MB':>7}  entrada -> saidas")
    for size, task, ptname, fname, mb, inputs, outputs, _ in rows:
        print(f"{size:<4} {task:<9} {fname:<24} {mb:>7.1f}  {inputs} -> {outputs}")

    print("\nsha256 dos pesos:")
    for size, task, ptname, fname, mb, inputs, outputs, digest in rows:
        print(f"  {ptname:<20} {digest}")

    if failed:
        print(f"\n{len(failed)} falha(s):", file=sys.stderr)
        for name, err in failed:
            print(f"  {name}: {err}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
