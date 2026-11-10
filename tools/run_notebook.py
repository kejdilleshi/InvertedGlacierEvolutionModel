"""Execute a notebook headlessly (for tmux). The live notebook is only read; the executed copy goes to OUT.

    python run_notebook.py NOTEBOOK OUT [--results DIR] [--iters N]
--results / --iters override RESULTS / N_ITERS in memory only (used for smoke tests)."""
import argparse, re, sys, time, traceback
import nbformat
from nbclient import NotebookClient

ap = argparse.ArgumentParser()
ap.add_argument("notebook"); ap.add_argument("out")
ap.add_argument("--results"); ap.add_argument("--iters", type=int)
a = ap.parse_args()

nb = nbformat.read(a.notebook, as_version=4)
setup = next(c for c in nb.cells if c.cell_type == "code" and "RESULTS = " in c.source)
if a.results:
    setup.source, n = re.subn(r'^RESULTS = .*$', f'RESULTS = "{a.results}"', setup.source, flags=re.M); assert n == 1
if a.iters:
    setup.source, n = re.subn(r'^N_ITERS = \d+', f'N_ITERS = {a.iters}', setup.source, flags=re.M); assert n == 1
for c in nb.cells:
    if c.cell_type == "code": c.outputs = []; c.execution_count = None
print("settings:", [l for l in setup.source.splitlines() if l.startswith(("RESULTS", "N_ITERS", "T_FLAT", "EPOCHS", "RUN_"))], flush=True)

t0 = time.time()
def started(cell, cell_index): print(f"[{(time.time() - t0) / 60:6.1f} min] start cell {cell_index}", flush=True)
def done(cell, cell_index, execute_reply):
    print(f"[{(time.time() - t0) / 60:6.1f} min] done  cell {cell_index}", flush=True)
    nbformat.write(nb, a.out)          # partial copy after every cell

client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": "notebooks"}},
                        on_cell_start=started, on_cell_executed=done)
try:
    client.execute()
    print(f"FINISHED OK in {(time.time() - t0) / 60:.1f} min", flush=True)
    code = 0
except Exception:
    traceback.print_exc(); print("FAILED", flush=True); code = 1
finally:
    nbformat.write(nb, a.out)
sys.exit(code)
