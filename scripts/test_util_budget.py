import subprocess, sys, os
S = os.path.join(os.path.dirname(__file__), "util_budget.py")
def _run(*a): return subprocess.run([sys.executable, S, *a], capture_output=True, text=True)

def test_disk_fits():      r = _run("--mode", "disk", "--free-gib", "584", "--bits", "2.5"); assert r.returncode == 0 and "FITS" in r.stdout
def test_disk_tight():     r = _run("--mode", "disk", "--free-gib", "430", "--bits", "2.5"); assert r.returncode == 1 and "STOP" in r.stdout
def test_vram_reports_comfort(): r = _run("--mode", "vram", "--bits", "2.5", "--cq", "3"); assert r.returncode == 0 and "comfort" in r.stdout
def test_ram_ok():               r = _run("--mode", "ram", "--mcs", "321", "--bits", "2.5"); assert r.returncode == 0
