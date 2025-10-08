#!/usr/bin/env python3

import argparse, os, shlex, subprocess, sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# ----- valid task slugs (no numbers allowed) -----
TASKS = [
    "dielectric", "jdft2d", "log_gvrh", "log_kvrh",
    "mp_e_form", "mp_gap", "perovskites", "phonons",
]

# ----- detect script dir and data dirs -----
SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR    = SCRIPTS_DIR / "benchmark_data"
LOG_DIR     = DATA_DIR / "logs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def parse_task_list(spec: str):
    """Parse comma-separated task slugs; reject numbers and unknowns."""
    spec = (spec or "").strip()
    if not spec:
        return TASKS[:]  # default: all
    out, seen = [], set()
    for tok in spec.split(","):
        s = tok.strip()
        if not s:
            continue
        if s.isdigit():
            raise ValueError(f"Numeric task '{s}' not allowed. Use slugs like 'mp_gap,phonons'.")
        if s not in TASKS:
            raise ValueError(f"Unknown task slug '{s}'. Valid: {', '.join(TASKS)}")
        if s not in seen:
            out.append(s); seen.add(s)
    return out

def parse_steps(spec: str, all_steps):
    if not spec:
        return list(all_steps)
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part: continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    seen, ordered = set(), []
    for s in out:
        if s in all_steps and s not in seen:
            ordered.append(s); seen.add(s)
    return ordered

def path_is_python(p: Path) -> bool:
    return p.is_file() and os.access(str(p), os.X_OK)

def resolve_python(
    explicit_python: Optional[str],
    env_root: Optional[str],
    conda_env_name: Optional[str],
    fallback: str = "python"
) -> str:
    """
    Resolution priority:
      1) explicit python path (--py-*)
         - if it's a directory, assume it's an env root and append /bin/python
      2) env root path (--*-env), we append /bin/python
      3) conda env name (--*-conda) -> 'conda run -n <name> python'
      4) fallback (usually 'python')
    """
    # 1) explicit python path or env root
    if explicit_python:
        p = Path(os.path.expanduser(explicit_python))
        if p.is_dir():
            py = p / "bin" / "python"
            return str(py)
        return str(p)

    # 2) env root → append bin/python
    if env_root:
        root = Path(os.path.expanduser(env_root))
        py = root / "bin" / "python"
        return str(py)

    # 3) conda env name
    if conda_env_name:
        return f"conda run -n {conda_env_name} python"

    # 4) fallback
    return fallback

def run_cmd(cmd, env=None, log_file: Optional[Path] = None, dry_run: bool = False):
    print(f"→ {cmd}")
    if dry_run:
        return 0
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"\n===== {datetime.now().isoformat()} =====\n$ {cmd}\n")
            lf.flush()
            proc = subprocess.Popen(
                cmd, shell=True, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
            proc.wait()
            lf.write(f"\n(exit {proc.returncode})\n")
            return proc.returncode
    return subprocess.call(cmd, shell=True, env=env)

def main():
    ap = argparse.ArgumentParser(
        description="Run matbench pipeline (slug-only tasks, portable data dir, selectable MLIP & MODEL)."
    )
    # selection
    ap.add_argument("--tasks", default="", help="Comma-separated slugs (no numbers). E.g. 'mp_gap,phonons'. Empty = all.")
    ap.add_argument("--steps", default="", help="Steps to run, e.g. '1-3,7'. Default: all.")
    ap.add_argument("--mlip",  default="orb2",  help="Featurizer backend slug. Default: orb2 (expects 3_featurize_<mlip>.py).")
    ap.add_argument("--model", default="modnet",help="ML model slug. Default: modnet (expects 5_train_<model>.py etc.).")

    # envs — three ways each: explicit python, env root, or conda name
    ap.add_argument("--py-matbench",   default=None, help="Absolute path to python for matbench/modnet env.")
    ap.add_argument("--matbench-env",  default=None, help="Path to conda env root for matbench/modnet, e.g. /home/.../envs/matbench_env")
    ap.add_argument("--matbench-conda",default=None, help="Conda env name for matbench/modnet (uses 'conda run -n <name> python').")

    ap.add_argument("--py-mlip",       default=None, help="Absolute path to python for MLIP env.")
    ap.add_argument("--mlip-env",      default=None, help="Path to conda env root for MLIP, e.g. /home/.../envs/orb2_env")
    ap.add_argument("--mlip-conda",    default=None, help="Conda env name for MLIP (uses 'conda run -n <name> python').")

    ap.add_argument("--py-mlmodel",       default=None, help="Absolute path to python for MLMODEL env.")
    ap.add_argument("--mlmodel-env",      default=None, help="Path to conda env root for MLMODEL, e.g. /home/.../envs/modnet_env")
    ap.add_argument("--mlmodel-conda",    default=None, help="Conda env name for MLMODEL (uses 'conda run -n <name> python').")

    # behavior
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args()

    # validate tasks
    try:
        task_slugs = parse_task_list(args.tasks)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(2)

    mlip  = args.mlip.strip()
    model = args.model.strip()

    # resolve python executables
    PY_MATBENCH = resolve_python(args.py_matbench, args.matbench_env, args.matbench_conda, fallback="python")
    PY_MLIP  = resolve_python(args.py_mlip, args.mlip_env, args.mlip_conda, fallback="python")
    PY_MLMODEL  = resolve_python(args.py_mlmodel, args.mlmodel_env, args.mlmodel_conda, fallback="python")

    # step map
    STEPS = {
        1: ("1_retrieve_data.py",              "matbench"),
        2: ("2_build_sc.py",                   "mlip"),
        3: (f"3_featurize_{mlip}.py",          "mlip"),
        4: ("4_construct_pkl.py",              "mlip"),
        5: (f"5_train_{model}.py",             "mlmodel"),
        6: (f"6_opt_hp_{model}.py",            "mlmodel"),
        7: (f"7_get_parity_data.py",           "mlmodel"),
    }

    steps = parse_steps(args.steps, STEPS.keys())

    # check scripts exist early
    missing = [fn for s,(fn,_) in STEPS.items() if s in steps and not (SCRIPTS_DIR / fn).exists()]
    if missing:
        print(f"[ERROR] Missing scripts in {SCRIPTS_DIR}: {missing}")
        sys.exit(1)

    print(f"[INFO] scripts dir : {SCRIPTS_DIR}")
    print(f"[INFO] data dir    : {DATA_DIR}")
    print(f"[INFO] logs dir    : {LOG_DIR}")
    print(f"[INFO] MLIP        : {mlip}")
    print(f"[INFO] MODEL       : {model}")
    print(f"[INFO] tasks       : {', '.join(task_slugs)}")
    print("[INFO] plan:")
    for s in steps:
        fn, lab = STEPS[s]
        py = {"matbench": PY_MATBENCH, "mlip": PY_MLIP, "mlmodel": PY_MLMODEL}[lab]
        print(f"  step {s}: {fn}  (env={lab} → {py})")

    # --- run ---
    for s in steps:
        fn, lab = STEPS[s]
        py = {"matbench": PY_MATBENCH, "mlip": PY_MLIP, "mlmodel": PY_MLMODEL}[lab]
        cmd = f'{py} {shlex.quote(str(SCRIPTS_DIR / fn))}' if py.startswith("conda run") \
            else f'{shlex.quote(py)} {shlex.quote(str(SCRIPTS_DIR / fn))}'

        child_env = os.environ.copy()
        child_env["BENCH_DATA_DIR"] = str(DATA_DIR)
        child_env["BENCH_TASKS"]    = ",".join(task_slugs)
        child_env["BENCH_MLIP"]     = mlip
        child_env["BENCH_MODEL"]    = model

        log = LOG_DIR / f"{Path(fn).stem}.log"
        print(f"\n[RUN] step {s} → {fn}")
        rc = run_cmd(cmd, env=child_env, log_file=log, dry_run=args.dry_run)
        if rc != 0:
            print(f"[ERROR] step {s} failed (exit {rc}). Log: {log}")
            if args.stop_on_error:
                sys.exit(rc)

    print("\n[OK] finished.")

if __name__ == "__main__":
    main()