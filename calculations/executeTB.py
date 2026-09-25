"""
executeTB.py -- runs tight-binding HHG calculations from .xtoml / .toml input files.

Usage (from the folder of the input file; `runTB` is an optional bash alias of this script):
    python executeTB.py tilt_config.xtoml                    # generate runs/*.toml and run them in sequence
    python executeTB.py tilt_config.xtoml --generate-only    # only generate runs/*.toml
    python executeTB.py tilt_config.xtoml --background       # like nohup ... > tilt_config.out 2>&1 &
    python executeTB.py runs/graphene_tilt_01.toml      # run already generated .toml files

Extended TOML (.xtoml)
----------------------
A .xtoml file is valid TOML with two extensions:
  * Python expressions:  key = "python(<expression>)"  -> evaluated with numpy available as `np`
                         (nothing else: no builtins, no imports, no references to other keys).
  * Sweeps:              key = { sweep = [v0, v1, ...] }  or  { sweep = "python(...)" }
                         -> one calculation per value. Several sweeps must have the same length
                         and are run together (value i of each sweep in calculation i).
                         A plain list (e.g. [0, 0, 1]) is a VALUE, never a sweep.
Each point of the .xtoml becomes a plain .toml in runs/ (numbers only), named
<crystal.name>_<calculation.ID>[_NN].toml; the .log and the results .txt are written next to it.
Keys ending in 'filename' are paths relative to the folder of the input file; in the generated
.toml they are written as absolute paths, so every .toml runs from any folder.
"""
import argparse
import logging
import pprint
import re
import subprocess
import sys
import tomllib
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import tomli_w

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # repo/src, independent of cwd
import TBcrystal as cr
import grid
import TBevolution
import Field

RUNS_DIR = "runs"

grid_type = {
    'UniformCartesianGrid': grid.UniformCartesianGrid,
    'UniformCylindricalGrid': grid.UniformCylindricalGrid,
}
filter_type = {
    'polygonfilter': grid.polygonfilter,
}
evolution_type = {
    'TBevolution_CMCP': TBevolution.TBevolution_CMCP,
    'TBevolution_Bloch': TBevolution.TBevolution_Bloch,
}
field_type = {
    'PulsedField': Field.PulsedField,
}
field_env_type = {
    'env_sin2': Field.env_sin2,
    'env_lin': Field.env_lin,
}


# =====================================================================================
# 1. Extended TOML: python(...) expressions and sweeps
# =====================================================================================
PYTHON_EXPR = re.compile(r"\s*python\((.*)\)\s*", re.S)


def to_plain(value):
    """numpy arrays and scalars -> plain Python lists / numbers (writable as TOML)."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate(value):
    """Recursively replace every "python(...)" string by the value of its expression."""
    if isinstance(value, dict):
        return {k: evaluate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [evaluate(v) for v in value]
    if isinstance(value, str):
        m = PYTHON_EXPR.fullmatch(value)
        if m:
            try:
                result = eval(m.group(1), {"__builtins__": {}, "np": np})
            except Exception as err:
                raise ValueError(f"cannot evaluate {value!r}: {err}") from err
            return to_plain(result)
    return value


def is_sweep(value):
    return isinstance(value, dict) and set(value) == {'sweep'}


def find_sweeps(config, prefix=()):
    """{key path (tuple): list of values} for every { sweep = ... } in the (evaluated) config."""
    sweeps = {}
    for k, v in config.items():
        path = prefix + (k,)
        if is_sweep(v):
            values = v['sweep']
            if not isinstance(values, list) or len(values) == 0:
                raise ValueError(f"{'.'.join(path)}: 'sweep' must be a non-empty list, got {values!r}")
            sweeps[path] = values
        elif isinstance(v, dict):
            sweeps.update(find_sweeps(v, path))
    return sweeps


def set_path(config, path, value):
    d = config
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


def replace_sweeps(config, sweeps, i):
    """Copy of config with every sweep replaced by its i-th value."""
    out = {k: (replace_sweeps(v, {}, i) if isinstance(v, dict) and not is_sweep(v) else v)
           for k, v in config.items()}                              # deep copy of the dicts
    for path, values in sweeps.items():
        set_path(out, path, values[i])
    return out


def absolute_paths(config, base_dir, prefix=''):
    """Keys ending in 'filename' are relative to base_dir (absolute and '~/...' paths also valid)
    -> make them absolute.
    The file must exist: a wrong path is reported here, before any calculation starts."""
    out = {}
    for k, v in config.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out[k] = absolute_paths(v, base_dir, key + '.')
        elif k.endswith('filename') and isinstance(v, str):
            p = (base_dir / Path(v).expanduser()).resolve()   # '~' allowed; absolute paths kept
            if not p.is_file():
                raise FileNotFoundError(f"{key} = {v!r} -> {p} does not exist "
                                        f"(relative paths are relative to {base_dir})")
            out[k] = str(p)
        else:
            out[k] = v
    return out


def expand_xtoml(path):
    """Read a .xtoml (or .toml) and return [(name, config, swept values), ...], one per calculation."""
    path = Path(path).resolve()
    with open(path, 'rb') as f:
        raw = tomllib.load(f)

    config = absolute_paths(evaluate(raw), path.parent)
    sweeps = find_sweeps(config)

    ID = config['calculation']['ID']
    if not isinstance(ID, str):
        raise ValueError("calculation.ID must be a plain string (the sweep index is appended automatically)")

    lengths = {len(v) for v in sweeps.values()}
    if len(lengths) > 1:
        detail = ', '.join(f"{'.'.join(p)}={len(v)}" for p, v in sweeps.items())
        raise ValueError(f"sweeps with different lengths: {detail}")
    n = lengths.pop() if lengths else 1

    width = max(2, len(str(n - 1)))
    runs = []
    for i in range(n):
        cfg = replace_sweeps(config, sweeps, i)
        if sweeps:
            cfg['calculation']['ID'] = f"{ID}_{i:0{width}d}"
        name = f"{cfg['crystal']['name']}_{cfg['calculation']['ID']}"
        swept = {'.'.join(p): v[i] for p, v in sweeps.items()}
        runs.append((name, cfg, swept))
    return runs


def calc_prefix(runs):
    """<crystal.name>_<ID> without the sweep index: common part of the names of all calculations."""
    name, _, swept = runs[0]
    return name.rsplit('_', 1)[0] if swept else name


def old_results(runs_dir, prefix):
    """Names of calculations with this prefix that already have a .log in runs_dir
    (prefix.log or prefix_NN.log): every calculation writes its .log as soon as it starts."""
    pattern = re.compile(re.escape(prefix) + r"(_\d+)?\.log")
    return sorted(p.stem for p in Path(runs_dir).glob("*.log") if pattern.fullmatch(p.name))


def remove_results(runs_dir, names, calc_type):
    """Delete the .toml, .log and _<calc_type>.txt of the given calculations."""
    for name in names:
        for suffix in (".toml", ".log", f"_{calc_type}.txt"):
            (Path(runs_dir) / f"{name}{suffix}").unlink(missing_ok=True)


def write_runs(xtoml_path, runs):
    """Write runs/<name>.toml for every calculation (+ a sweep summary). Returns the .toml paths."""
    xtoml_path = Path(xtoml_path).resolve()
    runs_dir = xtoml_path.parent / RUNS_DIR
    runs_dir.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    toml_paths = []
    for i, (name, cfg, swept) in enumerate(runs):
        header = f"# GENERATED from {xtoml_path.name} on {now} -- do not edit (edit the .xtoml and regenerate)\n"
        if swept:
            header += f"# sweep point {i} of {len(runs)}: " + \
                      ", ".join(f"{k} = {v}" for k, v in swept.items()) + "\n"
        p = runs_dir / f"{name}.toml"
        p.write_text(header + "\n" + tomli_w.dumps(cfg))
        toml_paths.append(p)

    sweep_file = runs_dir / f"{calc_prefix(runs)}_sweep.txt"
    if runs[0][2]:                                                  # sweep summary
        keys = list(runs[0][2])
        # tab-separated table with a real header row (column titles = swept keys):
        #   pandas.read_csv(path, sep='\t', comment='#')
        lines = [f"# sweep of {xtoml_path.name} ({now})",
                 "\t".join(["index", "name"] + keys)]
        lines += ["\t".join([str(i), name] + [str(swept[k]) for k in keys])
                  for i, (name, _, swept) in enumerate(runs)]
        sweep_file.write_text("\n".join(lines) + "\n")
    else:
        sweep_file.unlink(missing_ok=True)                          # stale table of an older sweep
    return toml_paths


# =====================================================================================
# 2. One calculation from a plain .toml
# =====================================================================================
def run(config, name, out_dir):
    """Run ONE calculation described by a plain (already evaluated) config dict.
    Writes <out_dir>/<name>_<calculation type>.txt. Logging goes wherever the caller configured it
    (main() sends it to <out_dir>/<name>.log)."""
    out_dir = Path(out_dir)
    msg = f"Tight-binding calculation started at {datetime.now()}"
    logging.info(msg)
    logging.info("=" * len(msg))
    logging.info('\n ' + pprint.pformat(config, indent=2))

    # ---- k grid
    logging.info('Constructing the BZ')
    ndim = config['crystal']['dim']
    bz = config['BZ']
    if ndim == 1:
        limits = [bz['kx_min'], bz['kx_max']]
        nk = [bz['nkx']]
    elif ndim == 2:
        limits = [[bz['kx_min'], bz['kx_max']], [bz['ky_min'], bz['ky_max']]]
        nk = [bz['nkx'], bz['nky']]
    else:
        limits = [[bz['kx_min'], bz['kx_max']], [bz['ky_min'], bz['ky_max']], [bz['kz_min'], bz['kz_max']]]
        nk = [bz['nkx'], bz['nky'], bz['nkz']]

    if 'filter' in bz and bz['filter']['type'] in filter_type:
        k_filter = filter_type[bz['filter']['type']]
        filter_args = bz['filter']['args']
    else:
        k_filter, filter_args = None, {}

    limits_list = limits if ndim > 1 else [limits]
    dx = [(lim[1] - lim[0])/nk[i] for i, lim in enumerate(limits_list)]
    origin = tuple(d/2 for d in dx)                                 # k nodes symmetric about 0
    gr = grid.UniformCartesianGrid(ndim, limits, nk, origin=origin, filter=k_filter, filter_args=filter_args)
    logging.info(f'Number of points in the original Grid: {gr.npts}')
    logging.info(f'Number of points in the filtered Grid: {gr.inGrid.sum()}')

    # ---- crystal
    logging.info('Constructing the TB crystal')
    TBcr = cr.crystal.from_W90_TB_file(filename=config['crystal']['W90filename'], grid=gr,
                                       species_name=config['crystal']['name'],
                                       threshold_hopping=config['crystal']['threshold_hopping'])

    # ---- time grid (tini, tfin in optical periods)
    logging.info('Constructing the temporal grid')
    lambda0 = config['Field']['lambda']*1e-9
    T0 = Field.lambda_to_T(lambda0)
    tt = grid.UniformCartesianGrid(1, limits=[config['time']['tini']*T0, config['time']['tfin']*T0],
                                   nptx=config['time']['nt'])

    # ---- field
    logging.info('Constructing the Field')
    fc = config['Field']
    Efield = field_type[fc['type']](tt, I_W__cm2=fc['I_W__cm2'], lambda0_nm=lambda0*1e9,
                                    env=field_env_type[fc['env']['type']], env_parameters=fc['env']['args'],
                                    varphi_rad=fc['varphi'], chi_rad=fc['chi'], ellip=fc['ellip'],
                                    theta_rad=fc.get('theta', 0.0),
                                    s_direction_cartesian=np.array(fc.get('s_direction', [0, 0, 1]), dtype=float),
                                    field_type=fc.get('field_type', 'E'))

    # ---- evolution
    logging.info('Constructing the TB evolver')
    TBev = evolution_type[config['evolver']['type']](TBcr, Efield)

    calc = config['evolver']['calculation']
    if calc['type'] != 'dipole_velocity':
        raise ValueError(f"unknown evolver.calculation.type {calc['type']!r}")

    logging.info('Starting the calculation: dipole_velocity')
    t, vx, vy, vz = TBev.rk_dipole_velocity(npt=calc['args']['nt'])

    filename = out_dir / f"{name}_{calc['type']}.txt"
    logging.info(f'Calculation finished. Writing to {filename}')
    msg1 = f"# CALCULATION: {calc['type']}"
    msg2 = f"# DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    rule = '#' + '=' * max(len(msg1), len(msg2))
    commented = '\n'.join('# ' + line for line in pprint.pformat(config, indent=2).splitlines())
    with open(filename, "w") as f:
        f.write(f"{msg1}\n{msg2}\n{rule}\n{commented}\n{rule}\n")
        f.write("# t  vx  vy      (v = sum_k dV v_k, dV in 1/A^2, v in m/s)\n")
        for i_t, i_vx, i_vy in zip(t, vx.real, vy.real):
            f.write(f"{i_t:.16e} \t {i_vx:.16e} \t {i_vy:.16e} \n")
    logging.info('Done')
    return filename


# =====================================================================================
# 3. Command line
# =====================================================================================
def main():
    parser = argparse.ArgumentParser(description='Tight-binding HHG calculations from .xtoml/.toml files')
    parser.add_argument('inputs', nargs='+', help='.xtoml and/or .toml files (run in the given order)')
    parser.add_argument('--generate-only', action='store_true',
                        help='only generate runs/*.toml from the .xtoml files, do not calculate')
    parser.add_argument('--overwrite', action='store_true',
                        help='delete previous results with the same name before running '
                             '(without it, executeTB refuses to run if they exist)')
    parser.add_argument('--background', action='store_true',
                        help='run detached from the terminal (like nohup ... &); stdout and stderr go to '
                             '<first input without extension>.out, next to it')
    args = parser.parse_args()

    # 1) collect the .toml files to run (a .xtoml generates its runs/*.toml first)
    tomls = []
    for inp in args.inputs:
        p = Path(inp)
        if p.suffix == '.xtoml':
            try:
                runs = expand_xtoml(p)
            except (ValueError, FileNotFoundError, tomllib.TOMLDecodeError) as err:
                sys.exit(f"ERROR in {p}: {err}\nNothing was generated or run.")
            runs_dir = p.parent.resolve() / RUNS_DIR
            old = old_results(runs_dir, calc_prefix(runs))
            if old and not args.generate_only:
                if not args.overwrite:
                    sys.exit(f"ERROR: {runs_dir} already has results of {calc_prefix(runs)}: "
                             f"{', '.join(old)}\nUse --overwrite to delete them, or change calculation.ID. "
                             "Nothing was run.")
                remove_results(runs_dir, old, runs[0][1]['evolver']['calculation']['type'])
                print(f"{p}: deleted previous results of {', '.join(old)}")
            generated = write_runs(p, runs)
            print(f"{p}: {len(generated)} .toml generated in {p.parent.resolve() / RUNS_DIR}")
            tomls += generated
        elif p.suffix == '.toml':
            if p.with_suffix('.log').exists() and not (args.overwrite or args.generate_only):
                sys.exit(f"ERROR: {p} already has results ({p.with_suffix('.log').name})\n"
                         "Use --overwrite to overwrite them. Nothing was run.")
            tomls.append(p.resolve())
        else:
            parser.error(f"{inp}: expected a .xtoml or .toml file")
    if args.generate_only:
        return

    if args.background:
        # generation and checks are done above, in the foreground, so their errors show in the terminal.
        # Now relaunch this script on the generated .toml files, in a new session: it survives closing
        # the terminal (as with nohup) and all its output (prints, tracebacks, warnings) goes to .out
        out_path = Path(args.inputs[0]).with_suffix('.out')
        cmd = [sys.executable, str(Path(__file__).resolve())] + [str(t) for t in tomls] + \
              (['--overwrite'] if args.overwrite else [])
        with open(out_path, 'w') as out:
            proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        print(f"running in background, PID {proc.pid}; output in {out_path}")
        print(f"  follow it with:  tail -f {out_path}      stop it with:  kill {proc.pid}")
        return

    # 2) run them in sequence; a failure is logged and the next one is run
    failed = []
    for k, toml_path in enumerate(tomls, 1):
        name = toml_path.stem
        print(f"[{k}/{len(tomls)}] {name} ...", flush=True)
        # one .log per calculation, next to its .toml (force=True: reconfigure for every calculation)
        logging.basicConfig(filename=toml_path.parent / f"{name}.log", filemode="w", level=logging.INFO,
                            format="%(asctime)s %(levelname)s: %(message)s", force=True)
        try:
            with open(toml_path, 'rb') as f:
                config = tomllib.load(f)
            if find_sweeps(config):
                raise ValueError("a .toml must not contain sweeps: use a .xtoml")
            config = absolute_paths(config, toml_path.parent)      # no-op for generated .toml
            run(config, name, toml_path.parent)
        except Exception:
            logging.error(traceback.format_exc())
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            failed.append(name)

    print(f"\n{len(tomls) - len(failed)} of {len(tomls)} calculations finished correctly")
    if failed:
        print("FAILED: " + ", ".join(failed) + "  (see their .log)")
        sys.exit(1)


if __name__ == '__main__':
    main()
