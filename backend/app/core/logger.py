"""
Enterprise Terminal Logger Ã¢â‚¬â€ Nexsure AI Engine
Provides rich, colored, structured terminal output without external dependencies.
Uses ANSI escape codes with Windows-safe character fallbacks.
"""

import sys
from datetime import datetime
from typing import Any, Dict, Optional

# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Windows ANSI Support Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
try:
    import colorama
    colorama.init(autoreset=True)
    _ANSI_SUPPORTED = True
except ImportError:
    _ANSI_SUPPORTED = False

# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Windows encoding fix Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
# Force stdout to UTF-8 on Windows to avoid cp1252 UnicodeEncodeErrors
try:
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.platform == "win32" and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ ANSI Color Codes Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
class _C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    WHITE   = "\033[97m"
    CYAN    = "\033[96m"
    BLUE    = "\033[94m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    GRAY    = "\033[90m"


def _ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _c(color: str, text: str, bold: bool = False) -> str:
    prefix = (color + _C.BOLD) if bold else color
    return f"{prefix}{text}{_C.RESET}"


def _print(*args, **kwargs):
    """Safe print that handles encoding errors gracefully."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fallback: strip non-ASCII and print
        safe_args = [str(a).encode("ascii", errors="replace").decode("ascii") for a in args]
        print(*safe_args, **kwargs)


# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬ Public API Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

def banner():
    """Print the Nexsure AI Engine startup banner."""
    line = "=" * 52
    _print(f"\n{_c(_C.CYAN, line, bold=True)}")
    _print(_c(_C.CYAN, "  NEXSURE AI ENGINE", bold=True))
    _print(_c(_C.GRAY, f"  Autonomous ML Lifecycle v2.0  |  {_ts()}"))
    _print(f"{_c(_C.CYAN, line, bold=True)}\n")


def section(title: str):
    """Print a section divider with title."""
    _print(f"\n{_c(_C.BLUE, '>>', bold=True)} {_c(_C.WHITE, title, bold=True)}")
    _print(_c(_C.GRAY, "  " + "-" * 48))


def success(msg: str):
    """Print a success log line."""
    _print(f"  {_c(_C.GREEN, '[OK]')}  {_c(_C.WHITE, msg)}")


def info(msg: str, detail: Optional[str] = None):
    """Print an informational log line."""
    line = f"  {_c(_C.CYAN, '-->')}  {_c(_C.GRAY, msg)}"
    if detail:
        line += f" {_c(_C.WHITE, detail)}"
    _print(line)


def warn(msg: str):
    """Print a warning log line."""
    _print(f"  {_c(_C.YELLOW, '[WARN]')}  {_c(_C.YELLOW, msg)}")


def error(msg: str):
    """Print an error log line."""
    _print(f"  {_c(_C.RED, '[ERR]')}  {_c(_C.RED, msg)}", file=sys.stderr)


def step(label: str):
    """Print a training step indicator."""
    ts = _c(_C.GRAY, f"[{_ts()}]")
    _print(f"  {_c(_C.BLUE, '...')}  {ts} {_c(_C.WHITE, label)}")


def model_table(metrics: Dict[str, Dict[str, Any]]):
    """Print a formatted model comparison table."""
    col_w = [26, 10, 10, 10, 10]
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
    sep = "  " + "-" * 68

    _print(f"\n{_c(_C.CYAN, '  MODEL EVALUATION RESULTS', bold=True)}")
    _print(sep)

    hdr = "  "
    hdr += _c(_C.WHITE, headers[0].ljust(col_w[0]), bold=True)
    for i, h in enumerate(headers[1:], 1):
        hdr += _c(_C.GRAY, h.rjust(col_w[i]), bold=True)
    _print(hdr)
    _print(sep)

    for model_name, m in metrics.items():
        acc  = m.get("accuracy",  0.0)
        prec = m.get("precision", 0.0)
        rec  = m.get("recall",    0.0)
        f1   = m.get("f1_score",  0.0)

        row = "  "
        row += _c(_C.WHITE, model_name.ljust(col_w[0]))
        row += _c(_C.CYAN,  f"{acc*100:.2f}%".rjust(col_w[1]))
        row += _c(_C.CYAN,  f"{prec*100:.2f}%".rjust(col_w[2]))
        row += _c(_C.CYAN,  f"{rec*100:.2f}%".rjust(col_w[3]))
        row += _c(_C.CYAN,  f"{f1*100:.2f}%".rjust(col_w[4]))
        _print(row)

    _print(sep)


def best_model_announcement(model_name: str, metrics: Dict[str, Any]):
    """Print the best model selection announcement."""
    acc  = metrics.get("accuracy",  0.0)
    prec = metrics.get("precision", 0.0)
    rec  = metrics.get("recall",    0.0)
    f1   = metrics.get("f1_score",  0.0)
    conf = metrics.get("confusion_matrix", [])

    _print(f"\n{_c(_C.GREEN, '  [BEST MODEL SELECTED]', bold=True)}")
    _print(f"  {_c(_C.WHITE, '   Model    :', bold=True)} {_c(_C.CYAN, model_name, bold=True)}")
    _print(f"  {_c(_C.WHITE, '   Accuracy :', bold=True)} {_c(_C.GREEN, f'{acc*100:.2f}%')}")
    _print(f"  {_c(_C.WHITE, '   Precision:', bold=True)} {_c(_C.GREEN, f'{prec*100:.2f}%')}")
    _print(f"  {_c(_C.WHITE, '   Recall   :', bold=True)} {_c(_C.GREEN, f'{rec*100:.2f}%')}")
    _print(f"  {_c(_C.WHITE, '   F1 Score :', bold=True)} {_c(_C.GREEN, f'{f1*100:.2f}%')}")

    if conf and len(conf) == 2:
        tn, fp, fn, tp = conf[0][0], conf[0][1], conf[1][0], conf[1][1]
        _print(f"\n  {_c(_C.WHITE, 'Confusion Matrix', bold=True)}")
        _print(f"  {_c(_C.GRAY, '  +----------+----------+')}")
        _print(f"  {_c(_C.GRAY, '  |')} {_c(_C.GREEN, f'TN={tn:<6}')} {_c(_C.GRAY, '|')} {_c(_C.RED, f'FP={fp:<6}')} {_c(_C.GRAY, '|')}")
        _print(f"  {_c(_C.GRAY, '  +----------+----------+')}")
        _print(f"  {_c(_C.GRAY, '  |')} {_c(_C.RED, f'FN={fn:<6}')} {_c(_C.GRAY, '|')} {_c(_C.GREEN, f'TP={tp:<6}')} {_c(_C.GRAY, '|')}")
        _print(f"  {_c(_C.GRAY, '  +----------+----------+')}")


def system_ready(dataset_size: int, feature_count: int, duration_s: float):
    """Print the final system-ready status block."""
    line = "=" * 52
    _print(f"\n{_c(_C.GREEN, '  ' + line, bold=True)}")
    _print(_c(_C.GREEN, "  [READY] NEXSURE ENGINE OPERATIONAL", bold=True))
    _print(f"  {_c(_C.GRAY, f'Dataset  : {dataset_size} records')}")
    _print(f"  {_c(_C.GRAY, f'Features : {feature_count}')}")
    _print(f"  {_c(_C.GRAY, f'Duration : {duration_s:.1f}s')}")
    _print(f"  {_c(_C.GRAY, f'Status   : ')} {_c(_C.GREEN, 'HEALTHY', bold=True)}")
    _print(f"{_c(_C.GREEN, '  ' + line, bold=True)}\n")


def artifact_loaded(artifact_name: str, path: str):
    """Log a successful artifact load."""
    _print(f"  {_c(_C.GREEN, '[OK]')}  {_c(_C.GRAY, 'Loaded')} {_c(_C.WHITE, artifact_name)} "
           f"{_c(_C.GRAY, f'<- {path}')}")


# --------------------------------------------------------------------------- #
#  Benchmark Engine Logger Extensions                                          #
# --------------------------------------------------------------------------- #

def benchmark_model_start(index: int, total: int, display_name: str):
    """Print a model training progress header."""
    _print(f"\n  {_c(_C.BLUE, f'[{index}/{total}]', bold=True)} {_c(_C.WHITE, display_name, bold=True)}")


def benchmark_model_done(display_name: str, result: dict):
    """Print per-model outcome (PASS/FAIL) with key metrics."""
    status = result.get("status", "unknown")
    if status == "success":
        f1    = result.get("f1_score", 0.0)
        roc   = result.get("roc_auc") or 0.0
        lat   = result.get("model_only_latency_ms", 0.0)
        dur   = result.get("training_duration_s", 0.0)
        _print(f"  {_c(_C.GREEN, '[PASS]')}  {_c(_C.GRAY, display_name)}"
               f"  f1={_c(_C.CYAN, f'{f1:.4f}')}"
               f"  roc={_c(_C.CYAN, f'{roc:.4f}')}"
               f"  lat={_c(_C.YELLOW, f'{lat:.2f}ms')}"
               f"  dur={_c(_C.GRAY, f'{dur:.1f}s')}")
    else:
        err = result.get("error", "Unknown error")
        _print(f"  {_c(_C.RED, '[FAIL]')}  {_c(_C.GRAY, display_name)}  {_c(_C.RED, str(err)[:80])}")


def benchmark_table(results: dict, ranking: list):
    """Print full benchmark comparison table for all models."""
    col_w = [22, 8, 8, 8, 8, 10, 10, 8, 8]
    headers = ["Model", "Acc%", "Prec%", "Rec%", "F1%", "ROC", "CV-F1", "Lat(ms)", "Size(MB)"]
    sep = "  " + "-" * 92

    _print(f"\n{_c(_C.CYAN, '  BENCHMARK RESULTS Ã¢â‚¬â€ ALL MODELS', bold=True)}")
    _print(sep)
    hdr = "  "
    for i, h in enumerate(headers):
        hdr += _c(_C.GRAY, h.ljust(col_w[i]) if i == 0 else h.rjust(col_w[i]), bold=True)
    _print(hdr)
    _print(sep)

    for rank, key in enumerate(ranking, 1):
        r = results.get(key, {})
        if r.get("status") != "success":
            row = f"  {_c(_C.RED, key.ljust(col_w[0]))} {_c(_C.RED, 'FAILED')}"
            _print(row)
            continue

        crown = " [C]" if rank == 1 else ("  #" + str(rank))
        name  = (r.get("display_name", key) + crown)[:col_w[0]]
        acc   = f"{r.get('accuracy', 0)*100:.1f}"
        prec  = f"{r.get('precision', 0)*100:.1f}"
        rec   = f"{r.get('recall', 0)*100:.1f}"
        f1    = f"{r.get('f1_score', 0)*100:.1f}"
        roc   = f"{r.get('roc_auc') or 0:.4f}"
        cvf1  = f"{r.get('cv_f1_mean') or 0:.4f}"
        lat   = f"{r.get('model_only_latency_ms', 0):.2f}"
        size  = f"{r.get('model_size_mb', 0):.3f}"

        color = _C.GREEN if rank == 1 else _C.WHITE
        row = "  " + _c(color, name.ljust(col_w[0]), bold=(rank == 1))
        row += _c(_C.CYAN, acc.rjust(col_w[1]))
        row += _c(_C.CYAN, prec.rjust(col_w[2]))
        row += _c(_C.CYAN, rec.rjust(col_w[3]))
        row += _c(_C.CYAN, f1.rjust(col_w[4]))
        row += _c(_C.CYAN, roc.rjust(col_w[5]))
        row += _c(_C.CYAN, cvf1.rjust(col_w[6]))
        row += _c(_C.YELLOW, lat.rjust(col_w[7]))
        row += _c(_C.GRAY, size.rjust(col_w[8]))
        _print(row)

    _print(sep)

    # Show failed models below table
    failed = [k for k, v in results.items() if v.get("status") != "success"]
    if failed:
        _print(f"  {_c(_C.RED, '[FAILED]')} {', '.join(failed)}")


def champion_summary(champion: str, runner_up: str, metrics: dict):
    """Print the champion model announcement banner."""
    display = metrics.get("display_name", champion)
    f1    = metrics.get("f1_score", 0)
    roc   = metrics.get("roc_auc") or 0
    acc   = metrics.get("accuracy", 0)
    lat_m = metrics.get("model_only_latency_ms", 0)
    lat_e = metrics.get("end_to_end_latency_ms", 0)
    cv_f1 = metrics.get("cv_f1_mean") or 0
    size  = metrics.get("model_size_mb", 0)
    conf  = metrics.get("confusion_matrix", [])

    _print(f"\n{_c(_C.GREEN, '  CHAMPION SELECTED', bold=True)}")
    line = "=" * 52
    _print(f"  {_c(_C.GREEN, line, bold=True)}")
    _print(f"  {_c(_C.WHITE, '  Model       :', bold=True)} {_c(_C.GREEN, display, bold=True)}")
    _print(f"  {_c(_C.WHITE, '  Accuracy    :', bold=True)} {_c(_C.CYAN, f'{acc*100:.2f}%')}")
    _print(f"  {_c(_C.WHITE, '  F1 Score    :', bold=True)} {_c(_C.CYAN, f'{f1*100:.2f}%')}")
    _print(f"  {_c(_C.WHITE, '  ROC-AUC     :', bold=True)} {_c(_C.CYAN, f'{roc:.4f}')}")
    _print(f"  {_c(_C.WHITE, '  CV F1 Mean  :', bold=True)} {_c(_C.CYAN, f'{cv_f1:.4f}')}")
    _print(f"  {_c(_C.WHITE, '  Lat (model) :', bold=True)} {_c(_C.YELLOW, f'{lat_m:.2f}ms')}")
    _print(f"  {_c(_C.WHITE, '  Lat (e2e)   :', bold=True)} {_c(_C.YELLOW, f'{lat_e:.2f}ms')}")
    _print(f"  {_c(_C.WHITE, '  Size        :', bold=True)} {_c(_C.GRAY, f'{size:.3f} MB')}")
    if runner_up:
        _print(f"  {_c(_C.WHITE, '  Runner-up   :', bold=True)} {_c(_C.GRAY, runner_up)}")
    _print(f"  {_c(_C.GREEN, line, bold=True)}\n")

    if conf and len(conf) == 2:
        tn, fp = conf[0][0], conf[0][1]
        fn, tp = conf[1][0], conf[1][1]
        _print(f"  {_c(_C.WHITE, 'Confusion Matrix', bold=True)}")
        _print(f"  {_c(_C.GRAY, '  +----------+----------+')}")
        _print(f"  {_c(_C.GRAY, '  |')} {_c(_C.GREEN, f'TN={tn:<6}')} {_c(_C.GRAY, '|')} {_c(_C.RED, f'FP={fp:<6}')} {_c(_C.GRAY, '|')}")
        _print(f"  {_c(_C.GRAY, '  +----------+----------+')}")
        _print(f"  {_c(_C.GRAY, '  |')} {_c(_C.RED, f'FN={fn:<6}')} {_c(_C.GRAY, '|')} {_c(_C.GREEN, f'TP={tp:<6}')} {_c(_C.GRAY, '|')}")
        _print(f"  {_c(_C.GRAY, '  +----------+----------+')}")



def registry_summary(
    champion: Dict[str, Any],
    challenger: Optional[Dict[str, Any]],
    status: str,
    promotion_count: int = 0,
    active_version: str = "v2.1",
):
    """Print the Nexsure Model Registry governance summary."""
    champ_name = champion.get("display_name") or champion.get("name") or "None"
    champ_f1   = champion.get("f1_score") or 0.0
    champ_roc  = champion.get("roc_auc") or 0.0
    champ_lat  = champion.get("latency_ms") or 0.0
    
    chal_name  = challenger.get("display_name") or challenger.get("name") or "None" if challenger else "None"
    chal_f1    = challenger.get("f1_score") or 0.0 if challenger else 0.0
    chal_roc   = challenger.get("roc_auc") or 0.0 if challenger else 0.0
    chal_lat   = challenger.get("latency_ms") or 0.0 if challenger else 0.0

    _print(f"\n{_c(_C.CYAN, '====================================================', bold=True)}")
    _print(_c(_C.CYAN, "  NEXSURE MODEL REGISTRY (CHAMPION / CHALLENGER)", bold=True))
    _print(f"{_c(_C.CYAN, '====================================================', bold=True)}")
    _print(f"  {_c(_C.GREEN, 'Active Champion :', bold=True)} {_c(_C.WHITE, f'{champ_name} ({active_version})')}  [SERVING]")
    _print(f"    Metrics: F1={champ_f1*100:.2f}% | ROC-AUC={champ_roc:.4f} | Latency={champ_lat:.2f}ms")
    _print(f"  {_c(_C.YELLOW, 'Challenger      :', bold=True)} {_c(_C.WHITE, f'{chal_name} ({active_version})')}  [NON-SERVING]")
    if challenger:
        _print(f"    Metrics: F1={chal_f1*100:.2f}% | ROC-AUC={chal_roc:.4f} | Latency={chal_lat:.2f}ms")
    _print(f"  {_c(_C.WHITE, 'Registry Status :', bold=True)} {_c(_C.GREEN if status == 'healthy' else _C.RED, status.upper(), bold=True)}")
    _print(f"  {_c(_C.WHITE, 'Audit Events    :', bold=True)} {_c(_C.CYAN, f'{promotion_count} recorded lifecycle events')}")
    _print(f"{_c(_C.CYAN, '====================================================', bold=True)}\n")


def promotion_event(
    action: str,
    from_model: Optional[str],
    from_version: Optional[str],
    to_model: str,
    to_version: str,
    reason: str,
    actor: str = "system",
):
    """Print an audit promotion event to the terminal."""
    _print(f"\n{_c(_C.GREEN, f'>> MODEL PROMOTION EVENT [{action.upper()}]', bold=True)}")
    _print(f"  {_c(_C.GRAY, 'Transition :')} {_c(_C.YELLOW, f'{from_model} ({from_version})')} -> {_c(_C.GREEN, f'{to_model} ({to_version})', bold=True)}")
    _print(f"  {_c(_C.GRAY, 'Reason     :')} {_c(_C.WHITE, reason)}")
    _print(f"  {_c(_C.GRAY, 'Actor      :')} {_c(_C.CYAN, actor)}")
    _print(f"  {_c(_C.GRAY, 'Timestamp  :')} {_c(_C.GRAY, _ts())}\n")


def rollback_event(
    from_model: str,
    from_version: Optional[str],
    to_model: str,
    to_version: str,
    reason: str,
    actor: str = "system",
):
    """Print an audit rollback event to the terminal."""
    _print(f"\n{_c(_C.YELLOW, '>> MODEL ROLLBACK EVENT [FAULT RECOVERY]', bold=True)}")
    _print(f"  {_c(_C.GRAY, 'Reverted   :')} {_c(_C.RED, f'{from_model} ({from_version})')} -> {_c(_C.GREEN, f'{to_model} ({to_version})', bold=True)}")
    _print(f"  {_c(_C.GRAY, 'Reason     :')} {_c(_C.WHITE, reason)}")
    _print(f"  {_c(_C.GRAY, 'Actor      :')} {_c(_C.CYAN, actor)}")
    _print(f"  {_c(_C.GRAY, 'Timestamp  :')} {_c(_C.GRAY, _ts())}\n")


def startup_registry_status(
    is_valid: bool,
    champion_name: str,
    challenger_name: Optional[str],
    version: str = "v2.1",
):
    """Print concise startup registry validation status."""
    if is_valid:
        _print(f"  {_c(_C.GREEN, '[REGISTRY]')} Verified: Champion={_c(_C.WHITE, champion_name, bold=True)} ({version}) | Challenger={_c(_C.GRAY, challenger_name or 'None')} ({version})")
    else:
        _print(f"  {_c(_C.YELLOW, '[REGISTRY]')} Incomplete or corrupt registry; triggering rebuild...")

def prediction_latency_summary(
    pred_id: int,
    stages_ms: Dict[str, float],
    total_ms: float,
    rolling_metrics: Dict[str, Any],
    champion_name: str,
    cache_hit: bool = True,
    shap_cache_hit: bool = True,
    shap_diagnostics: Optional[Dict[str, Any]] = None,
):
    """Print structured, color-coded stage-by-stage latency for an inference request with SHAP Optimization tracking."""
    _print(f"\n{_c(_C.CYAN, f'>> INFERENCE TELEMETRY [PREDICTION #{pred_id}]', bold=True)} {_c(_C.GRAY, f'({champion_name})')}")
    _print(f"  {_c(_C.GRAY, '------------------------------------------------')}")

    shap_tag = "  " + _c(_C.GREEN, "[CACHE HIT]") if shap_cache_hit else "  " + _c(_C.YELLOW, "[CACHE MISS]")

    stages_order = [
        ("validation", "Validation .............", ""),
        ("preprocessing", "Preprocessing ..........", ""),
        ("feature_ordering", "Feature Ordering .......", ""),
        ("inference", "Model Inference ........", ""),
        ("shap", "SHAP Generation ........", shap_tag),
        ("explanation_formatting", "Explain Formatting .....", ""),
        ("serialization", "Serialization ..........", ""),
    ]
    for key, label, tag in stages_order:
        val = stages_ms.get(key, 0.0)
        _print(f"  {_c(_C.WHITE, label)} {_c(_C.YELLOW, f'{val:.3f} ms')}{tag}")

    _print(f"  {_c(_C.GRAY, '------------------------------------------------')}")
    _print(f"  {_c(_C.GREEN, 'TOTAL LATENCY ..........', bold=True)} {_c(_C.GREEN, f'{total_ms:.3f} ms', bold=True)}")

    # Rolling telemetry summary
    tot_stats = rolling_metrics.get("total", {})
    p50 = tot_stats.get("p50", 0.0)
    p95 = tot_stats.get("p95", 0.0)
    p99 = tot_stats.get("p99", 0.0)
    hit_icon = "[CACHE HIT]" if cache_hit else "[CACHE MISS]"
    _print(f"  {_c(_C.GRAY, 'Rolling Stats (P50/P95/P99):')} {_c(_C.CYAN, f'{p50:.2f}ms / {p95:.2f}ms / {p99:.2f}ms')} {_c(_C.GREEN, hit_icon)}")

    # SHAP Optimization telemetry
    if shap_diagnostics:
        prev_shap = shap_diagnostics.get("baseline_shap_latency_ms", 9.87)
        cur_shap = shap_diagnostics.get("current_shap_average_ms", stages_ms.get("shap", 0.0))
        reduction = shap_diagnostics.get("shap_latency_reduction_pct", 0.0)
        _print(f"  {_c(_C.GRAY, '------------------------------------------------')}")
        _print(f"  {_c(_C.MAGENTA, 'SHAP Improvement', bold=True)}")
        _print(f"  {_c(_C.WHITE, 'Previous Avg ......')} {_c(_C.GRAY, f'{prev_shap:.2f} ms')}")
        _print(f"  {_c(_C.WHITE, 'Current Avg .......')} {_c(_C.GREEN, f'{cur_shap:.2f} ms', bold=True)}")
        _print(f"  {_c(_C.WHITE, 'Reduction .........')} {_c(_C.GREEN, f'{reduction:.1f}%', bold=True)}\n")
    else:
        _print("")

def target_semantics_verified(
    positive_class_name: str,
    negative_class_name: str,
    positive_class: int = 1,
    negative_class: int = 0,
    threshold_source: str = "training_split_median",
    target_formula: str = "charges < training_median",
):
    """Print Target Semantics startup verification."""
    _print(f"\n{_c(_C.CYAN, '====================================================', bold=True)}")
    _print(f"  {_c(_C.GREEN, '>> TARGET SEMANTICS VERIFIED', bold=True)}")
    _print(f"  {_c(_C.WHITE, 'Positive Class →', bold=True)} {_c(_C.GREEN, f'{positive_class_name} ({positive_class})', bold=True)}")
    _print(f"  {_c(_C.WHITE, 'Negative Class →', bold=True)} {_c(_C.RED, f'{negative_class_name} ({negative_class})', bold=True)}")
    _print(f"  {_c(_C.WHITE, 'Threshold Source →', bold=True)} {_c(_C.CYAN, threshold_source)}")
    _print(f"  {_c(_C.WHITE, 'Target Formula →', bold=True)} {_c(_C.GRAY, target_formula)}")
    _print(f"{_c(_C.CYAN, '====================================================', bold=True)}\n")
