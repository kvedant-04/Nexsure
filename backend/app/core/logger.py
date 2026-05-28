"""
Enterprise Terminal Logger — Nexsure AI Engine
Provides rich, colored, structured terminal output without external dependencies.
Uses ANSI escape codes with Windows-safe character fallbacks.
"""

import sys
from datetime import datetime
from typing import Any, Dict, Optional

# ─── Windows ANSI Support ──────────────────────────────────────────────────────
try:
    import colorama
    colorama.init(autoreset=True)
    _ANSI_SUPPORTED = True
except ImportError:
    _ANSI_SUPPORTED = False

# ─── Windows encoding fix ─────────────────────────────────────────────────────
# Force stdout to UTF-8 on Windows to avoid cp1252 UnicodeEncodeErrors
try:
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.platform == "win32" and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ─── ANSI Color Codes ─────────────────────────────────────────────────────────
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


# ─── Public API ───────────────────────────────────────────────────────────────

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
