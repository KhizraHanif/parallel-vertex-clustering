#!/usr/bin/env python3
import os, re, sys, csv, math
from collections import defaultdict, OrderedDict

# -------------------------
# Inputs / defaults
# -------------------------
LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else "../logs"
CSV_OUT = sys.argv[2] if len(sys.argv) > 2 else "../csv/all_times.csv"

os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)

# -------------------------
# Filename patterns we support
#   A) eps<eps>alg<AlgName>n0t<threads>data<dataset>.log   (Nima-style)
#   B) <dataset>_<AlgName>_eps<eps>_<threads>cores.log     (older/alt)
# -------------------------
pat_a = re.compile(
    r"^eps(?P<eps>[0-9.eE+-]+)alg(?P<alg>.+)n(?P<n>\d+)t(?P<threads>\d+)data(?P<dataset>\w+)\.log$"
)
pat_b = re.compile(
    r"^(?P<dataset>\w+?)_(?P<alg>[^_]+?)_eps(?P<eps>[0-9.eE+-]+)_(?P<threads>\d+)cores\.log$"
)

# Normalize algorithm names a bit
ALG_CANON = {
    "S-Weld": "S-Weld",
    "P-Weld": "P-Weld",
    "P-Weld-Async": "P-Weld-Async",
    "P-Weld Async": "P-Weld-Async",
    "forward": "P-Weld",
    "forward_async": "P-Weld-Async",
    "Open3D": "S-Weld",
}

def canon_alg(name: str) -> str:
    name = name.strip()
    return ALG_CANON.get(name, name)

# -------------------------
# Line parsers
# -------------------------

# 1) Generic “<label> took <value> [unit]”
NUM = r"([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)"
TOOK_RE = re.compile(r"^(?P<label>.+?)\s+took\s+(?P<val>"+NUM+r")\s*(?P<unit>ms|s|sec|seconds)?\s*$", re.IGNORECASE)

# 2) Special total/average aliases
AVG_RE_COLON = re.compile(r"^average time:\s*"+NUM+r"\s*(ms|s|sec|seconds)?", re.IGNORECASE)
AVG_RE_TOOK  = re.compile(r"^average time\s+took\s*"+NUM+r"\s*(ms|s|sec|seconds)?", re.IGNORECASE)

# 3) Trial runtime lines
RT_RE = re.compile(r"^Trial\s+\d+\s+runtime:\s*"+NUM+r"\s*(ms|s|sec|seconds)?", re.IGNORECASE)

# 4) Other counters
ITER_RE = re.compile(r"^numIterations:\s*(\d+)", re.IGNORECASE)
ORG_RE  = re.compile(r"^original vertices:\s*(\d+)", re.IGNORECASE)
SIMP_RE = re.compile(r"^vertices after:\s*(\d+)", re.IGNORECASE)

def unit_to_seconds(val_str: str, unit: str|None) -> float:
    v = float(val_str)
    u = (unit or "").lower()
    if u == "ms":
        return v / 1000.0
    # treat "", "s", "sec", "seconds" the same (seconds)
    return v

def normalize_label(raw: str) -> str:
    s = raw.strip().lower()

    # Common normalizations so columns line up
    repl = [
        (r"^populating adj list",      "Populating adj list"),
        (r"^neighbor search",          "Populating adj list"),
        (r"^clustering\b",             "Clustering"),
        (r"^while loop\b",             "While loop"),
        (r"^single region\b",          "Single region"),
        (r"^update representatives",   "Update representatives"),
        (r"^update new vertices",      "Update new vertices"),
        (r"^update mesh(?:\s*\(manual\))?", "Update mesh"),
        (r"^p-?weld internal: all but stack unwinding", "P-Weld internal"),
        (r"^s-?weld total",            "S-Weld total"),
        (r"^average time\b",           "Average time"),
    ]
    for pat, to in repl:
        if re.match(pat, s, re.IGNORECASE):
            return to
    # Title-case fallback (keeps punctuation if any)
    return raw.strip()

def parse_one_log(path: str):
    """
    Returns:
      meta dict: {dataset, alg, eps, threads}
      phases dict: {phase_label: [list of seconds seen]}
      stats dict:  {runtime_ms: [list], numIter, numOrg, numSimp}
    """
    fname = os.path.basename(path)
    m = pat_a.match(fname) or pat_b.match(fname)
    if not m:
        return None, None, None

    meta = {
        "dataset": m.group("dataset"),
        "alg":     canon_alg(m.group("alg")),
        "eps":     m.group("eps"),
        "threads": int(m.group("threads")),
    }

    phases = defaultdict(list)
    stats  = {
        "runtime_ms": [],
        "numIter":    None,
        "numOrg":     None,
        "numSimp":    None,
    }

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # average time aliases
            mavg = AVG_RE_COLON.search(line) or AVG_RE_TOOK.search(line)
            if mavg:
                val = float(mavg.group(1))
                unit = mavg.group(2).lower() if mavg.lastindex and mavg.lastindex >= 2 and mavg.group(2) else ""
                secs = unit_to_seconds(val, unit)
                phases["Average time"].append(secs)
                continue

            # trial runtime
            mrt = RT_RE.search(line)
            if mrt:
                val = float(mrt.group(1))
                unit = mrt.group(2).lower() if mrt.lastindex and mrt.lastindex >= 2 and mrt.group(2) else ""
                ms = val if unit == "ms" else (val * 1000.0)
                stats["runtime_ms"].append(ms)
                continue

            # generic "<label> took ..."
            mt = TOOK_RE.search(line)
            if mt:
                label = normalize_label(mt.group("label"))
                secs = unit_to_seconds(mt.group("val"), mt.group("unit"))
                phases[label].append(secs)
                continue

            # counters
            mi = ITER_RE.search(line)
            if mi:
                stats["numIter"] = int(mi.group(1))
                continue
            mo = ORG_RE.search(line)
            if mo:
                stats["numOrg"] = int(mo.group(1))
                continue
            ms = SIMP_RE.search(line)
            if ms:
                stats["numSimp"] = int(ms.group(1))
                continue

    return meta, phases, stats

def mean(a): 
    return sum(a)/len(a) if a else math.nan
def stddev(a):
    if not a or len(a) < 2: return math.nan
    m = mean(a)
    return math.sqrt(sum((x-m)*(x-m) for x in a)/(len(a)-1))

# -------------------------
# Scan logs
# -------------------------
rows = []
all_phase_names = set()

for fn in sorted(os.listdir(LOG_DIR)):
    if not fn.lower().endswith(".log"): 
        continue
    meta, phases, stats = parse_one_log(os.path.join(LOG_DIR, fn))
    if meta is None:
        continue

    # Remember all phases seen (superset of columns)
    for p in phases.keys():
        all_phase_names.add(p)

    rows.append((meta, phases, stats))

if not rows:
    print(f"No logs found (or filenames not matched) in: {LOG_DIR}")
    sys.exit(0)

# Nice deterministic column order (common ones first)
common_order = [
    "Populating adj list",
    "Clustering",
    "While loop",
    "Single region",
    "Update representatives",
    "Update new vertices",
    "Update mesh",
    "P-Weld internal",
    "S-Weld total",
    "Average time",
]
others = [p for p in sorted(all_phase_names) if p not in common_order]
phase_cols = [p for p in common_order if p in all_phase_names] + others

# -------------------------
# Build table
# -------------------------
table = []
for meta, phases, stats in rows:
    runtime = stats["runtime_ms"]
    row = OrderedDict()
    row["Dataset"]  = meta["dataset"]
    row["Algorithm"]= meta["alg"]
    row["Epsilon"]  = meta["eps"]
    row["Threads"]  = meta["threads"]
    row["numIter"]  = stats["numIter"] if stats["numIter"] is not None else -1
    row["numOrg"]   = stats["numOrg"]  if stats["numOrg"]  is not None else -1
    row["numSimp"]  = stats["numSimp"] if stats["numSimp"] is not None else -1
    # Per-phase means (seconds)
    for pc in phase_cols:
        row[pc] = mean(phases.get(pc, []))
    # Runtime aggregates (ms)
    row["runtime_ms_mean"] = mean(runtime)
    row["runtime_ms_std"]  = stddev(runtime)
    row["runtime_ms_min"]  = min(runtime) if runtime else math.nan
    row["runtime_ms_max"]  = max(runtime) if runtime else math.nan
    row["runtime_trials"]  = len(runtime)
    table.append(row)

# Sort
def eps_val(eps_str: str)->float:
    try: return float(eps_str)
    except: return float("inf")

table.sort(key=lambda r: (r["Dataset"], r["Algorithm"], eps_val(r["Epsilon"]), r["Threads"]))

# -------------------------
# Print to console
# -------------------------
def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)): return ""
    if isinstance(x, float): return f"{x:.6f}"
    return str(x)

header = list(table[0].keys())
print(" | ".join(header))
print("-" * (len(" | ".join(header)) + 2))
for r in table:
    print(" | ".join(fmt(r[k]) for k in header))

# -------------------------
# Write CSV
# -------------------------
with open(CSV_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for r in table:
        w.writerow([r[k] if not (isinstance(r[k], float) and math.isnan(r[k])) else "" for k in header])

print(f"\nWrote: {CSV_OUT}")
