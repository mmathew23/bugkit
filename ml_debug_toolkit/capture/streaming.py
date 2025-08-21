# === Capture & Compare Toolkit (Streaming + PyTorch 2.6-safe loader) ===
# - Streams each module call to disk (no giant in-RAM list)
# - Optional compression: "none" | "gzip" | "xz" | "bz2"
# - Store policy: "full" (inputs+outputs), "outputs_only", or "inputs_only"
# - Filters: leaf-only, include/exclude by name regex and/or class type
# - Streaming comparator that casts to fp32 and writes a CSV
# - Robust torch.load handling for PyTorch 2.6 (weights_only default)

import os, re, io, json, csv, time, hashlib, contextlib, glob
from typing import Any, Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import pandas as pd

EPS = 1e-12

# ------------------------ Utilities ------------------------

def to_cpu_detached(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu()

def extract_tensors(x: Any) -> Any:
    """Mirror the structure, keeping only tensors (moved to CPU, detached)."""
    if torch.is_tensor(x):
        return to_cpu_detached(x)
    if isinstance(x, (list, tuple)):
        return type(x)(extract_tensors(i) for i in x)
    if isinstance(x, dict):
        return {k: extract_tensors(v) for k, v in x.items()}
    return None

def extract_dtypes(x: Any) -> Any:
    if torch.is_tensor(x):
        return str(x.dtype).replace("torch.", "")
    if isinstance(x, (list, tuple)):
        return type(x)(extract_dtypes(i) for i in x)
    if isinstance(x, dict):
        return {k: extract_dtypes(v) for k, v in x.items()}
    return None

def extract_shapes(x: Any) -> Any:
    if torch.is_tensor(x):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        return type(x)(extract_shapes(i) for i in x)
    if isinstance(x, dict):
        return {k: extract_shapes(v) for k, v in x.items()}
    return None

def is_float_tensor(t: torch.Tensor) -> bool:
    return torch.is_tensor(t) and t.is_floating_point()

def summarize_param_dtypes(model: nn.Module) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in model.parameters(recurse=True):
        dt = str(p.dtype).replace("torch.", "")
        counts[dt] = counts.get(dt, 0) + p.numel()
    return counts

def build_name_map(model: nn.Module) -> Dict[nn.Module, str]:
    return {m: n for n, m in model.named_modules()}

def _compile_regex_list(xs: Optional[List[str]]) -> Optional[List[re.Pattern]]:
    if not xs:
        return None
    return [re.compile(x) for x in xs]

def _name_matches(name: str, inc: Optional[List[re.Pattern]], exc: Optional[List[re.Pattern]]) -> bool:
    if inc and not any(rx.search(name) for rx in inc):
        return False
    if exc and any(rx.search(name) for rx in exc):
        return False
    return True

def _safe_name(s: str) -> str:
    """Short, filesystem-safe filename with a stable hash suffix."""
    h = hashlib.blake2b(s.encode(), digest_size=8).hexdigest()
    core = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    core = (core[:80] + "...") if len(core) > 80 else core
    return f"{core}__{h}"

# ------------------------ Compressed I/O ------------------------

def _open_write(base: str, compression: str):
    compression = (compression or "none").lower()
    if compression == "gzip":
        import gzip; return gzip.open(base + ".gz", "wb")
    if compression == "xz":
        import lzma; return lzma.open(base + ".xz", "wb")
    if compression == "bz2":
        import bz2;  return bz2.open(base + ".bz2", "wb")
    return open(base + ".pt", "wb")

def _open_read(base: str) -> io.BufferedReader:
    """Try known extensions in order, then raw .pt."""
    for ext, opener in [(".gz","gzip"), (".xz","lzma"), (".bz2","bz2"), (".pt",None)]:
        p = base + ext
        if os.path.exists(p):
            if opener == "gzip":
                import gzip; return gzip.open(p, "rb")
            if opener == "lzma":
                import lzma; return lzma.open(p, "rb")
            if opener == "bz2":
                import bz2;  return bz2.open(p, "rb")
            return open(p, "rb")
    raise FileNotFoundError(f"No capture file found for base path: {base}")

# ------------------------ PyTorch 2.6-safe loader ------------------------

def _torch_load_capture(fileobj, map_location="cpu", loader_mode: str = "unsafe"):
    """
    loader_mode:
      - "unsafe": use weights_only=False (recommended for your own capture files).
        This mirrors PyTorch <=2.5 behavior and avoids the PyTorch 2.6 UnpicklingError.
      - "safe":   keep weights_only=True but allowlist TorchVersion; use ONLY if you truly want a restricted loader.
    """
    if loader_mode == "unsafe":
        # Prefer explicit weights_only=False for PyTorch 2.6+.
        try:
            return torch.load(fileobj, map_location=map_location, weights_only=False)
        except TypeError:
            # Older PyTorch without weights_only kwarg
            return torch.load(fileobj, map_location=map_location)
    elif loader_mode == "safe":
        sg_ctx = getattr(torch.serialization, "safe_globals", None)
        if sg_ctx is not None:
            try:
                # Allowlist TorchVersion (per PyTorch error hint) then use weights_only=True
                with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                    return torch.load(fileobj, map_location=map_location, weights_only=True)
            except Exception:
                # As a last resort (if the file contains more Python objects), fall back unsafely.
                return torch.load(fileobj, map_location=map_location, weights_only=False)
        else:
            # No safe_globals available; fall back unsafely.
            return torch.load(fileobj, map_location=map_location, weights_only=False)
    else:
        raise ValueError("loader_mode must be 'unsafe' or 'safe'")

# ------------------------ Streaming writer ------------------------

class StreamingCaptureWriter:
    """
    Streams each record to disk immediately and appends to a manifest.jsonl (no in-RAM list).
    """
    def __init__(self, out_dir: str, meta: Dict[str, Any], compression: str = "none"):
        self.out_dir = out_dir
        self.meta = meta
        self.compression = compression
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self.manifest_path = os.path.join(out_dir, "manifest.jsonl")
        self.manifest_fp = open(self.manifest_path, "a", buffering=1)
        self.count = 0

    def add(self, rec: Dict[str, Any]):
        base = os.path.join(self.out_dir, f"rec_{_safe_name(rec['module_name'])}__{rec['call_idx']:06d}")
        with _open_write(base, self.compression) as f:
            torch.save(rec, f)
        man_row = {
            "idx": self.count,
            "module_name": rec["module_name"],
            "module_type": rec["module_type"],
            "call_idx": rec["call_idx"],
            "file_base": os.path.basename(base)  # extension added by compressor
        }
        self.manifest_fp.write(json.dumps(man_row) + "\n")
        self.count += 1

    def close(self):
        try:
            self.manifest_fp.close()
        except Exception:
            pass

# ------------------------ Hook registration & capture ------------------------

def register_forward_hooks(
    model: nn.Module,
    on_record,
    *,
    capture_leaves_only: bool = False,
    include_name_patterns: Optional[List[str]] = None,
    exclude_name_patterns: Optional[List[str]] = None,
    include_types: Optional[List[str]] = None,  # e.g., ["Linear","LayerNorm","Embedding","Conv2d"]
    exclude_types: Optional[List[str]] = None,
    store: str = "full",                         # "full" | "outputs_only" | "inputs_only"
):
    assert store in ("full", "outputs_only", "inputs_only")
    name_map = build_name_map(model)
    inc_rx = _compile_regex_list(include_name_patterns)
    exc_rx = _compile_regex_list(exclude_name_patterns)

    def is_leaf(m: nn.Module) -> bool:
        return len(list(m.children())) == 0

    call_counts: Dict[str, int] = {}
    handles = []

    for mod, name in name_map.items():
        if capture_leaves_only and not is_leaf(mod):
            continue
        if include_types and (mod.__class__.__name__ not in include_types):
            continue
        if exclude_types and (mod.__class__.__name__ in exclude_types):
            continue
        if not _name_matches(name, inc_rx, exc_rx):
            continue

        def make_hook(nm: str, mref: nn.Module):
            def hook(mref, inputs, outputs):
                idx = call_counts.get(nm, 0)
                call_counts[nm] = idx + 1
                rec = {
                    "module_name": nm,
                    "module_type": mref.__class__.__name__,
                    "call_idx": idx,
                    "input_dtypes": extract_dtypes(inputs),
                    "output_dtypes": extract_dtypes(outputs),
                    "input_shapes": extract_shapes(inputs),
                    "output_shapes": extract_shapes(outputs),
                    "inputs": None,
                    "outputs": None,
                }
                if store in ("full", "inputs_only"):
                    rec["inputs"] = extract_tensors(inputs)
                if store in ("full", "outputs_only"):
                    rec["outputs"] = extract_tensors(outputs)
                on_record(rec)
            return hook

        handles.append(mod.register_forward_hook(make_hook(name, mod)))

    return handles

def capture_forward_pass(
    model: nn.Module,
    example_inputs: Any,              # you prepare (already on device & desired dtypes)
    out_dir: str,                     # directory for streaming capture
    *,
    run_tag: Optional[str] = None,
    set_eval: bool = True,
    no_grad: bool = True,
    capture_leaves_only: bool = False,
    include_name_patterns: Optional[List[str]] = None,
    exclude_name_patterns: Optional[List[str]] = None,
    include_types: Optional[List[str]] = None,
    exclude_types: Optional[List[str]] = None,
    store: str = "full",              # "full" | "outputs_only" | "inputs_only"
    compression: str = "none",        # "none" | "gzip" | "xz" | "bz2"
) -> Dict[str, Any]:
    """
    Registers hooks, runs one forward, and streams records to out_dir.
    Does NOT cast or move your model or inputs.
    """
    if set_eval:
        model.eval()

    meta = {
        "run_tag": run_tag,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": str(torch.__version__),  # store as plain string
        "capture_leaves_only": capture_leaves_only,
        "store": store,
        "compression": compression,
        "filters": {
            "include_name_patterns": include_name_patterns,
            "exclude_name_patterns": exclude_name_patterns,
            "include_types": include_types,
            "exclude_types": exclude_types,
        },
        "param_dtype_counts": summarize_param_dtypes(model),
    }

    writer = StreamingCaptureWriter(out_dir=out_dir, meta=meta, compression=compression)
    handles = register_forward_hooks(
        model,
        on_record=writer.add,
        capture_leaves_only=capture_leaves_only,
        include_name_patterns=include_name_patterns,
        exclude_name_patterns=exclude_name_patterns,
        include_types=include_types,
        exclude_types=exclude_types,
        store=store,
    )

    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with ctx:
        if isinstance(example_inputs, dict):
            _ = model(**example_inputs)
        elif isinstance(example_inputs, (list, tuple)):
            _ = model(*example_inputs)
        else:
            _ = model(example_inputs)

    for h in handles:
        h.remove()

    writer.close()
    print(f"Saved {writer.count} records to directory: {out_dir}")
    return {"meta": meta, "num_records": writer.count}

# ------------------------ Comparison ------------------------

def _flatten_tensors_with_meta(tobj: Any, dtype_obj: Any, shape_obj: Any):
    """Return list of (tensor, dtype_str, shape_tuple) leaves in traversal order."""
    out = []
    def rec(t, dt, sh):
        if torch.is_tensor(t):
            out.append((t, dt, sh))
        elif isinstance(t, (list, tuple)):
            for i in range(len(t)):
                rec(t[i], dt[i] if dt is not None else None, sh[i] if sh is not None else None)
        elif isinstance(t, dict):
            for k in list(t.keys()):
                rec(t[k],
                    dt.get(k) if isinstance(dt, dict) else None,
                    sh.get(k) if isinstance(sh, dict) else None)
    rec(tobj, dtype_obj, shape_obj)
    return out

def _compare_one_pair_fp32(ta: torch.Tensor, tb: torch.Tensor) -> Dict[str, float]:
    ta32 = ta.to(torch.float32)
    tb32 = tb.to(torch.float32)
    diff = ta32 - tb32
    abs_diff = diff.abs()
    mae = torch.nanmean(abs_diff).item()
    mse = torch.nanmean(diff * diff).item()
    maxae = torch.nanmax(abs_diff).item()
    den_a = torch.nanmean(ta32.abs()).item()
    den_b = torch.nanmean(tb32.abs()).item()
    rel_mae_refA = mae / (den_a + EPS)
    rel_mae_refB = mae / (den_b + EPS)
    v1 = ta32.flatten()
    v2 = tb32.flatten()
    dot = torch.dot(v1, v2).item()
    n1 = v1.norm().item()
    n2 = v2.norm().item()
    cosine = dot / ((n1 + EPS) * (n2 + EPS))
    return {
        "mae32": mae,
        "mse32": mse,
        "maxae32": maxae,
        "rel_mae_refA": rel_mae_refA,
        "rel_mae_refB": rel_mae_refB,
        "cosine": cosine,
        "numel": v1.numel(),
    }

def _rec_base_path(out_dir: str, module_name: str, call_idx: int) -> str:
    return os.path.join(out_dir, f"rec_{_safe_name(module_name)}__{call_idx:06d}")

def compare_captures_streaming(
    dir_a: str,            # capture directory for run A (e.g., bf16)
    dir_b: str,            # capture directory for run B (e.g., fp16)
    out_csv: str,
    *,
    compare_inputs: bool = True,
    compare_outputs: bool = True,
    loader_mode: str = "unsafe",   # "unsafe" uses weights_only=False to avoid PyTorch 2.6 error
) -> int:
    """
    Streams over manifest of B; for each (module_name, call_idx) loads just that pair from A & B,
    casts to fp32, computes metrics, and appends one CSV row per tensor compared.
    Returns number of CSV rows written. Constant RAM usage.
    """
    man_b = os.path.join(dir_b, "manifest.jsonl")
    assert os.path.exists(man_b), f"Missing manifest: {man_b}"
    rows_written = 0

    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "module_name","module_type","call_idx","io","tensor_index","shape",
            "dtype_a","dtype_b","mae32","mse32","maxae32","rel_mae_refA","rel_mae_refB","cosine","numel"
        ])

        with open(man_b, "r") as f:
            for line in f:
                mb = json.loads(line)
                name = mb["module_name"]; idx = mb["call_idx"]
                base_a = _rec_base_path(dir_a, name, idx)
                base_b = os.path.join(dir_b, mb["file_base"])

                # Load A (may not exist if call counts differ)
                try:
                    with _open_read(base_a) as fa:
                        ra = _torch_load_capture(fa, map_location="cpu", loader_mode=loader_mode)
                except FileNotFoundError:
                    continue  # present only in B; skip

                # Load B
                with _open_read(base_b) as fb:
                    rb = _torch_load_capture(fb, map_location="cpu", loader_mode=loader_mode)

                def push_rows(ra, rb, io_label: str):
                    nonlocal rows_written
                    a_list = _flatten_tensors_with_meta(ra[io_label], ra[f"{io_label[:-1]}_dtypes"], ra[f"{io_label[:-1]}_shapes"])
                    b_list = _flatten_tensors_with_meta(rb[io_label], rb[f"{io_label[:-1]}_dtypes"], rb[f"{io_label[:-1]}_shapes"])
                    n = min(len(a_list), len(b_list))
                    for i in range(n):
                        ta, dta, sha = a_list[i]
                        tb, dtb, shb = b_list[i]
                        if not (is_float_tensor(ta) and is_float_tensor(tb)):
                            continue
                        stats = _compare_one_pair_fp32(ta, tb)
                        writer.writerow([
                            ra["module_name"], ra["module_type"], ra["call_idx"],
                            "input" if io_label == "inputs" else "output",
                            i, str(sha if sha is not None else tuple(ta.shape)),
                            dta, dtb,
                            stats["mae32"], stats["mse32"], stats["maxae32"],
                            stats["rel_mae_refA"], stats["rel_mae_refB"],
                            stats["cosine"], stats["numel"]
                        ])
                        rows_written += 1

                if compare_inputs and rb.get("inputs") is not None and ra.get("inputs") is not None:
                    push_rows(ra, rb, "inputs")
                if compare_outputs and rb.get("outputs") is not None and ra.get("outputs") is not None:
                    push_rows(ra, rb, "outputs")

    print(f"Wrote metrics rows -> {out_csv} (rows={rows_written})")
    return rows_written

# ------------------------ Optional: quick summarizer ------------------------

def summarize_compare_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    by_mod = (df
              .groupby(["module_name", "module_type", "io"], as_index=False)
              .agg(mae32_mean=("mae32", "mean"),
                   maxae32_max=("maxae32", "max"),
                   mse32_mean=("mse32", "mean"),
                   cosine_mean=("cosine", "mean"),
                   numel_sum=("numel", "sum")))
    return by_mod
