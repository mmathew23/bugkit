import os, re, json, time, contextlib
from typing import Any, Dict, List, Tuple, Optional
import torch
import torch.nn as nn

EPS = 1e-12

def to_cpu_detached(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu()

def extract_tensors(x: Any) -> Any:
    """Mirror structure, but keep only tensors (moved to CPU, detached)."""
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

def register_forward_hooks(
    model: nn.Module,
    on_record,
    capture_leaves_only: bool = False,
    include_name_patterns: Optional[List[str]] = None,
    exclude_name_patterns: Optional[List[str]] = None,
    include_types: Optional[List[str]] = None,   # class name strings (e.g., ["Linear","LayerNorm"])
    exclude_types: Optional[List[str]] = None,
):
    """
    on_record(rec: Dict) is called for each (module_name, call_idx) with:
      {
        "module_name","module_type","call_idx",
        "input_dtypes","output_dtypes","input_shapes","output_shapes",
        "inputs","outputs"
      }
    """
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
                    "inputs": extract_tensors(inputs),
                    "outputs": extract_tensors(outputs),
                }
                on_record(rec)
            return hook

        handles.append(mod.register_forward_hook(make_hook(name, mod)))

    return handles

class CaptureWriter:
    """
    Writes records either to memory (single .pt) or to a directory of shards.
    """
    def __init__(self, out_path: str, out_format: str, meta: Dict[str, Any]):
        assert out_format in ("pt", "dir"), "out_format must be 'pt' or 'dir'"
        self.out_path = out_path
        self.out_format = out_format
        self.meta = meta
        self.records: List[Dict[str, Any]] = []
        self.count = 0

        if out_format == "dir":
            os.makedirs(out_path, exist_ok=True)
            with open(os.path.join(out_path, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            self.manifest = []
        else:
            # single file -> keep in memory, save once at end
            self.manifest = None

    def add(self, rec: Dict[str, Any]):
        if self.out_format == "dir":
            fn = f"rec_{self.count:06d}.pt"
            torch.save(rec, os.path.join(self.out_path, fn))
            self.manifest.append({"idx": self.count,
                                  "module_name": rec["module_name"],
                                  "call_idx": rec["call_idx"],
                                  "file": fn})
        else:
            self.records.append(rec)
        self.count += 1

    def close(self):
        if self.out_format == "dir":
            with open(os.path.join(self.out_path, "manifest.jsonl"), "w") as f:
                for row in self.manifest:
                    f.write(json.dumps(row) + "\n")
        else:
            payload = {"meta": self.meta, "records": self.records}
            os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
            torch.save(payload, self.out_path)

def capture_forward_pass(
    model: nn.Module,
    example_inputs: Any,             # you prepare (already on correct device/dtypes)
    out_path: str,                   # "/content/capture_bf16.pt" or "/content/capture_bf16_dir"
    out_format: str = "pt",          # "pt" (single) or "dir" (sharded)
    *,
    run_tag: Optional[str] = None,   # e.g., "bf16", "fp16", "mixed"
    set_eval: bool = True,
    no_grad: bool = True,
    capture_leaves_only: bool = False,
    include_name_patterns: Optional[List[str]] = None,
    exclude_name_patterns: Optional[List[str]] = None,
    include_types: Optional[List[str]] = None,
    exclude_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Registers hooks, runs one forward, writes capture. Does NOT cast or move model/inputs.
    """
    if set_eval:
        model.eval()

    meta = {
        "run_tag": run_tag,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        "capture_leaves_only": capture_leaves_only,
        "filters": {
            "include_name_patterns": include_name_patterns,
            "exclude_name_patterns": exclude_name_patterns,
            "include_types": include_types,
            "exclude_types": exclude_types,
        },
        "param_dtype_counts": summarize_param_dtypes(model),
    }

    writer = CaptureWriter(out_path=out_path, out_format=out_format, meta=meta)
    handles = register_forward_hooks(
        model,
        on_record=writer.add,
        capture_leaves_only=capture_leaves_only,
        include_name_patterns=include_name_patterns,
        exclude_name_patterns=exclude_name_patterns,
        include_types=include_types,
        exclude_types=exclude_types,
    )

    ctx = contextlib.nullcontext()
    if no_grad:
        ctx = torch.no_grad()

    with ctx:
        # Call the model with your provided inputs (no casting/moving).
        if isinstance(example_inputs, dict):
            _ = model(**example_inputs)
        elif isinstance(example_inputs, (list, tuple)):
            _ = model(*example_inputs)
        else:
            _ = model(example_inputs)

    for h in handles:
        h.remove()

    writer.close()
    print(f"Saved {writer.count} module call records to {out_path} (format={out_format})")
    return {"meta": meta, "num_records": writer.count}


import glob
import pandas as pd

def load_capture_any(path: str) -> Dict[str, Any]:
    """
    Supports:
      - Single file: .pt with {"meta":..., "records":[...]}
      - Directory: contains meta.json and rec_*.pt (+ manifest.jsonl)
    """
    if os.path.isdir(path):
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)
        rec_files = sorted(glob.glob(os.path.join(path, "rec_*.pt")))
        records = [torch.load(fn, map_location="cpu") for fn in rec_files]
        return {"meta": meta, "records": records}
    else:
        return torch.load(path, map_location="cpu")

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
            for k in (t.keys() if isinstance(t, dict) else []):
                rec(t[k],
                    dt.get(k) if isinstance(dt, dict) else None,
                    sh.get(k) if isinstance(sh, dict) else None)
        # else: ignore
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

def compare_captures(
    path_a: str,   # e.g., bf16 capture
    path_b: str,   # e.g., fp16 capture
    *,
    out_csv: Optional[str] = None,
    compare_inputs: bool = True,
    compare_outputs: bool = True,
):
    """
    Aligns by (module_name, call_idx), compares float tensors only.
    """
    cap_a = load_capture_any(path_a)
    cap_b = load_capture_any(path_b)
    recs_a = cap_a["records"]
    recs_b = cap_b["records"]

    key = lambda r: (r["module_name"], r["call_idx"])
    map_a = {key(r): r for r in recs_a}
    map_b = {key(r): r for r in recs_b}

    shared_keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    missing_a = sorted(set(map_b.keys()) - set(map_a.keys()))
    missing_b = sorted(set(map_a.keys()) - set(map_b.keys()))
    if missing_a:
        print(f"Warning: {len(missing_a)} calls only in B (skipped).")
    if missing_b:
        print(f"Warning: {len(missing_b)} calls only in A (skipped).")

    rows = []

    def push_rows(ra, rb, io_label: str):
        a_list = _flatten_tensors_with_meta(ra[io_label], ra[f"{io_label[:-1]}_dtypes"], ra[f"{io_label[:-1]}_shapes"])
        b_list = _flatten_tensors_with_meta(rb[io_label], rb[f"{io_label[:-1]}_dtypes"], rb[f"{io_label[:-1]}_shapes"])
        n = min(len(a_list), len(b_list))
        if len(a_list) != len(b_list):
            print(f"Note: {ra['module_name']}[{ra['call_idx']}] {io_label} count differs: {len(a_list)} vs {len(b_list)}. Using {n}.")
        for i in range(n):
            ta, dta, sha = a_list[i]
            tb, dtb, shb = b_list[i]
            if not (is_float_tensor(ta) and is_float_tensor(tb)):
                continue
            stats = _compare_one_pair_fp32(ta, tb)
            rows.append({
                "module_name": ra["module_name"],
                "module_type": ra["module_type"],
                "call_idx": ra["call_idx"],
                "io": "input" if io_label == "inputs" else "output",
                "tensor_index": i,
                "shape": str(sha if sha is not None else tuple(ta.shape)),
                "dtype_a": dta,
                "dtype_b": dtb,
                **stats,
            })

    for k in shared_keys:
        ra, rb = map_a[k], map_b[k]
        if compare_inputs:
            push_rows(ra, rb, "inputs")
        if compare_outputs:
            push_rows(ra, rb, "outputs")

    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"Wrote CSV with {len(df)} rows -> {out_csv}")

    if len(df) > 0:
        by_mod = (df
                  .groupby(["module_name", "module_type", "io"], as_index=False)
                  .agg(mae32_mean=("mae32", "mean"),
                       maxae32_max=("maxae32", "max"),
                       mse32_mean=("mse32", "mean"),
                       cosine_mean=("cosine", "mean"),
                       numel_sum=("numel", "sum"))
                 )
    else:
        by_mod = df

    return df, by_mod
