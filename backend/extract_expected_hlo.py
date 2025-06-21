#!/usr/bin/env python3
import re, sys, operator

# Binary ops → Python functions
BIN_OPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,        # <--- fixed here
    "div": operator.floordiv,
    "mod": operator.mod,
    "and": operator.and_,
    "or":  operator.or_,
    "xor": operator.xor,
}

# Unary ops → Python functions
UN_OPS = {
    "not": lambda a: ~a,
    "neg": operator.neg,
}

MEASURE_OP = "measure"

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <hlo-file.mlir>", file=sys.stderr)
    sys.exit(1)

lines = open(sys.argv[1]).read().splitlines()

# 1) Read all quantum.init constants
init_re = re.compile(r'^\s*%(\w+)\s*=\s*"quantum\.init"\(\)\s*\{[^}]*value\s*=\s*([-]?\d+)')
values = {}
for L in lines:
    m = init_re.match(L)
    if m:
        reg, val = m.groups()
        values[f"%{reg}"] = int(val)

if not values:
    sys.exit("ERROR: no quantum.init constants found")

# 2) Replay each quantum.<op> in program order
op_re = re.compile(
    r'^\s*(?P<dests>%\w+(?:\s*,\s*%\w+)*)\s*=\s*"quantum\.(?P<op>\w+)"\s*\(\s*(?P<args>[%\w,\s]+)\)'
)

measured = None
for L in lines:
    m = op_re.match(L)
    if not m:
        continue

    dests = [d.strip() for d in m.group("dests").split(",")]
    op    = m.group("op")
    regs  = [r.strip() for r in m.group("args").split(",") if r.strip()]

    if op == MEASURE_OP:
        src = regs[0]
        if src not in values:
            sys.exit(f"ERROR: measure of unknown register {src}")
        measured = values[src]
        break

    if op in BIN_OPS and len(regs) == 2:
        a, b = regs
        values[dests[0]] = BIN_OPS[op](values[a], values[b])
    elif op in UN_OPS and len(regs) == 1:
        src = regs[0]
        values[dests[0]] = UN_OPS[op](values[src])
    else:
        # skip post_inc, post_dec, alloc, circuit‐ops, etc.
        continue

# 3) Fallback to last dest if no measure
if measured is None:
    all_dests = [d for m in op_re.finditer("\n".join(lines))
                   for d in m.group("dests").split(",")]
    for reg in reversed(all_dests):
        reg = reg.strip()
        if reg in values:
            measured = values[reg]
            break

if measured is None:
    sys.exit("ERROR: no measurement or computable result found")

print(measured)

