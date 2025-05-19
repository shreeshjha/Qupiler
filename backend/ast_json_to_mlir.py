import json
import sys
from xdsl.context import Context
from xdsl.ir import Block, Region
from xdsl.dialects.builtin import IntegerAttr, StringAttr, ModuleOp, i1, i32
from xdsl.dialects.func import ReturnOp
from quantum_dialect import (
    register_quantum_dialect,
    QuantumInitOp, QuantumNotOp, QuantumCNOTOp, QuantumMeasureOp,
    QuantumAddOp, QuantumSubOp, QuantumMulOp, QuantumDivOp,
    QuantumXorOp, QuantumAndOp, QuantumOrOp, QuantumFuncOp
)

# SSA name generator
ssa_counter = 0
def next_ssa():
    global ssa_counter
    ssa_counter += 1
    return f"%q{ssa_counter}"

# Map C variable names to SSAValue
ssa_map = {}

# Build quantum.func
def build_quantum_func(ctx: Context, name: str) -> QuantumFuncOp:
    func = QuantumFuncOp(
        attributes={"func_name": StringAttr(name)},
        regions=[Region()]
    )
    entry_block = Block()
    func.regions[0].add_block(entry_block)
    return func

# Find function body recursively
def find_func_body(node, target_name: str):
    if node.get("kind") == "FunctionDecl" and node.get("name") == target_name:
        for child in node.get("inner", []):
            if child.get("kind") == "CompoundStmt":
                return child.get("inner", [])
        return []
    for child in node.get("inner", []):
        res = find_func_body(child, target_name)
        if res:
            return res
    return []

# Translate statements
def translate_stmt(stmt, block):
    kind = stmt.get("kind")
    if kind == "DeclStmt":
        for decl in stmt.get("inner", []):
            if decl.get("kind") != "VarDecl": continue
            var = decl.get("name")
            init_val = 0
            # find initializer expr: BinaryOperator or IntegerLiteral
            expr = None
            for child in decl.get("inner", []):
                if child.get("kind") in ("BinaryOperator", "IntegerLiteral"):
                    expr = child
                    break
            if expr:
                k2 = expr.get("kind")
                if k2 == "IntegerLiteral":
                    init_val = int(expr.get("value", 0))
                elif k2 == "BinaryOperator":
                    children = expr.get("inner", [])
                    refs = [c.get("inner", [])[0].get("name") for c in children if c.get("kind") == "DeclRefExpr"]
                    if len(refs) == 2:
                        lhs, rhs = refs
                        op = expr.get("opcode")
                        arith = {"+": QuantumAddOp, "-": QuantumSubOp, "*": QuantumMulOp, "/": QuantumDivOp}
                        logic = {"^": QuantumXorOp, "&&": QuantumAndOp, "||": QuantumOrOp}
                        if op in arith and lhs in ssa_map and rhs in ssa_map:
                            o = arith[op](result_types=[i32], operands=[ssa_map[lhs], ssa_map[rhs]])
                            block.add_op(o); ssa_map[var] = o.results[0]; continue
                        if op in logic and lhs in ssa_map and rhs in ssa_map:
                            o = logic[op](result_types=[i1], operands=[ssa_map[lhs], ssa_map[rhs]])
                            block.add_op(o); ssa_map[var] = o.results[0]; continue
            init_op = QuantumInitOp(result_types=[i32], attributes={"type": i32, "value": IntegerAttr(init_val, i32)})
            block.add_op(init_op)
            ssa_map[var] = init_op.results[0]
    elif kind == "CallExpr" and stmt.get("callee", {}).get("name") == "printf":
        args = stmt.get("args", [])
        if args:
            for c in args[0].get("inner", []):
                if c.get("kind") == "DeclRefExpr":
                    nm = c.get("inner", [])[0].get("name")
                    if nm in ssa_map:
                        m = QuantumMeasureOp(result_types=[i1], operands=[ssa_map[nm]])
                        block.add_op(m)
                    break

# Main
if __name__ == "__main__":
    if len(sys.argv)!=3: print("Usage: ..."); sys.exit(1)
    ast_file, mlir_out = sys.argv[1], sys.argv[2]
    ast = json.load(open(ast_file))
    ctx = Context(); register_quantum_dialect(ctx)
    module = ModuleOp([])
    if not module.regions: module.add_region(Region())
    if not module.regions[0].blocks: module.regions[0].add_block(Block())
    func = build_quantum_func(ctx, "quantum_circuit")
    blk = func.regions[0].blocks[0]
    for st in find_func_body(ast, "quantum_circuit"): translate_stmt(st, blk)
    blk.add_op(ReturnOp()); module.regions[0].blocks[0].add_op(func)
    with open(mlir_out, "w") as f: f.write(str(module))

