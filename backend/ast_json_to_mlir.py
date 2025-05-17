import json
import sys
from xdsl.context import Context
from xdsl.ir import Block, Region
from xdsl.dialects.builtin import IntegerAttr, StringAttr, ModuleOp, i1, i32
from xdsl.dialects.func import ReturnOp
from quantum_dialect import (
    QuantumInitOp, QuantumNotOp, QuantumCNOTOp,
    QuantumMeasureOp, QuantumFuncOp, register_quantum_dialect
)

ssa_counter = 0

def next_ssa():
    global ssa_counter
    ssa_counter += 1
    return f"%q{ssa_counter}"

ssa_map = {}

# === Helpers === #
def build_basic_quantum_func(ctx: Context, func_name: str) -> QuantumFuncOp:
    func = QuantumFuncOp(
        attributes={"func_name": StringAttr(func_name)},
        regions=[Region()]
    )
    block = Block()
    func.regions[0].add_block(block)
    return func


def find_func_body(node, name):
    """
    Recursively search for FunctionDecl named `name` and return its CompoundStmt inner list.
    """
    if node.get("kind") == "FunctionDecl" and node.get("name") == name:
        for child in node.get("inner", []):
            if child.get("kind") == "CompoundStmt":
                return child.get("inner", [])
        return []
    for child in node.get("inner", []):
        result = find_func_body(child, name)
        if result:
            return result
    return []
    for child in node.get("inner", []):
        result = find_func_body(child, name)
        if result:
            return result
    return []


def translate_stmt(stmt, block):
    kind = stmt.get('kind')
    if kind == "DeclStmt":
        # Variable declaration with optional initializer
        decl = stmt.get("inner", [])[0]
        name = decl.get("name")
        val = 0
        for init in decl.get("inner", []):
            if init.get("kind") == "IntegerLiteral":
                val = int(init.get("value", 0))
        init_op = QuantumInitOp(result_types=[i32], attributes={"type": i32, "value": IntegerAttr(val, i32)})
        block.add_op(init_op)
        ssa_map[name] = init_op.results[0]

    elif kind == "BinaryOperator":
        opcode = stmt.get("opcode")
        if opcode == "-":
            # x - y
            children = stmt.get("inner", [])
            lhs_ref = children[0].get("inner", [])[0].get("name")
            rhs_ref = children[1].get("inner", [])[0].get("name")
            # simulate subtraction as not + cnot
            not_rhs = QuantumNotOp(ssa_map[rhs_ref])
            block.add_op(not_rhs)
            cnot = QuantumCNOTOp(ssa_map[lhs_ref], not_rhs.results[0])
            block.add_op(cnot)
            temp_name = next_ssa()
            ssa_map[temp_name] = cnot.results[0]

    elif kind == "CallExpr":
        callee = stmt.get("callee", {}).get("name")
        if callee == "printf":
            args = stmt.get("args", [])
            if args:
                arg_node = args[0]
                # drill into DeclRefExpr
                name = arg_node.get("inner", [])[0].get("name")
                meas = QuantumMeasureOp(ssa_map[name])
                block.add_op(meas)


# === Entry === #
def main():
    if len(sys.argv) != 3:
        print("Usage: python ast_json_to_mlir.py <input.json> <output.mlir>")
        sys.exit(1)

    input_json = sys.argv[1]
    output_mlir = sys.argv[2]

    with open(input_json) as f:
        ast = json.load(f)

    ctx = Context()
    register_quantum_dialect(ctx)

    module = ModuleOp([])
    if not module.regions:
        module.add_region(Region())
    if not module.regions[0].blocks:
        module.regions[0].add_block(Block())

    func = build_basic_quantum_func(ctx, "quantum_circuit")
    block = func.regions[0].blocks[0]

    stmts = find_func_body(ast, "quantum_circuit")
    for stmt in stmts:
        translate_stmt(stmt, block)

    block.add_op(ReturnOp())

    # Insert the function into the module
    module.regions[0].blocks[0].add_op(func)

    with open(output_mlir, "w") as f:
        f.write(str(module))


if __name__ == "__main__":
    main()

