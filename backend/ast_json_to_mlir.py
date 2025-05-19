# Process a CallExpr to find the printf arguments
def process_printf_args(call_expr_node):
    debug_print("Processing printf call")
    
    args = []
    # Find the args in the CallExpr
    for child in call_expr_node.get("inner", []):
        if child.get("kind") != "ImplicitCastExpr" or "printf" in str(child):
            continue
        
        # Skip format string arg
        if "char" in str(child.get("type", {})):
            continue
            
        var_name = extract_var_name(child)
        if var_name:
            args.append(var_name)
            
    return args# ast_json_to_mlir.py
import json, sys
from xdsl.context import Context
from xdsl.ir import Block, Region
from xdsl.dialects.builtin import IntegerAttr, StringAttr, ModuleOp, i1, i32
from xdsl.dialects.func import ReturnOp
from quantum_dialect import (
    register_quantum_dialect,
    QuantumInitOp, QuantumMeasureOp,
    QuantumAddOp, QuantumSubOp, QuantumMulOp, QuantumDivOp,
    QuantumXorOp, QuantumAndOp, QuantumOrOp,
    QuantumFuncOp
)

# Enable debug output if needed
DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# Map C variable names to SSA values
ssa_map = {}

# Build a quantum.func with one entry block
def build_quantum_func(ctx: Context, name: str) -> QuantumFuncOp:
    fn = QuantumFuncOp(
        attributes={"func_name": StringAttr(name)},
        regions=[Region()]
    )
    blk = Block()
    fn.regions[0].add_block(blk)
    return fn

# Recursively find the CompoundStmt body of a FunctionDecl
def find_body(node, target_name: str):
    if node.get("kind") == "FunctionDecl" and node.get("name") == target_name:
        for child in node.get("inner", []):
            if child.get("kind") == "CompoundStmt":
                debug_print(f"Found body for {target_name}")
                return child.get("inner", [])
        return []
    for child in node.get("inner", []):
        res = find_body(child, target_name)
        if res:
            return res
    return []

# Extract the actual variable name from a node that might be nested in cast expressions
def extract_var_name(node):
    if node.get("kind") == "DeclRefExpr":
        # Direct reference
        inner = node.get("inner", [])
        if inner and len(inner) > 0:
            return inner[0].get("name")
        # Try referencedDecl
        ref_decl = node.get("referencedDecl", {})
        if ref_decl:
            return ref_decl.get("name")
    # Handle ImplicitCastExpr
    elif node.get("kind") == "ImplicitCastExpr":
        inner = node.get("inner", [])
        if inner and len(inner) > 0:
            return extract_var_name(inner[0])
    # Recursive for other nested structures
    inner = node.get("inner", [])
    for child in inner:
        result = extract_var_name(child)
        if result:
            return result
    return None

# Extract variable references from BinaryOperator handling cast expressions
def extract_binop_refs(binop_node):
    refs = []
    for child in binop_node.get("inner", []):
        var_name = extract_var_name(child)
        if var_name:
            refs.append(var_name)
    return refs

# Translate a single statement into quantum MLIR ops
def translate_stmt(stmt, blk):
    kind = stmt.get("kind")
    debug_print(f"Processing statement of kind: {kind}")
    
    if kind == "DeclStmt":
        debug_print("Processing declaration statement")
        for decl in stmt.get("inner", []):
            if decl.get("kind") != "VarDecl":
                continue
            var = decl.get("name")
            debug_print(f"Processing variable declaration: {var}")
            
            # find initializer
            init_node = None
            for ch in decl.get("inner", []):
                if ch.get("kind") in ("IntegerLiteral", "BinaryOperator"):
                    init_node = ch
                    debug_print(f"Found initializer: {ch.get('kind')}")
                    break
            
            # integer literal
            if init_node and init_node["kind"] == "IntegerLiteral":
                val = int(init_node.get("value", "0"))
                debug_print(f"Initializing {var} with literal value {val}")
                op = QuantumInitOp(
                    result_types=[i32],
                    attributes={"type": i32, "value": IntegerAttr(val, i32)}
                )
                blk.add_op(op)
                ssa_map[var] = op.results[0]
                continue
                
            # binary operator
            if init_node and init_node.get("kind") == "BinaryOperator":
                debug_print(f"Processing binary op initializer for {var}")
                opcode = init_node.get("opcode")
                debug_print(f"Binary operator opcode: {opcode}")
                
                # Extract variable references, handling implicit casts
                var_refs = extract_binop_refs(init_node)
                debug_print(f"Found variable references: {var_refs}")
                
                if len(var_refs) == 2 and all(r in ssa_map for r in var_refs):
                    lhs_var, rhs_var = var_refs
                    debug_print(f"Creating binary op {opcode} with {lhs_var} and {rhs_var}")
                    arith_map = {"+": QuantumAddOp, "-": QuantumSubOp, 
                                "*": QuantumMulOp, "/": QuantumDivOp}
                    logic_map = {"^": QuantumXorOp, "&&": QuantumAndOp, 
                                "||": QuantumOrOp}
                    
                    if opcode in arith_map:
                        o = arith_map[opcode](
                            result_types=[i32], 
                            operands=[ssa_map[lhs_var], ssa_map[rhs_var]]
                        )
                        blk.add_op(o)
                        ssa_map[var] = o.results[0]
                        continue
                        
                    if opcode in logic_map:
                        o = logic_map[opcode](
                            result_types=[i1], 
                            operands=[ssa_map[lhs_var], ssa_map[rhs_var]]
                        )
                        blk.add_op(o)
                        ssa_map[var] = o.results[0]
                        continue
            
            # fallback zero init
            debug_print(f"Fallback initialization for {var} with value 0")
            op = QuantumInitOp(
                result_types=[i32],
                attributes={"type": i32, "value": IntegerAttr(0, i32)}
            )
            blk.add_op(op)
            ssa_map[var] = op.results[0]
            
    elif kind == "BinaryOperator":
        # Handle standalone assignment
        if stmt.get("opcode") == "=":
            lhs_var = extract_var_name(stmt.get("inner", [])[0]) if len(stmt.get("inner", [])) > 0 else None
            rhs_node = stmt.get("inner", [])[1] if len(stmt.get("inner", [])) > 1 else None
            
            if lhs_var and rhs_node and rhs_node.get("kind") == "BinaryOperator":
                var_refs = extract_binop_refs(rhs_node)
                if len(var_refs) == 2 and all(r in ssa_map for r in var_refs):
                    op1, op2 = var_refs
                    opcode = rhs_node.get("opcode")
                    arith_map = {"+": QuantumAddOp, "-": QuantumSubOp, 
                                "*": QuantumMulOp, "/": QuantumDivOp}
                    logic_map = {"^": QuantumXorOp, "&&": QuantumAndOp, 
                                "||": QuantumOrOp}
                    
                    if opcode in arith_map:
                        o = arith_map[opcode](
                            result_types=[i32], 
                            operands=[ssa_map[op1], ssa_map[op2]]
                        )
                        blk.add_op(o)
                        ssa_map[lhs_var] = o.results[0]
            
    elif kind == "CallExpr" and (stmt.get("callee",{}).get("name") == "printf" or 
                              "printf" in str(stmt.get("inner", [0]))):
        var_names = process_printf_args(stmt)
        for var_name in var_names:
            if var_name in ssa_map:
                debug_print(f"Adding measurement for {var_name}")
                m = QuantumMeasureOp(result_types=[i1], operands=[ssa_map[var_name]])
                blk.add_op(m)
                break

# Main
def main():
    if len(sys.argv)!=3:
        print("Usage: python ast_json_to_mlir.py <input.json> <output.mlir>")
        sys.exit(1)
        
    ast = json.load(open(sys.argv[1]))
    debug_print("Loaded AST from", sys.argv[1])
    
    ctx = Context()
    register_quantum_dialect(ctx)
    
    mod = ModuleOp([])
    if not mod.regions: 
        mod.add_region(Region())
    if not mod.regions[0].blocks: 
        mod.regions[0].add_block(Block())
    
    fn = build_quantum_func(ctx, "quantum_circuit")
    blk = fn.regions[0].blocks[0]
    
    # Find and process the body of the quantum_circuit function
    body = find_body(ast, "quantum_circuit")
    debug_print(f"Found {len(body)} statements in quantum_circuit")
    
    # Process each statement
    for idx, stmt in enumerate(body):
        debug_print(f"\nProcessing statement {idx+1}/{len(body)}")
        translate_stmt(stmt, blk)
    
    # Add return and finalize module
    blk.add_op(ReturnOp())
    mod.regions[0].blocks[0].add_op(fn)
    
    # Write output
    with open(sys.argv[2], "w") as f:
        f.write(str(mod))
    debug_print("MLIR output written to", sys.argv[2])
    
    # Debug: show final SSA map
    debug_print("\nFinal SSA map:")
    for var, val in ssa_map.items():
        debug_print(f"  {var} -> {val}")

if __name__ == "__main__": 
    main()
