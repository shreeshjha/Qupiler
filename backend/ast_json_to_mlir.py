# ast_json_to_mlir.py
import json, sys
from xdsl.context import Context
from xdsl.ir import Block, Region, BlockArgument
from xdsl.dialects.builtin import IntegerAttr, StringAttr, ModuleOp, i1, i32
from xdsl.dialects.func import ReturnOp

from quantum_dialect import (
    register_quantum_dialect,
    QuantumInitOp, QuantumMeasureOp,
    QuantumAddOp, QuantumSubOp, QuantumMulOp, QuantumDivOp,
    QuantumNegOp,
    QuantumNotOp,
    QuantumXorOp, QuantumAndOp, QuantumOrOp,
    QuantumFuncOp,
    QuantumIncrementOp, QuantumDecrementOp,
    QuantumPreIncrementOp, QuantumPostIncrementOp,
    QuantumPreDecrementOp, QuantumPostDecrementOp,
    QuantumModOp,
    QuantumLeftShiftOp, QuantumRightShiftOp,
    QuantumLessThanOp, QuantumGreaterThanOp,
    QuantumEqualOp, QuantumNotEqualOp,
    QuantumLessThanEqualOp, QuantumGreaterThanEqualOp,
    QuantumWhileOp, QuantumConditionOp, QuantumIfOp
)

# Enable debug output if needed
DEBUG = True

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# Map C variable names to SSA values
ssa_map = {}

# Used to track changes to SSA variables inside control flow blocks
ssa_changes = []

# SSA variable scoping stack to handle nested scopes 
scopes = []

# Track variables modified in loops
loop_modified_vars = set()

# Flag to track if we're currently processing a loop body
in_loop_body = False

# Global current block tracker
current_block = None

# Save the current SSA state when entering a new scope
def push_scope():
    scopes.append(ssa_map.copy())

# Restore the SSA state when exiting a scope 
def pop_scope():
    global ssa_map 
    if scopes:
        ssa_map = scopes.pop()  # Fixed: was 'scores.pop()'

# Update SSA map with changes from a nested scope
def merge_ssa_changes(changes):
    for var, val in changes.items():
        ssa_map[var] = val

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


# Extract a compound statement body
def extract_compound_body(node):
    if node.get("kind") == "CompoundStmt":
        return node.get("inner", [])
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
            
    return args

# Determine if a unary operator is postfix
def is_postfix_operator(node):
    # Dump the node for debugging
    if DEBUG:
        debug_print(f"Checking if operator is postfix: {node.get('opcode')}")
        debug_print(f"Node details: {node}")
    
    # Check for isPostfix attribute in AST
    if "isPostfix" in node:
        return node.get("isPostfix") == True
    
    # Alternative detection - check for valueCategory or tokens
    # Some AST formats might indicate postfix ops differently
    if "valueCategory" in node and node.get("valueCategory") == "postfix":
        return True
    
    # If none of the above, use opcode as fallback
    opcode = node.get("opcode")
    return opcode in ["postinc", "postdec"] or opcode.startswith("post")


# Generate code for a condition expression (used in while loops and if statements)
def generate_condition(expr, blk):
    debug_print(f"Generating condition for expression type: {expr.get('kind')}")
    
    if expr.get("kind") == "BinaryOperator":
        opcode = expr.get("opcode")
        debug_print(f"Condition opcode: {opcode}")
        
        var_refs = extract_binop_refs(expr)
        debug_print(f"Condition variables: {var_refs}")
        
        if len(var_refs) == 2 and all(r in ssa_map for r in var_refs):
            lhs_var, rhs_var = var_refs
            
            cmp_map = {
                "<": QuantumLessThanOp,
                ">": QuantumGreaterThanOp,
                "==": QuantumEqualOp,
                "!=": QuantumNotEqualOp,
                "<=": QuantumLessThanEqualOp,
                ">=": QuantumGreaterThanEqualOp
            }
            
            logic_map = {
                "&&": QuantumAndOp,
                "||": QuantumOrOp
            }
            
            if opcode in cmp_map:
                op = cmp_map[opcode](
                    result_types=[i1],
                    operands=[ssa_map[lhs_var], ssa_map[rhs_var]]
                )
                blk.add_op(op)
                return op.results[0]
            
            elif opcode in logic_map:
                op = logic_map[opcode](
                    result_types=[i1],
                    operands=[ssa_map[lhs_var], ssa_map[rhs_var]]
                )
                blk.add_op(op)
                return op.results[0]
    
    # Fallback case: If we can't parse the condition properly
    debug_print("Could not generate condition properly, using default")
    # Return a default true condition
    op = QuantumInitOp(
        result_types=[i1],
        attributes={"type": i1, "value": IntegerAttr(1, i1)}
    )
    blk.add_op(op)
    return op.results[0]

# Process a WhileStmt node and create a simplified representation
def process_while_stmt(stmt, blk):
    """
    Process while loop with improved loop-carried value tracking
    """
    global current_block, loop_modified_vars, in_loop_body
    
    # Extract condition and body
    inner = stmt.get("inner", [])
    if len(inner) < 2:
        debug_print("Warning: WhileStmt doesn't have enough children")
        return
        
    cond_node = inner[0]
    body_node = inner[1]
    
    debug_print("Processing while loop with improved variable tracking")
    
    # Step 1: Identify variables used in condition
    cond_vars = set()
    if cond_node.get("kind") == "BinaryOperator":
        var_refs = extract_binop_refs(cond_node)
        cond_vars.update(var_refs)
    
    debug_print(f"Variables used in condition: {cond_vars}")
    
    # Step 2: Find variables modified in body (dry run)
    loop_modified_vars = set()
    
    if body_node.get("kind") == "CompoundStmt":
        stmts = body_node.get("inner", [])
    else:
        stmts = [body_node]
    
    # Analyze body statements to find modifications
    for stmt in stmts:
        if stmt.get("kind") == "BinaryOperator" and stmt.get("opcode") == "=":
            lhs_var = extract_var_name(stmt.get("inner", [])[0]) if len(stmt.get("inner", [])) > 0 else None
            if lhs_var:
                loop_modified_vars.add(lhs_var)
                debug_print(f"Variable modified in loop: {lhs_var}")
    
    # Step 3: Create while operation (using existing dialect)
    while_op = QuantumWhileOp(regions=[Region(), Region()])
    
    # Create condition and body blocks
    cond_block = Block()
    body_block = Block()
    
    while_op.regions[0].add_block(cond_block)
    while_op.regions[1].add_block(body_block)
    
    # Step 4: Generate condition (same as before)
    debug_print("Generating condition for while loop")
    cond_result = generate_condition(cond_node, cond_block)
    cond_op = QuantumConditionOp(operands=[cond_result])
    cond_block.add_op(cond_op)
    
    # Step 5: Process body statements
    debug_print(f"Processing {len(stmts)} statements in while loop body")
    
    push_scope()
    old_current_block = current_block
    current_block = body_block
    in_loop_body = True
    
    for stmt in stmts:
        debug_print(f"Processing body statement: {stmt.get('kind')}")
        translate_stmt(stmt, body_block)
    
    current_block = old_current_block
    in_loop_body = False
    pop_scope()
    
    # Step 6: Add while operation to current block
    blk.add_op(while_op)
    
    # Step 7: Create post-loop values for modified variables
    debug_print(f"Creating post-loop values for modified variables: {loop_modified_vars}")
    
    for var in loop_modified_vars:
        if var in ssa_map:
            debug_print(f"Creating post-loop representation for {var}")
            
            # Create a symbolic post-loop value
            # This represents the value of the variable after the loop has executed
            # In a full implementation, this would be derived from the loop's execution
            post_loop_op = QuantumInitOp(
                result_types=[i32],
                attributes={
                    "type": i32, 
                    "value": IntegerAttr(0, i32)  # Symbolic value representing "result after loop"
                }
            )
            blk.add_op(post_loop_op)
            
            # Update SSA map
            old_val = ssa_map[var]
            ssa_map[var] = post_loop_op.results[0]
            debug_print(f"Updated {var}: {old_val} -> {ssa_map[var]} (post-loop)")
    
    debug_print("Finished processing while loop with improved tracking")

# Translate a single statement into quantum MLIR ops
def translate_stmt(stmt, blk):
    kind = stmt.get("kind")
    debug_print(f"Processing statement of kind: {kind}")
    
    # Handle While statements
    if kind == "WhileStmt":
        process_while_stmt(stmt, blk)
        return  # Added explicit return after handling while

    elif kind == "DeclStmt":
        debug_print("Processing declaration statement")
        for decl in stmt.get("inner", []):
            if decl.get("kind") != "VarDecl":
                continue
            var = decl.get("name")
            debug_print(f"Processing variable declaration: {var}")
            
            # find initializer
            init_node = None
            for ch in decl.get("inner", []):
                if ch.get("kind") in ("IntegerLiteral", "BinaryOperator", "UnaryOperator", "DeclRefExpr", "ImplicitCastExpr"):
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
            
            # variable reference (e.g., int sum = x;)
            elif init_node and (init_node.get("kind") == "DeclRefExpr" or init_node.get("kind") == "ImplicitCastExpr"):
                src_var = extract_var_name(init_node)
                if src_var and src_var in ssa_map:
                    debug_print(f"Initializing {var} from variable {src_var}")
                    ssa_map[var] = ssa_map[src_var]  # Direct assignment
                else:
                    debug_print(f"Could not find source variable {src_var}, using fallback initialization")
                    # fallback zero init
                    op = QuantumInitOp(
                        result_types=[i32],
                        attributes={"type": i32, "value": IntegerAttr(0, i32)}
                    )
                    blk.add_op(op)
                    ssa_map[var] = op.results[0]
                
            # binary operator
            elif init_node and init_node.get("kind") == "BinaryOperator":
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
                                "*": QuantumMulOp, "/": QuantumDivOp,
                                "%": QuantumModOp}
                    logic_map = {"^": QuantumXorOp, "&&": QuantumAndOp, 
                                "||": QuantumOrOp}
                    shift_map = {"<<": QuantumLeftShiftOp, ">>": QuantumRightShiftOp}
                    cmp_map = {"<": QuantumLessThanOp, ">": QuantumGreaterThanOp,
                              "==": QuantumEqualOp, "!=": QuantumNotEqualOp,
                              "<=": QuantumLessThanEqualOp, ">=": QuantumGreaterThanEqualOp}
                    
                    if opcode in arith_map:
                        o = arith_map[opcode](
                            result_types=[i32], 
                            operands=[ssa_map[lhs_var], ssa_map[rhs_var]]
                        )
                        blk.add_op(o)
                        ssa_map[var] = o.results[0]
                        
                    elif opcode in logic_map:
                        o = logic_map[opcode](
                            result_types=[i1], 
                            operands=[ssa_map[lhs_var], ssa_map[rhs_var]]
                        )
                        blk.add_op(o)
                        ssa_map[var] = o.results[0]
                        
                    elif opcode in shift_map:
                        o = shift_map[opcode](
                            result_types=[i32], 
                            operands=[ssa_map[lhs_var], ssa_map[rhs_var]]
                        )
                        blk.add_op(o)
                        ssa_map[var] = o.results[0]
                        
                    elif opcode in cmp_map:
                        o = cmp_map[opcode](
                            result_types=[i1], 
                            operands=[ssa_map[lhs_var], ssa_map[rhs_var]]
                        )
                        blk.add_op(o)
                        ssa_map[var] = o.results[0]
                else:
                    # fallback zero init if we can't process binary op
                    debug_print(f"Could not process binary op for {var}, using fallback")
                    op = QuantumInitOp(
                        result_types=[i32],
                        attributes={"type": i32, "value": IntegerAttr(0, i32)}
                    )
                    blk.add_op(op)
                    ssa_map[var] = op.results[0]
            
            # unary operator
            elif init_node and init_node.get("kind") == "UnaryOperator":
                debug_print(f"Processing unary op initializer for {var}")
                opcode = init_node.get("opcode")
                debug_print(f"Unary operator opcode: {opcode}")
                
                # Check if it's a postfix or prefix operator
                is_postfix = is_postfix_operator(init_node)
                debug_print(f"Is postfix operator: {is_postfix}")
                
                # Extract the operand
                operand_node = init_node.get("inner", [])[0] if len(init_node.get("inner", [])) > 0 else None
                operand_var = extract_var_name(operand_node)
                
                if operand_var and operand_var in ssa_map:
                    debug_print(f"Found operand: {operand_var}")
                   
                    # Handle unary minus (negation)
                    if opcode == "-" or opcode == "minus":
                        debug_print(f"Creating negation for {operand_var}")
                        op = QuantumNegOp(
                                result_types=[i32],
                                operands=[ssa_map[operand_var]]
                                )
                        blk.add_op(op)
                        ssa_map[var] = op.results[0]
                        debug_print(f"Negation created: {var} = -{operand_var}")

                    # Handle increment (++) operators
                    if opcode == "++" or opcode == "inc":
                        if is_postfix:
                            # Post-increment (x++)
                            debug_print(f"Creating post-increment for {operand_var}")
                            op = QuantumPostIncrementOp(
                                result_types=[i32, i32],
                                operands=[ssa_map[operand_var]]
                            )
                            blk.add_op(op)
                            ssa_map[var] = op.results[0]  # Original value
                            # Update the original variable with the incremented value
                            ssa_map[operand_var] = op.results[1]
                        else:
                            # Pre-increment (++x)
                            debug_print(f"Creating pre-increment for {operand_var}")
                            op = QuantumPreIncrementOp(
                                result_types=[i32],
                                operands=[ssa_map[operand_var]]
                            )
                            blk.add_op(op)
                            ssa_map[var] = op.results[0]
                            # Update the original variable too
                            ssa_map[operand_var] = op.results[0]
                    
                    # Handle decrement (--) operators
                    elif opcode == "--" or opcode == "dec":
                        if is_postfix:
                            # Post-decrement (x--)
                            debug_print(f"Creating post-decrement for {operand_var}")
                            op = QuantumPostDecrementOp(
                                result_types=[i32, i32],
                                operands=[ssa_map[operand_var]]
                            )
                            blk.add_op(op)
                            ssa_map[var] = op.results[0]  # Original value
                            # Update the original variable with the decremented value
                            ssa_map[operand_var] = op.results[1]
                        else:
                            # Pre-decrement (--x)
                            debug_print(f"Creating pre-decrement for {operand_var}")
                            op = QuantumPreDecrementOp(
                                result_types=[i32],
                                operands=[ssa_map[operand_var]]
                            )
                            blk.add_op(op)
                            ssa_map[var] = op.results[0]
                            # Update the original variable too
                            ssa_map[operand_var] = op.results[0]

                    # Handle bitwise NOT (~) operator
                    elif opcode == "~" or opcode == "not":
                        debug_print(f"Creating bitwise NOT for {operand_var}")
                        op = QuantumNotOp(
                            result_types=[i32],
                            operands=[ssa_map[operand_var]]
                        )
                        blk.add_op(op)
                        ssa_map[var] = op.results[0]
                        debug_print(f"Bitwise NOT created: {var} = ~{operand_var}")
                else:
                    # fallback zero init if we can't process unary op
                    debug_print(f"Could not process unary op for {var}, using fallback")
                    op = QuantumInitOp(
                        result_types=[i32],
                        attributes={"type": i32, "value": IntegerAttr(0, i32)}
                    )
                    blk.add_op(op)
                    ssa_map[var] = op.results[0]
            
            # fallback zero init for any other case
            else:
                debug_print(f"No recognizable initializer for {var}, using fallback initialization with value 0")
                op = QuantumInitOp(
                    result_types=[i32],
                    attributes={"type": i32, "value": IntegerAttr(0, i32)}
                )
                blk.add_op(op)
                ssa_map[var] = op.results[0]
            
    elif kind == "BinaryOperator":
        # Handle standalone assignment
        if stmt.get("opcode") == "=":
            debug_print(f"Processing assignment operation")
            lhs_var = extract_var_name(stmt.get("inner", [])[0]) if len(stmt.get("inner", [])) > 0 else None
            rhs_node = stmt.get("inner", [])[1] if len(stmt.get("inner", [])) > 1 else None
            
            debug_print(f"Assignment: {lhs_var} = {rhs_node.get('kind') if rhs_node else 'None'}")
            
            if lhs_var and rhs_node:
                debug_print(f"Processing assignment to {lhs_var}")
                
                # Handle different right-hand side node types
                if rhs_node.get("kind") == "BinaryOperator":
                    opcode = rhs_node.get("opcode")
                    debug_print(f"Processing binary operation: {opcode}")
                    
                    # Extract operands from the binary operation
                    rhs_inner = rhs_node.get("inner", [])
                    if len(rhs_inner) >= 2:
                        left_operand = rhs_inner[0]
                        right_operand = rhs_inner[1]
                        
                        # Get the left operand (could be variable or literal)
                        left_var = extract_var_name(left_operand)
                        left_val = None
                        if left_var and left_var in ssa_map:
                            left_val = ssa_map[left_var]
                            debug_print(f"Left operand: variable {left_var}")
                        
                        # Get the right operand (could be variable or literal) 
                        right_var = extract_var_name(right_operand)
                        right_val = None
                        if right_var and right_var in ssa_map:
                            right_val = ssa_map[right_var]
                            debug_print(f"Right operand: variable {right_var}")
                        elif right_operand.get("kind") == "IntegerLiteral":
                            # Handle integer literal
                            lit_val = int(right_operand.get("value", "0"))
                            debug_print(f"Right operand: literal {lit_val}")
                            # Create a constant for the literal
                            const_op = QuantumInitOp(
                                result_types=[i32],
                                attributes={"type": i32, "value": IntegerAttr(lit_val, i32)}
                            )
                            blk.add_op(const_op)
                            right_val = const_op.results[0]
                        
                        # If we have both operands, create the operation
                        if left_val is not None and right_val is not None:
                            arith_map = {"+": QuantumAddOp, "-": QuantumSubOp, 
                                        "*": QuantumMulOp, "/": QuantumDivOp,
                                        "%": QuantumModOp}
                            logic_map = {"^": QuantumXorOp, "&&": QuantumAndOp, 
                                        "||": QuantumOrOp}
                            shift_map = {"<<": QuantumLeftShiftOp, ">>": QuantumRightShiftOp}
                            cmp_map = {"<": QuantumLessThanOp, ">": QuantumGreaterThanOp,
                                      "==": QuantumEqualOp, "!=": QuantumNotEqualOp,
                                      "<=": QuantumLessThanEqualOp, ">=": QuantumGreaterThanEqualOp}
                            
                            if opcode in arith_map:
                                debug_print(f"Creating arithmetic operation: {opcode}")
                                o = arith_map[opcode](
                                    result_types=[i32], 
                                    operands=[left_val, right_val]
                                )
                                blk.add_op(o)
                                debug_print(f"Added {opcode} operation to block")
                                ssa_map[lhs_var] = o.results[0]
                                debug_print(f"Updated SSA mapping: {lhs_var} -> {o.results[0]}")
                                
                                # If we're in a loop body, track this variable as modified
                                if in_loop_body:
                                    loop_modified_vars.add(lhs_var)
                                    debug_print(f"Tracked {lhs_var} as modified in loop body")
                                
                            elif opcode in logic_map:
                                o = logic_map[opcode](
                                    result_types=[i1], 
                                    operands=[left_val, right_val]
                                )
                                blk.add_op(o)
                                ssa_map[lhs_var] = o.results[0]
                                
                            elif opcode in shift_map:
                                o = shift_map[opcode](
                                    result_types=[i32], 
                                    operands=[left_val, right_val]
                                )
                                blk.add_op(o)
                                ssa_map[lhs_var] = o.results[0]
                                
                            elif opcode in cmp_map:
                                o = cmp_map[opcode](
                                    result_types=[i1], 
                                    operands=[left_val, right_val]
                                )
                                blk.add_op(o)
                                ssa_map[lhs_var] = o.results[0]
                        else:
                            debug_print(f"Could not resolve operands for {opcode} operation")
                
                # Handle UnaryOperator on right side
                elif rhs_node.get("kind") == "UnaryOperator":
                    opcode = rhs_node.get("opcode")
                    is_postfix = is_postfix_operator(rhs_node)
                    
                    operand_node = rhs_node.get("inner", [])[0] if len(rhs_node.get("inner", [])) > 0 else None
                    operand_var = extract_var_name(operand_node)
                    
                    if operand_var and operand_var in ssa_map:
                        # Handle unary minus (negation)
                        if opcode == "-" or opcode == "minus":
                            debug_print(f"Creating negation assignment: {lhs_var} = -{operand_var}")
                            op = QuantumNegOp(
                                    result_types=[i32],
                                    operands=[ssa_map[operand_var]]
                                    )
                            blk.add_op(op)
                            ssa_map[lhs_var] = op.results[0]

                            # Track variable modification in loop body 
                            if in_loop_body:
                                loop_modified_vars.add(lhs_var)
                                debug_print(f"Tracked {lhs_var} as modified in loop body (negation assignment)")
                        # Handle increment (++) operators
                        if opcode == "++" or opcode == "inc":
                            if is_postfix:
                                # Post-increment (x++)
                                op = QuantumPostIncrementOp(
                                    result_types=[i32, i32],
                                    operands=[ssa_map[operand_var]]
                                )
                                blk.add_op(op)
                                ssa_map[lhs_var] = op.results[0]
                                ssa_map[operand_var] = op.results[1]
                                
                                # Track variable modifications in loop body
                                if in_loop_body:
                                    loop_modified_vars.add(lhs_var)
                                    loop_modified_vars.add(operand_var)
                                    debug_print(f"Tracked {lhs_var} and {operand_var} as modified in loop body")
                            else:
                                # Pre-increment (++x)
                                op = QuantumPreIncrementOp(
                                    result_types=[i32],
                                    operands=[ssa_map[operand_var]]
                                )
                                blk.add_op(op)
                                ssa_map[lhs_var] = op.results[0]
                                ssa_map[operand_var] = op.results[0]
                        
                        # Handle decrement (--) operators
                        elif opcode == "--" or opcode == "dec":
                            if is_postfix:
                                # Post-decrement (x--)
                                op = QuantumPostDecrementOp(
                                    result_types=[i32, i32],
                                    operands=[ssa_map[operand_var]]
                                )
                                blk.add_op(op)
                                ssa_map[lhs_var] = op.results[0]  # Original value
                                ssa_map[operand_var] = op.results[1]  # Decremented value
                            else:
                                # Pre-decrement (--x)
                                op = QuantumPreDecrementOp(
                                    result_types=[i32],
                                    operands=[ssa_map[operand_var]]
                                )
                                blk.add_op(op)
                                ssa_map[lhs_var] = op.results[0]
                                ssa_map[operand_var] = op.results[0]
                
                # Handle simple variable reference on the right side
                else:
                    rhs_var = extract_var_name(rhs_node)
                    if rhs_var and rhs_var in ssa_map:
                        debug_print(f"Assignment from variable: {rhs_var}")
                        ssa_map[lhs_var] = ssa_map[rhs_var]
                        
                        # Track variable modification in loop body
                        if in_loop_body:
                            loop_modified_vars.add(lhs_var)
                            debug_print(f"Tracked {lhs_var} as modified in loop body (simple assignment)")
    
    # Handle standalone unary operations
    elif kind == "UnaryOperator":
        opcode = stmt.get("opcode")
        debug_print(f"Processing standalone unary operator: {opcode}")
        is_postfix = is_postfix_operator(stmt)
        debug_print(f"Is postfix: {is_postfix}")
        
        # Get the operand
        operand_node = stmt.get("inner", [])[0] if len(stmt.get("inner", [])) > 0 else None
        operand_var = extract_var_name(operand_node)
        
        if operand_var and operand_var in ssa_map:
            # Handle unary minus (negation) - standalone negation doesn't modify the original variable 
            if opcode == "-" or opcode == "minus":
                debug_print(f"Standalone negation of {operand_var} (no assignment)")
                # For standalone negation, we might just want to create the operation but not store it 
                # This is unusal in C, but we can handle it 
                op = QuantumNegOp(
                        result_types=[i32],
                        operands=[ssa_map[operand_var]]
                        )
                blk.add_op(op)

            # Handle increment (++) operators
            if opcode == "++" or opcode == "inc":
                if is_postfix:
                    # Post-increment (x++)
                    op = QuantumPostIncrementOp(
                        result_types=[i32, i32],
                        operands=[ssa_map[operand_var]]
                    )
                    blk.add_op(op)
                    # Update the variable with the incremented value
                    ssa_map[operand_var] = op.results[1]
                    
                    # Track variable modification in loop body
                    if in_loop_body:
                        loop_modified_vars.add(operand_var)
                        debug_print(f"Tracked {operand_var} as modified in loop body (standalone post-increment)")
                else:
                    # Pre-increment (++x)
                    op = QuantumPreIncrementOp(
                        result_types=[i32],
                        operands=[ssa_map[operand_var]]
                    )
                    blk.add_op(op)
                    ssa_map[operand_var] = op.results[0]
                    
                    # Track variable modification in loop body
                    if in_loop_body:
                        loop_modified_vars.add(operand_var)
                        debug_print(f"Tracked {operand_var} as modified in loop body (standalone pre-increment)")
            
            # Handle decrement (--) operators
            elif opcode == "--" or opcode == "dec":
                if is_postfix:
                    # Post-decrement (x--)
                    op = QuantumPostDecrementOp(
                        result_types=[i32, i32],
                        operands=[ssa_map[operand_var]]
                    )
                    blk.add_op(op)
                    # Update the variable with the decremented value
                    ssa_map[operand_var] = op.results[1]
                    
                    # Track variable modification in loop body
                    if in_loop_body:
                        loop_modified_vars.add(operand_var)
                        debug_print(f"Tracked {operand_var} as modified in loop body (standalone post-decrement)")
                else:
                    # Pre-decrement (--x)
                    op = QuantumPreDecrementOp(
                        result_types=[i32],
                        operands=[ssa_map[operand_var]]
                    )
                    blk.add_op(op)
                    ssa_map[operand_var] = op.results[0]
                    
                    # Track variable modification in loop body
                    if in_loop_body:
                        loop_modified_vars.add(operand_var)
                        debug_print(f"Tracked {operand_var} as modified in loop body (standalone pre-decrement)")
                
    # Handle compound assignments (+=, -=, *=, /=)
    elif kind == "CompoundAssignOperator":
        opcode = stmt.get("opcode")
        debug_print(f"Processing compound assignment: {opcode}")
        
        # Extract left and right hand sides
        lhs_node = stmt.get("inner", [])[0] if len(stmt.get("inner", [])) > 0 else None
        rhs_node = stmt.get("inner", [])[1] if len(stmt.get("inner", [])) > 1 else None
        
        lhs_var = extract_var_name(lhs_node)
        rhs_var = extract_var_name(rhs_node)
        
        if lhs_var and lhs_var in ssa_map and rhs_var and rhs_var in ssa_map:
            # Map compound operators to binary operators
            op_map = {
                "+=": QuantumAddOp,
                "-=": QuantumSubOp,
                "*=": QuantumMulOp,
                "/=": QuantumDivOp,
                "%=": QuantumModOp,
                "<<=": QuantumLeftShiftOp,
                ">>=": QuantumRightShiftOp,
                "^=": QuantumXorOp,
                "&=": QuantumAndOp,
                "|=": QuantumOrOp
            }
            
            if opcode in op_map:
                # Create the corresponding binary operation
                result_type = i32 if opcode not in ["^=", "&=", "|="] else i1
                
                o = op_map[opcode](
                    result_types=[result_type],
                    operands=[ssa_map[lhs_var], ssa_map[rhs_var]]
                )
                blk.add_op(o)
                ssa_map[lhs_var] = o.results[0]
                
                # Track variable modification in loop body
                if in_loop_body:
                    loop_modified_vars.add(lhs_var)
                    debug_print(f"Tracked {lhs_var} as modified in loop body (compound assignment)")
            
    elif kind == "CallExpr" and (stmt.get("callee",{}).get("name") == "printf" or 
                              "printf" in str(stmt.get("inner", [0]))):
        var_names = process_printf_args(stmt)
        for var_name in var_names:
            if var_name in ssa_map:
                debug_print(f"Adding measurement for {var_name}")
                m = QuantumMeasureOp(result_types=[i1], operands=[ssa_map[var_name]])
                blk.add_op(m)
                break

    # Handle compound statements (eg: block of code)
    elif kind == "CompoundStmt":
        # Create a new scope for local variables
        push_scope()
        
        # Process all statements in the compound statement
        for inner_stmt in stmt.get("inner", []):
            translate_stmt(inner_stmt, blk)
            
        # Exit the scope
        pop_scope()

# Main
def main():
    global current_block, in_loop_body
    
    # Initialize global flags
    in_loop_body = False
    
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
    current_block = blk
    
    # Find and process the body of the quantum_circuit function
    body = find_body(ast, "quantum_circuit")
    debug_print(f"Found {len(body)} statements in quantum_circuit")
    
    # Process each statement
    for idx, stmt in enumerate(body):
        debug_print(f"\nProcessing statement {idx+1}/{len(body)}")
        translate_stmt(stmt, current_block)
    
    # Add return and finalize module
    current_block.add_op(ReturnOp())
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
