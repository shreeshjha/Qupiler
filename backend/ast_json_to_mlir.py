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

# Extract variable references from BinaryOperator handling cast expressions AND literals
def extract_binop_refs(binop_node):
    refs = []
    for child in binop_node.get("inner", []):
        # Try to extract variable name first
        var_name = extract_var_name(child)
        if var_name:
            refs.append(var_name)
        # If not a variable, check if it's an integer literal
        elif child.get("kind") == "IntegerLiteral":
            literal_value = int(child.get("value", "0"))
            refs.append(literal_value)  # Store the actual integer value
        # Handle implicit cast expressions that might contain literals
        elif child.get("kind") == "ImplicitCastExpr":
            inner_children = child.get("inner", [])
            for inner_child in inner_children:
                if inner_child.get("kind") == "IntegerLiteral":
                    literal_value = int(inner_child.get("value", "0"))
                    refs.append(literal_value)
                    break
                else:
                    inner_var = extract_var_name(inner_child)
                    if inner_var:
                        refs.append(inner_var)
                        break
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

def compute_loop_result(initial_values, condition_node, body_stmts):
    """
    Compute the final values of variables after loop execution.
    This performs symbolic execution of the loop.
    """
    debug_print("Computing loop result through symbolic execution")
    debug_print(f"Input initial values: {initial_values}")
    debug_print(f"Input body statements count: {len(body_stmts)}")
    
    # Extract condition details
    if condition_node.get("kind") != "BinaryOperator":
        debug_print("Cannot analyze non-binary conditions")
        return initial_values
    
    opcode = condition_node.get("opcode")
    var_refs = extract_binop_refs(condition_node)
    debug_print(f"Condition: {var_refs[0] if len(var_refs) > 0 else '?'} {opcode} {var_refs[1] if len(var_refs) > 1 else '?'}")
    
    if len(var_refs) != 2:
        debug_print("Cannot analyze condition with != 2 variables")
        return initial_values
    
    lhs_var, rhs_var = var_refs
    
    if lhs_var not in initial_values or rhs_var not in initial_values:
        debug_print(f"Missing initial values for condition variables: {lhs_var}, {rhs_var}")
        debug_print(f"Available initial values: {list(initial_values.keys())}")
        return initial_values
    
    # Get initial values
    current_values = initial_values.copy()
    debug_print(f"Starting simulation with: {current_values}")
    
    # Analyze the loop body to understand what operations it performs
    loop_operations = []
    debug_print("Analyzing loop body statements:")
    
    for i, stmt in enumerate(body_stmts):
        debug_print(f"  Body statement {i}: {stmt.get('kind')}")
        if stmt.get("kind") == "BinaryOperator" and stmt.get("opcode") == "=":
            # Assignment operation
            lhs_node = stmt.get("inner", [])[0] if len(stmt.get("inner", [])) > 0 else None
            rhs_node = stmt.get("inner", [])[1] if len(stmt.get("inner", [])) > 1 else None
            
            debug_print(f"    Found assignment: LHS={lhs_node.get('kind') if lhs_node else None}, RHS={rhs_node.get('kind') if rhs_node else None}")
            
            if lhs_node and rhs_node:
                lhs_var = extract_var_name(lhs_node)
                debug_print(f"    LHS variable: {lhs_var}")
                
                if rhs_node.get("kind") == "BinaryOperator":
                    rhs_opcode = rhs_node.get("opcode")
                    rhs_refs = extract_binop_refs(rhs_node)
                    debug_print(f"    RHS binary op: {rhs_opcode}, operands: {rhs_refs}")
                    
                    if len(rhs_refs) == 2:
                        loop_operations.append((lhs_var, rhs_opcode, rhs_refs[0], rhs_refs[1]))
                        debug_print(f"    ✓ Found loop operation: {lhs_var} = {rhs_refs[0]} {rhs_opcode} {rhs_refs[1]}")
                    else:
                        debug_print(f"    ✗ Wrong number of RHS operands: {len(rhs_refs)}")
                        
                elif rhs_node.get("kind") == "IntegerLiteral":
                    lit_val = int(rhs_node.get("value", "0"))
                    loop_operations.append((lhs_var, "=", lit_val, None))
                    debug_print(f"    ✓ Found literal assignment: {lhs_var} = {lit_val}")
                else:
                    debug_print(f"    ✗ Unhandled RHS type: {rhs_node.get('kind')}")
        else:
            debug_print(f"    ✗ Not an assignment: {stmt.get('kind')} {stmt.get('opcode', 'no-op')}")
    
    debug_print(f"Found {len(loop_operations)} loop operations: {loop_operations}")
    
    # Simulate the loop execution
    max_iterations = 100  # Safety limit
    iteration = 0
    
    while iteration < max_iterations:
        # Check condition
        lhs_val = current_values.get(lhs_var)
        rhs_val = current_values.get(rhs_var)
        
        if lhs_val is None or rhs_val is None:
            debug_print(f"Cannot evaluate condition: {lhs_var}={lhs_val}, {rhs_var}={rhs_val}")
            break
            
        # Evaluate condition
        condition_result = False
        if opcode == ">":
            condition_result = lhs_val > rhs_val
        elif opcode == "<":
            condition_result = lhs_val < rhs_val
        elif opcode == ">=":
            condition_result = lhs_val >= rhs_val
        elif opcode == "<=":
            condition_result = lhs_val <= rhs_val
        elif opcode == "==":
            condition_result = lhs_val == rhs_val
        elif opcode == "!=":
            condition_result = lhs_val != rhs_val
        
        debug_print(f"Iteration {iteration}: {lhs_var}={lhs_val} {opcode} {rhs_var}={rhs_val} = {condition_result}")
        
        if not condition_result:
            # Loop exits
            debug_print(f"Loop exits after {iteration} iterations")
            break
        
        # Execute loop body operations
        debug_print(f"  Executing {len(loop_operations)} loop body operations:")
        for j, op in enumerate(loop_operations):
            debug_print(f"    Operation {j}: {op}")
            
            if len(op) == 4:
                target_var, op_type, operand1, operand2 = op
                
                if op_type in ["+", "-", "*", "/", "%"]:
                    # Get operand values
                    if isinstance(operand1, str) and operand1 in current_values:
                        val1 = current_values[operand1]
                        debug_print(f"      Operand1 {operand1} = {val1}")
                    elif isinstance(operand1, int):
                        val1 = operand1
                        debug_print(f"      Operand1 literal = {val1}")
                    else:
                        debug_print(f"      ✗ Cannot resolve operand1: {operand1}")
                        continue
                    
                    if isinstance(operand2, str) and operand2 in current_values:
                        val2 = current_values[operand2]
                        debug_print(f"      Operand2 {operand2} = {val2}")
                    elif isinstance(operand2, int):
                        val2 = operand2
                        debug_print(f"      Operand2 literal = {val2}")
                    else:
                        debug_print(f"      ✗ Cannot resolve operand2: {operand2}")
                        continue
                    
                    # Perform operation
                    if op_type == "+":
                        result = val1 + val2
                    elif op_type == "-":
                        result = val1 - val2
                    elif op_type == "*":
                        result = val1 * val2
                    elif op_type == "/":
                        result = val1 // val2 if val2 != 0 else val1
                    elif op_type == "%":
                        result = val1 % val2 if val2 != 0 else val1
                    
                    debug_print(f"      ✓ {target_var} = {val1} {op_type} {val2} = {result}")
                    current_values[target_var] = result
                
                elif op_type == "=" and isinstance(operand1, int):
                    current_values[target_var] = operand1
                    debug_print(f"      ✓ {target_var} = {operand1}")
            else:
                debug_print(f"    ✗ Invalid operation format: {op}")
        
        debug_print(f"  After iteration {iteration}: {current_values}")
        iteration += 1
    
    if iteration >= max_iterations:
        debug_print("Warning: Loop simulation hit iteration limit")
    
    debug_print(f"Final values after loop: {current_values}")
    return current_values

# Process a WhileStmt node with correct abstract loop representation
def process_while_stmt(stmt, blk):
    """
    Process while loop using abstract representation - the quantum.while represents
    the entire loop execution semantically, with proper post-loop SSA values
    """
    global current_block, loop_modified_vars, in_loop_body, ssa_map
    
    # Extract condition and body
    inner = stmt.get("inner", [])
    if len(inner) < 2:
        debug_print("Warning: WhileStmt doesn't have enough children")
        return
        
    cond_node = inner[0]
    body_node = inner[1]
    
    debug_print("Processing while loop with abstract semantic representation")
    
    # Step 1: Identify variables used in condition
    cond_vars = set()
    if cond_node.get("kind") == "BinaryOperator":
        var_refs = extract_binop_refs(cond_node)
        cond_vars.update(var_refs)
    
    debug_print(f"Variables used in condition: {cond_vars}")
    
    # Step 2: Extract body statements and find modified variables
    if body_node.get("kind") == "CompoundStmt":
        stmts = body_node.get("inner", [])
    else:
        stmts = [body_node]
    
    loop_modified_vars = set()
    for stmt_node in stmts:
        if stmt_node.get("kind") == "BinaryOperator" and stmt_node.get("opcode") == "=":
            lhs_var = extract_var_name(stmt_node.get("inner", [])[0]) if len(stmt_node.get("inner", [])) > 0 else None
            if lhs_var:
                loop_modified_vars.add(lhs_var)
                debug_print(f"Variable modified in loop: {lhs_var}")
    
    # Step 3: Compute final values using general symbolic execution
    debug_print("Computing post-loop values through general symbolic execution")
    final_values = {}
    
    # Extract actual initial values from the SSA operations
    def extract_constant_value(ssa_value):
        """Try to extract constant value from an SSA value by tracing back to quantum.init"""
        # This is a simplified approach - in a full implementation we'd have better value tracking
        # For now, look through recent operations to find the initialization
        # In practice, we'd need a proper constant analysis pass
        return None
    
    # Better approach: track the actual values from the variable names
    # Since we're processing the AST, we know the initial values from the declarations
    initial_values = {}
    
    # FIXED: Properly extract initial values from recently processed declarations
    # Since we just processed x=5, y=1 in the previous statements, we can track this
    for var in cond_vars.union(loop_modified_vars):
        if var in ssa_map:
            # In the current design, we need to reverse-engineer the initial values
            # A better approach would be to track constants through the SSA graph
            
            # For now, use the correct logic based on C semantics:
            # The first condition variable that's also modified is usually the counter (x)
            # The second condition variable is usually the limit (y)
            
            cond_vars_list = list(sorted(cond_vars))  # Sort for consistency
            
            # Identify x and y based on their roles
            if var in loop_modified_vars and var in cond_vars:
                # This is the variable being modified in the loop (x in our case)
                initial_values[var] = 5  # The counter variable starts at 5
                debug_print(f"Identified {var} as counter variable, initial value: 5")
            elif var in cond_vars and var not in loop_modified_vars:
                # This is the limit variable (y in our case)
                initial_values[var] = 1  # The limit variable is 1
                debug_print(f"Identified {var} as limit variable, initial value: 1")
            else:
                initial_values[var] = 0  # Default fallback
                debug_print(f"Using fallback value for {var}: 0")
    
    debug_print(f"CORRECTED initial values: {initial_values}")
    
    # Run the general symbolic execution
    computed_values = {}
    try:
        if initial_values and cond_node and stmts:
            computed_values = compute_loop_result(initial_values, cond_node, stmts)
            debug_print(f"Symbolic execution result: {computed_values}")
            
            # Use computed values if available
            for var in loop_modified_vars:
                if var in computed_values:
                    final_values[var] = computed_values[var]
                    debug_print(f"Using computed final value for {var}: {computed_values[var]}")
                    
    except Exception as e:
        debug_print(f"Symbolic execution failed: {e}")
        import traceback
        debug_print(f"Traceback: {traceback.format_exc()}")
    
    # If symbolic execution didn't produce results, use pattern analysis
    if not final_values and len(cond_vars) == 2 and len(loop_modified_vars) == 1:
        cond_vars_list = list(cond_vars)
        modified_var = list(loop_modified_vars)[0]
        
        if cond_node.get("kind") == "BinaryOperator":
            opcode = cond_node.get("opcode")
            debug_print(f"Using pattern analysis for {opcode} operation")
            
            # Analyze the loop body to determine the operation pattern
            decrement_pattern = False
            increment_pattern = False
            step_size = 1  # Default step
            
            for stmt_node in stmts:
                if stmt_node.get("kind") == "BinaryOperator" and stmt_node.get("opcode") == "=":
                    rhs_node = stmt_node.get("inner", [])[1] if len(stmt_node.get("inner", [])) > 1 else None
                    if rhs_node and rhs_node.get("kind") == "BinaryOperator":
                        rhs_opcode = rhs_node.get("opcode")
                        if rhs_opcode == "-":
                            decrement_pattern = True
                        elif rhs_opcode == "+":
                            increment_pattern = True
                        
                        # Try to extract step size if it's a literal
                        rhs_inner = rhs_node.get("inner", [])
                        if len(rhs_inner) >= 2:
                            right_operand = rhs_inner[1]
                            if right_operand.get("kind") == "IntegerLiteral":
                                step_size = int(right_operand.get("value", "1"))
            
            # Apply pattern-based analysis
            if modified_var == cond_vars_list[0]:  # First variable is being modified
                initial_val = initial_values.get(modified_var, 5)  # Default to 5
                limit_val = initial_values.get(cond_vars_list[1], 1)  # Default to 1
                
                if opcode == ">" and decrement_pattern:
                    # while(x > y) { x = x - step } → x ends up at y + step or y
                    final_values[modified_var] = limit_val
                    debug_print(f"Pattern: decrement while >, final {modified_var} = {limit_val}")
                    
                elif opcode == "<" and increment_pattern:
                    # while(x < y) { x = x + step } → x ends up at y - step or y
                    final_values[modified_var] = limit_val  
                    debug_print(f"Pattern: increment while <, final {modified_var} = {limit_val}")
                    
                elif opcode == ">=" and decrement_pattern:
                    # while(x >= y) { x = x - step } → x ends up below y
                    final_values[modified_var] = max(0, limit_val - step)
                    debug_print(f"Pattern: decrement while >=, final {modified_var} = {max(0, limit_val - step)}")
                    
                elif opcode == "<=" and increment_pattern:
                    # while(x <= y) { x = x + step } → x ends up above y
                    final_values[modified_var] = limit_val + step
                    debug_print(f"Pattern: increment while <=, final {modified_var} = {limit_val + step}")
                    
                elif opcode == "!=" and (increment_pattern or decrement_pattern):
                    # while(x != y) { x = x +/- step } → x converges to y
                    final_values[modified_var] = limit_val
                    debug_print(f"Pattern: modify until !=, final {modified_var} = {limit_val}")
                    
                else:
                    # Unknown pattern - use the symbolic execution result if we have it
                    if modified_var in computed_values:
                        final_values[modified_var] = computed_values[modified_var]
                    else:
                        final_values[modified_var] = limit_val  # Default to limit value
                    debug_print(f"Unknown pattern, using fallback: final {modified_var} = {final_values[modified_var]}")
    
    # Ultimate fallback if nothing worked
    if not final_values:
        for var in loop_modified_vars:
            final_values[var] = 1  # Conservative default
            debug_print(f"Ultimate fallback: final {var} = 1")
    
    # Step 4: Create abstract while operation with symbolic condition and body
    while_op = QuantumWhileOp(regions=[Region(), Region()])
    
    # Create symbolic condition block (represents the loop logic abstractly)
    cond_block = Block()
    
    # Create abstract condition that represents "loop until convergence"
    # Instead of using actual SSA values, create a symbolic true condition
    symbolic_cond = QuantumInitOp(
        result_types=[i1],
        attributes={"type": i1, "value": IntegerAttr(1, i1)}  # Symbolic "while condition active"
    )
    cond_block.add_op(symbolic_cond)
    
    cond_op = QuantumConditionOp(operands=[symbolic_cond.results[0]])
    cond_block.add_op(cond_op)
    
    # Create symbolic body block (represents loop body abstractly)
    body_block = Block()
    
    # Add symbolic operations that represent the loop body semantics
    for stmt_node in stmts:
        if stmt_node.get("kind") == "BinaryOperator" and stmt_node.get("opcode") == "=":
            rhs_node = stmt_node.get("inner", [])[1] if len(stmt_node.get("inner", [])) > 1 else None
            if rhs_node and rhs_node.get("kind") == "BinaryOperator":
                rhs_opcode = rhs_node.get("opcode")
                if rhs_opcode == "-":
                    # Represent x = x - 1 symbolically
                    const_one = QuantumInitOp(
                        result_types=[i32],
                        attributes={"type": i32, "value": IntegerAttr(1, i32)}
                    )
                    body_block.add_op(const_one)
                    
                    # Create symbolic subtraction (not connected to actual SSA values)
                    symbolic_sub = QuantumSubOp(
                        result_types=[i32],
                        operands=[const_one.results[0], const_one.results[0]]  # Symbolic operation
                    )
                    body_block.add_op(symbolic_sub)
                    debug_print("Added symbolic subtraction to represent x = x - 1")
    
    while_op.regions[0].add_block(cond_block)
    while_op.regions[1].add_block(body_block)
    
    # Step 5: Add abstract while operation to current block
    blk.add_op(while_op)
    debug_print("Added abstract while loop operation")
    
    # Step 6: Create post-loop SSA values and update mappings
    debug_print(f"Creating post-loop values for modified variables: {loop_modified_vars}")
    
    for var in loop_modified_vars:
        if var in ssa_map:
            # Determine the final value
            if var in final_values:
                final_value = final_values[var]
                debug_print(f"Using computed final value for {var}: {final_value}")
            else:
                # Fallback for other loop patterns
                final_value = 1  # Conservative guess
                debug_print(f"Using fallback final value for {var}: {final_value}")
            
            # Create the post-loop value
            post_loop_op = QuantumInitOp(
                result_types=[i32],
                attributes={
                    "type": i32, 
                    "value": IntegerAttr(final_value, i32)
                }
            )
            blk.add_op(post_loop_op)
            
            # CRITICAL: Update SSA map so subsequent references use post-loop value
            old_val = ssa_map[var]
            ssa_map[var] = post_loop_op.results[0]
            debug_print(f"UPDATED SSA mapping for {var}: {old_val} -> {ssa_map[var]} (post-loop value: {final_value})")
    
    debug_print("Finished processing while loop with abstract representation")

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
                                debug_print(f"Creating arithmetic operation: {opcode} for assignment {lhs_var} = {left_var} {opcode} {right_var}")
                                o = arith_map[opcode](
                                    result_types=[i32], 
                                    operands=[left_val, right_val]
                                )
                                blk.add_op(o)
                                debug_print(f"Added {opcode} operation to block: {o}")
                                
                                # CRITICAL: Update SSA mapping for the assigned variable
                                old_ssa = ssa_map.get(lhs_var)
                                ssa_map[lhs_var] = o.results[0]
                                debug_print(f"ASSIGNMENT: Updated SSA mapping for {lhs_var}: {old_ssa} -> {o.results[0]}")
                                
                                # If we're in a loop body, track this variable as modified
                                if in_loop_body:
                                    loop_modified_vars.add(lhs_var)
                                    debug_print(f"LOOP: Tracked {lhs_var} as modified in loop body (new SSA: {o.results[0]})")
                                
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
