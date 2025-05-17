from xdsl.ir import Dialect, OpResult, SSAValue, Region, Block
from xdsl.irdl import irdl_op_definition, Operand, AttrConstraint, result_def, operand_def
from xdsl.irdl import attr_def, region_def
from xdsl.irdl import IRDLOperation
from xdsl.dialects.builtin import FunctionType, StringAttr, IntegerAttr, ModuleOp, IntAttr
from xdsl.dialects import builtin
from xdsl.dialects.func import FuncOp 
from xdsl.context import Context
from xdsl.dialects.builtin import i1, i32
from xdsl.ir import Attribute

# === Dialect Definition === # 

@irdl_op_definition
class QuantumInitOp(IRDLOperation):
    name = "quantum.init"
    type_attr = attr_def(Attribute)
    value_attr = attr_def(Attribute)
    result = result_def()

@irdl_op_definition
class QuantumNotOp(IRDLOperation):
    name = "quantum.not"
    operand = operand_def()
    result = result_def()

@irdl_op_definition
class QuantumCNOTOp(IRDLOperation):
    name = "quantum.cnot"
    control = operand_def()
    target = operand_def()
    result = result_def()

@irdl_op_definition
class QuantumMeasureOp(IRDLOperation):
    name = "quantum.measure"
    qubit = operand_def()
    result = result_def()

@irdl_op_definition
class QuantumFuncOp(IRDLOperation):
    name = "quantum.func"
    body = region_def()
    func_name = attr_def(StringAttr)

# Create a dialect factory function
def get_quantum_dialect():
    return Dialect(
            "quantum",
            [QuantumInitOp, QuantumNotOp, QuantumCNOTOp, QuantumMeasureOp, QuantumFuncOp],
            []
    )

# Registering the dialect 
def register_quantum_dialect(ctx: Context):
    # Use the new dialect factory pattern
    ctx.register_dialect("quantum", get_quantum_dialect)
    
    # Try to load builtin dialect
    try:
        ctx.load_dialect("builtin")
    except Exception as e:
        print(f"Warning: Could not load builtin dialect: {e}. It may be automatically registered.")
