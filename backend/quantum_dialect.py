from xdsl.ir import Dialect, Region, Block
from xdsl.irdl import IRDLOperation, irdl_op_definition, Operand, result_def, operand_def, attr_def, region_def
from xdsl.dialects.builtin import StringAttr, IntegerAttr, i1, i32, Builtin
from xdsl.context import Context

@irdl_op_definition
class QuantumInitOp(IRDLOperation):
    name = "quantum.init"
    type_attr  = attr_def(StringAttr)
    value_attr = attr_def(IntegerAttr)
    result     = result_def()

@irdl_op_definition
class QuantumNotOp(IRDLOperation):
    name    = "quantum.not"
    operand = operand_def()
    result  = result_def()

@irdl_op_definition
class QuantumCNOTOp(IRDLOperation):
    name    = "quantum.cnot"
    control = operand_def()
    target  = operand_def()
    result  = result_def()

@irdl_op_definition
class QuantumMeasureOp(IRDLOperation):
    name  = "quantum.measure"
    qubit = operand_def()
    result= result_def()

@irdl_op_definition
class QuantumFuncOp(IRDLOperation):
    name      = "quantum.func"
    body      = region_def()
    func_name = attr_def(StringAttr)

# Arithmetic ops
@irdl_op_definition
class QuantumAddOp(IRDLOperation):
    name = "quantum.add"
    lhs  = operand_def()
    rhs  = operand_def()
    result = result_def()

@irdl_op_definition
class QuantumSubOp(IRDLOperation):
    name = "quantum.sub"
    lhs  = operand_def()
    rhs  = operand_def()
    result = result_def()

@irdl_op_definition
class QuantumMulOp(IRDLOperation):
    name = "quantum.mul"
    lhs  = operand_def()
    rhs  = operand_def()
    result = result_def()

@irdl_op_definition
class QuantumDivOp(IRDLOperation):
    name = "quantum.div"
    lhs  = operand_def()
    rhs  = operand_def()
    result = result_def()

# Boolean ops
@irdl_op_definition
class QuantumXorOp(IRDLOperation):
    name = "quantum.xor"
    a    = operand_def()
    b    = operand_def()
    result = result_def()

@irdl_op_definition
class QuantumAndOp(IRDLOperation):
    name = "quantum.and"
    a    = operand_def()
    b    = operand_def()
    result = result_def()

@irdl_op_definition
class QuantumOrOp(IRDLOperation):
    name = "quantum.or"
    a    = operand_def()
    b    = operand_def()
    result = result_def()

def register_quantum_dialect(ctx: Context):
    ops = [
      QuantumInitOp, QuantumNotOp, QuantumCNOTOp, QuantumMeasureOp,
      QuantumFuncOp,
      QuantumAddOp, QuantumSubOp, QuantumMulOp, QuantumDivOp,
      QuantumXorOp, QuantumAndOp, QuantumOrOp
    ]
    dialect = Dialect("quantum", ops, [])
    ctx.register_dialect("quantum", lambda c: dialect)
    ctx.register_dialect("builtin", lambda c: Builtin())


