from ..op import Operation, OpConstant


def sp_const_sub_tree(op: Operation):
    op.inputs = [sp_const_sub_tree(x) for x in op.inputs]
    if len(op.inputs) > 0 and all(map(lambda x: isinstance(x, OpConstant), op.inputs)):
        return OpConstant(op.forward())
    return op
