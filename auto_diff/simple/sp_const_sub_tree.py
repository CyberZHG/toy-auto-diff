import auto_diff as ad


def sp_const_sub_tree(op: ad.Operation):
    op.inputs = [sp_const_sub_tree(x) for x in op.inputs]
    if len(op.inputs) > 0 and all(map(lambda x: isinstance(x, ad.OpConstant), op.inputs)):
        return ad.OpConstant(op.forward())
    return op
