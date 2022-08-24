from torch.autograd import Function
import torch


class Proxy_table(Function):

    # def __init__(self, table, alpha=0.01):
    #     super(Proxy_table, self).__init__()
    #     self.table = table
    #     self.alpha = alpha
    @staticmethod
    def forward(ctx, inputs, targets,table,alpha):
        ctx.table = table
        ctx.alpha = alpha
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.table.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.table)
        for x, y in zip(inputs, targets):
            if y != -1.:
                ctx.table[y] = ctx.alpha * ctx.table[y] \
                                + (1. - ctx.alpha) * x
                ctx.table[y] /= ctx.table[y].norm()
        return grad_inputs, None, None, None
