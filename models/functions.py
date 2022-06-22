from torch.autograd import Function


class ReverseLayerF(Function):
    """ The gradient reverse layer class """

    @staticmethod
    def forward(ctx, x, alpha):
        """ The method for forward propagation """

        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """ The method for backpropagation """

        output = grad_output.neg() * ctx.alpha
        return output, None
