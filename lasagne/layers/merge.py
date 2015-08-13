import theano.tensor as T

from .base import MergeLayer


__all__ = [
    "ConcatLayer",
    "concat",
    "ElemwiseMergeLayer",
    "ElemwiseSumLayer",
]


class MergeCropLayer(MergeLayer):
    """
    This class adds cropping to MergeLayer.
    """
    CROP_NONE = None
    CROP_LOWER = 'lower'
    CROP_CENTER = 'center'
    CROP_UPPER = 'upper'

    def __init__(self, incomings, cropping=None, name=None):
        super(MergeCropLayer, self).__init__(incomings, name=name)

        self.cropping = cropping

    def _merge_input_shapes(self, input_shapes):
        if self.cropping is None:
            return input_shapes[0]
        else:
            result = []
            for sh, cr in zip(zip(*input_shapes), self.cropping):
                if cr is None:
                    result.append(sh[0])
                elif cr in {MergeCropLayer.CROP_LOWER,
                            MergeCropLayer.CROP_CENTER,
                            MergeCropLayer.CROP_UPPER}:
                    result.append(min(sh))
                else:
                    raise ValueError('Unknown crop mode \'{0}\''.format(cr))
            return tuple(result)

    def _crop_inputs(self, inputs):
        if self.cropping is None:
            # No cropping in any dimension
            return inputs
        else:
            # Get the number of dimensions
            ndim = inputs[0].ndim
            # Get the shape of each input, where each shape will be a Theano
            # expression
            shapes = [input.shape for input in inputs]
            # Convert the shapes to a matrix expression
            shapes_tensor = T.as_tensor_variable(shapes)
            # Min along axis 0 to get the minimum size in each dimension
            min_shape = T.min(shapes_tensor, axis=0)

            # Nested list of slices; each list in `slices` corresponds to
            # an input and contains a slice for each dimension
            slices_by_input = [[] for i in range(ndim)]

            # If there are more dimensions than cropping entries, pad
            # the cropping
            cropping = list(self.cropping)
            if ndim > len(cropping):
                cropping = list(cropping) + \
                             [None] * (ndim - len(cropping))

            # For each dimension
            for dim, cr in enumerate(cropping):
                if cr == MergeCropLayer.CROP_NONE:
                    # Don't crop this dimension
                    slice_all = slice(None)
                    for slices in slices_by_input:
                        slices.append(slice_all)
                else:
                    # We crop all inputs in the dimension `dim` so that they
                    # are the minimum found in this dimension from all inputs
                    sz = min_shape[dim]
                    if cr == MergeCropLayer.CROP_LOWER:
                        # Choose the first `sz` elements
                        slc_lower = slice(None, sz)
                        for slices in slices_by_input:
                            slices.append(slc_lower)
                    elif cr == MergeCropLayer.CROP_UPPER:
                        # Choose the last `sz` elements
                        slc_upper = slice(-sz, None)
                        for slices in slices_by_input:
                            slices.append(slc_upper)
                    elif cr == MergeCropLayer.CROP_CENTER:
                        # Choose `sz` elements from the center
                        for sh, slices in zip(shapes, slices_by_input):
                            offset = (sh[dim] - sz) // 2
                            slices.append(slice(offset, offset+sz))
                    else:
                        raise ValueError(
                            'Unknown crop mode \'{0}\''.format(cr))

            return [input[slices] for input, slices in
                    zip(inputs, slices_by_input)]


class ConcatLayer(MergeCropLayer):
    """
    Concatenates multiple inputs along the specified axis. Inputs should have
    the same shape except for the dimension specified in axis, which can have
    different sizes.

    Parameters
    -----------
    incomings : a list of :class:`Layer` instances or tuples
        The layers feeding into this layer, or expected input shapes

    axis : int
        Axis which inputs are joined over
    """
    def __init__(self, incomings, axis=1, cropping=None, **kwargs):
        super(ConcatLayer, self).__init__(incomings, cropping=cropping,
                                          **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shapes):
        sizes = [input_shape[self.axis] for input_shape in input_shapes]
        output_shape = list(input_shapes[0])  # make a mutable copy
        output_shape[self.axis] = sum(sizes)
        return tuple(output_shape)

    def get_output_for(self, inputs, **kwargs):
        return T.concatenate(inputs, axis=self.axis)

concat = ConcatLayer  # shortcut


class ElemwiseMergeLayer(MergeCropLayer):
    """
    This layer performs an elementwise merge of its input layers.
    It requires all input layers to have the same output shape.

    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes,
        with all incoming shapes being equal

    merge_function : callable
        the merge function to use. Should take two arguments and return the
        updated value. Some possible merge functions are ``theano.tensor``:
        ``mul``, ``add``, ``maximum`` and ``minimum``.

    See Also
    --------
    ElemwiseSumLayer : Shortcut for sum layer.
    """

    def __init__(self, incomings, merge_function, cropping=None, **kwargs):
        super(ElemwiseMergeLayer, self).__init__(incomings, cropping=cropping,
                                                 **kwargs)
        self.merge_function = merge_function

    def get_output_shape_for(self, input_shapes):
        if any(shape != input_shapes[0] for shape in input_shapes):
            raise ValueError("Mismatch: not all input shapes are the same")
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        output = None
        for input in inputs:
            if output is not None:
                output = self.merge_function(output, input)
            else:
                output = input
        return output


class ElemwiseSumLayer(ElemwiseMergeLayer):
    """
    This layer performs an elementwise sum of its input layers.
    It requires all input layers to have the same output shape.

    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes,
        with all incoming shapes being equal

    coeffs: list or scalar
        A same-sized list of coefficients, or a single coefficient that
        is to be applied to all instances. By default, these will not
        be included in the learnable parameters of this layer.

    Notes
    -----
    Depending on your architecture, this can be used to avoid the more
    costly :class:`ConcatLayer`. For example, instead of concatenating layers
    before a :class:`DenseLayer`, insert separate :class:`DenseLayer` instances
    of the same number of output units and add them up afterwards. (This avoids
    the copy operations in concatenation, but splits up the dot product.)
    """
    def __init__(self, incomings, coeffs=1, cropping=None, **kwargs):
        super(ElemwiseSumLayer, self).__init__(incomings, T.add,
                                               cropping=cropping, **kwargs)
        if isinstance(coeffs, list):
            if len(coeffs) != len(incomings):
                raise ValueError("Mismatch: got %d coeffs for %d incomings" %
                                 (len(coeffs), len(incomings)))
        else:
            coeffs = [coeffs] * len(incomings)

        self.coeffs = coeffs

    def get_output_for(self, inputs, **kwargs):
        # if needed multiply each input by its coefficient
        inputs = [input * coeff if coeff != 1 else input
                  for coeff, input in zip(self.coeffs, inputs)]

        # pass scaled inputs to the super class for summing
        return super(ElemwiseSumLayer, self).get_output_for(inputs, **kwargs)
