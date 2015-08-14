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
    This abstract base class class adds cropping to MergeLayer.

    Cropping takes a sequence of inputs and crops them per-axis in order to
    ensure that their sizes are consistent so that they can be combined
    in an element-wise fashion. When cropping is enabled - layers that
    support it take a cropping argument to the constructor - each
    axis has a cropping mode.  If cropping is enabled for a specific axis,
    the minimum size in that axis of all inputs is computed, and all
    inputs are cropped to that size.

    `CROP_NONE` (`None`): this axis is not cropped, inputs are unchanged in
    this axis

    `CROP_LOWER` (`'lower'`): inputs are cropped choosing the lower portion
    in this axis (`a[:crop_size, ...]`)

    `CROP_UPPER` (`'lower'`): inputs are cropped choosing the upper portion
    in this axis (`a[-crop_size:, ...]`)

    `CROP_CENTER` (`'center'`): inputs are cropped choosing the central
    portion in this axis (`a[offset:offset+crop_size, ...]` where
    `offset = (a.shape[0]-crop_size)//2)

    For example, given three inputs, whose shapes are:
    `a.shape == (1, 2, 3, 4)`
    `b.shape == (5, 4, 4, 2)`
    `c.shape == (7, 1, 8, 9)`

    with crop modes `[CROP_NONE, CROP_LOWER, CROP_CENTER, CROP_UPPER]`

    They will be left as is in axis 0 and cropped in all others,
    choosing the lower, center and upper portions:

    a[:, :1, :3, -2:]   # Choose all, lower 1 element, 3 central (all)
                        # and upper 2
    b[:, :1, :3, -2:]   # Choose all, lower 1 element, 3 central starting at 0
                        # and upper 2 (all)
    c[:, :1, 2:5:, -2:] # Choose all, lower 1 element (all),
                        # 3 central starting at 2 and upper 2 (all)
    """
    CROP_NONE = None
    CROP_LOWER = 'lower'
    CROP_CENTER = 'center'
    CROP_UPPER = 'upper'

    def __init__(self, incomings, cropping=None, **kwargs):
        super(MergeCropLayer, self).__init__(incomings, **kwargs)
        self.cropping = cropping

    def _crop_input_shapes(self, input_shapes):
        if self.cropping is None:
            return input_shapes
        else:
            result = []

            # If there are more dimensions than cropping entries, pad
            # the cropping
            ndim = len(input_shapes[0])
            cropping = list(self.cropping)
            if ndim > len(cropping):
                cropping = list(cropping) + \
                             [None] * (ndim - len(cropping))

            for sh, cr in zip(zip(*input_shapes), cropping):
                if cr is None:
                    result.append(sh)
                elif cr in {MergeCropLayer.CROP_LOWER,
                            MergeCropLayer.CROP_CENTER,
                            MergeCropLayer.CROP_UPPER}:
                    result.append([min(sh)] * len(sh))
                else:
                    raise ValueError('Unknown crop mode \'{0}\''.format(cr))
            return zip(*result)

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

    cropping : None or [crop]
        Cropping for each input axis. Cropping is described in the docstring
        for `MergeCropLayer`
    """
    def __init__(self, incomings, axis=1, cropping=None, **kwargs):
        if cropping is not None:
            # If cropping is enabled, don't crop on the selected axis
            cropping = list(cropping)
            cropping[axis] = MergeCropLayer.CROP_NONE
        super(ConcatLayer, self).__init__(incomings, cropping=cropping, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shapes):
        input_shapes = self._crop_input_shapes(input_shapes)
        sizes = [input_shape[self.axis] for input_shape in input_shapes]
        output_shape = list(input_shapes[0])  # make a mutable copy
        output_shape[self.axis] = sum(sizes)
        return tuple(output_shape)

    def get_output_for(self, inputs, **kwargs):
        inputs = self._crop_inputs(inputs)
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

    cropping : None or [crop]
        Cropping for each input axis. Cropping is described in the docstring
        for `MergeCropLayer`

    See Also
    --------
    ElemwiseSumLayer : Shortcut for sum layer.
    """

    def __init__(self, incomings, merge_function, cropping=None, **kwargs):
        super(ElemwiseMergeLayer, self).__init__(incomings, **kwargs)
        self.merge_function = merge_function
        self.cropping = cropping

    def get_output_shape_for(self, input_shapes):
        input_shapes = self._crop_input_shapes(input_shapes)
        if any(shape != input_shapes[0] for shape in input_shapes):
            raise ValueError("Mismatch: not all input shapes are the same")
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        inputs = self._crop_inputs(inputs)
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

    cropping : None or [crop]
        Cropping for each input axis. Cropping is described in the docstring
        for `MergeCropLayer`

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
