from mock import Mock
import numpy
import pytest
import theano


class TestMergeCropLayer:
    # Test internal helper methods of MergeCropLayer
    def test_merge_shapes(self):
        from lasagne.layers.merge import MergeCropLayer
        crop0 = MergeCropLayer([], cropping=None)
        crop1 = MergeCropLayer([], cropping=[
            MergeCropLayer.CROP_NONE,
            MergeCropLayer.CROP_LOWER,
            MergeCropLayer.CROP_CENTER,
            MergeCropLayer.CROP_UPPER
        ])

        assert crop0._merge_input_shapes(
            [(1, 2, 3, 4), (5, 6, 7, 8), (5, 4, 3, 2)]) == (1, 2, 3, 4)
        assert crop1._merge_input_shapes(
            [(1, 2, 3, 4), (5, 6, 7, 8), (5, 4, 3, 2)]) == (1, 2, 3, 2)

    def test_crop_inputs(self):
        from lasagne.layers.merge import MergeCropLayer
        from numpy.testing import assert_array_equal
        crop_0 = MergeCropLayer([], cropping=None)
        crop_1 = MergeCropLayer([], cropping=[
            MergeCropLayer.CROP_NONE,
            MergeCropLayer.CROP_LOWER,
            MergeCropLayer.CROP_CENTER,
            MergeCropLayer.CROP_UPPER
        ])
        crop_l = MergeCropLayer([], cropping=[
            MergeCropLayer.CROP_LOWER,
            MergeCropLayer.CROP_LOWER,
            MergeCropLayer.CROP_LOWER,
            MergeCropLayer.CROP_LOWER
        ])
        crop_c = MergeCropLayer([], cropping=[
            MergeCropLayer.CROP_CENTER,
            MergeCropLayer.CROP_CENTER,
            MergeCropLayer.CROP_CENTER,
            MergeCropLayer.CROP_CENTER
        ])
        crop_u = MergeCropLayer([], cropping=[
            MergeCropLayer.CROP_UPPER,
            MergeCropLayer.CROP_UPPER,
            MergeCropLayer.CROP_UPPER,
            MergeCropLayer.CROP_UPPER
        ])

        x0 = numpy.random.random((2, 3, 5, 7))
        x1 = numpy.random.random((1, 2, 3, 4))
        x2 = numpy.random.random((6, 3, 4, 2))

        def crop_test(cr, inputs, expected):
            inputs = [theano.shared(x) for x in inputs]
            outs = cr._crop_inputs(inputs)
            outs = [o.eval() for o in outs]
            assert len(outs) == len(expected)
            for o, e in zip(outs, expected):
                assert_array_equal(o, e)

        crop_test(crop_0, [x0, x1],
                  [x0, x1])
        crop_test(crop_1, [x0, x1],
                  [x0[:, :2, 1:4, 3:], x1[:, :, :, :]])
        crop_test(crop_l, [x0, x1],
                  [x0[:1, :2, :3, :4], x1[:, :, :, :]])
        crop_test(crop_c, [x0, x1],
                  [x0[:1, :2, 1:4, 1:5], x1[:, :, :, :]])
        crop_test(crop_u, [x0, x1],
                  [x0[1:, 1:, 2:, 3:], x1[:, :, :, :]])

        crop_test(crop_0, [x0, x2],
                  [x0, x2])
        crop_test(crop_1, [x0, x2],
                  [x0[:, :, :4, 5:], x2[:, :, :, :]])
        crop_test(crop_l, [x0, x2],
                  [x0[:, :, :4, :2], x2[:2, :, :, :]])
        crop_test(crop_c, [x0, x2],
                  [x0[:, :, :4, 2:4], x2[2:4, :, :, :]])
        crop_test(crop_u, [x0, x2],
                  [x0[:, :, 1:, 5:], x2[4:, :, :, :]])

        crop_test(crop_0, [x0, x1, x2],
                  [x0, x1, x2])
        crop_test(crop_1, [x0, x1, x2],
                  [x0[:, :2, 1:4, 5:], x1[:, :, :, 2:], x2[:, :2, :3, :]])
        crop_test(crop_l, [x0, x1, x2],
                  [x0[:1, :2, :3, :2], x1[:, :, :, :2], x2[:1, :2, :3, :]])
        crop_test(crop_c, [x0, x1, x2],
                  [x0[:1, :2, 1:4, 2:4], x1[:, :, :, 1:3], x2[2:3, :2, :3, :]])
        crop_test(crop_u, [x0, x1, x2],
                  [x0[1:, 1:, 2:, 5:], x1[:, :, :, 2:], x2[5:, 1:, 1:, :]])


class TestConcatLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.merge import ConcatLayer
        return ConcatLayer([Mock(), Mock()], axis=1)

    def test_get_output_shape_for(self, layer):
        input_shapes = [(3, 2), (3, 5)]
        result = layer.get_output_shape_for(input_shapes)
        assert result == (3, 7)

    def test_get_output_for(self, layer):
        inputs = [theano.shared(numpy.ones((3, 3))),
                  theano.shared(numpy.ones((3, 2)))]
        result = layer.get_output_for(inputs)
        result_eval = result.eval()
        desired_result = numpy.hstack([input.get_value() for input in inputs])
        assert (result_eval == desired_result).all()


class TestElemwiseSumLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.merge import ElemwiseSumLayer
        return ElemwiseSumLayer([Mock(), Mock()], coeffs=[2, -1])

    def test_get_output_for(self, layer):
        a = numpy.array([[0, 1], [2, 3]])
        b = numpy.array([[1, 2], [4, 5]])
        inputs = [theano.shared(a),
                  theano.shared(b)]
        result = layer.get_output_for(inputs)
        result_eval = result.eval()
        desired_result = 2*a - b
        assert (result_eval == desired_result).all()

    def test_bad_coeffs_fails(self, layer):
        from lasagne.layers.merge import ElemwiseSumLayer
        with pytest.raises(ValueError):
            ElemwiseSumLayer([Mock(), Mock()], coeffs=[2, 3, -1])

    def test_get_output_shape_for_fails(self, layer):
        input_shapes = [(3, 2), (3, 5)]
        with pytest.raises(ValueError):
            layer.get_output_shape_for(input_shapes)


class TestElemwiseMergeLayerMul:
    @pytest.fixture
    def layer(self):
        import theano.tensor as T
        from lasagne.layers.merge import ElemwiseMergeLayer
        return ElemwiseMergeLayer([Mock(), Mock()], merge_function=T.mul)

    def test_get_output_for(self, layer):
        a = numpy.array([[0, 1], [2, 3]])
        b = numpy.array([[1, 2], [4, 5]])
        inputs = [theano.shared(a),
                  theano.shared(b)]
        result = layer.get_output_for(inputs)
        result_eval = result.eval()
        desired_result = a*b
        assert (result_eval == desired_result).all()


class TestElemwiseMergeLayerMaximum:
    @pytest.fixture
    def layer(self):
        import theano.tensor as T
        from lasagne.layers.merge import ElemwiseMergeLayer
        return ElemwiseMergeLayer([Mock(), Mock()], merge_function=T.maximum)

    def test_get_output_for(self, layer):
        a = numpy.array([[0, 1], [2, 3]])
        b = numpy.array([[1, 2], [4, 5]])
        inputs = [theano.shared(a),
                  theano.shared(b)]
        result = layer.get_output_for(inputs)
        result_eval = result.eval()
        desired_result = numpy.maximum(a, b)
        assert (result_eval == desired_result).all()
