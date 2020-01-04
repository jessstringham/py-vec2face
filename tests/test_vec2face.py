import numpy as np
import pytest

import vec2face


class TestLinearVec2Face():
    def test_smoke_test_draw(self):
        for dims in vec2face.LinearVec2Face.list_supported_dims():
            v2f = vec2face.LinearVec2Face(dims=dims)
            vec = np.zeros(dims)
            v2f.draw_vec(vec)

    def test_data(self):
        for dims in vec2face.LinearVec2Face.list_supported_dims():
            v2f = vec2face.LinearVec2Face(dims=dims)
            vec = np.zeros(dims)
            data = v2f.data(vec)

    def test_set_source_params(self):
        dims = 16
        v2f = vec2face.LinearVec2Face(dims=dims)
        v2f.learn_source_params(np.random.randn(100, dims) + 10)
        vec = np.zeros(dims)
        # make sure we can still use the source params
        data = v2f.data(vec)

    def test_invalid_dims(self):
        # Get all supported dims
        supported_dims = vec2face.LinearVec2Face.list_supported_dims()
        unsupported_dims = max(supported_dims) + 1
        with pytest.raises(NotImplementedError):
            vec2face.LinearVec2Face(unsupported_dims)

    def test_set_scale(self):
        dims = 16
        v2f = vec2face.LinearVec2Face(dims=dims)
        vec = np.zeros(dims)
        v2f.draw_vec(vec, scale=10, offset_x=10, offset_y=10)


class TestVec2Face():
    def test_smoke_test(self):
        for dims in vec2face.LinearVec2Face.list_supported_dims():
            v2f = vec2face.Vec2Face(dims=dims)
            vec = np.zeros(dims)
            v2f.draw_vec(vec)
