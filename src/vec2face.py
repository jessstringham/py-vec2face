from collections import namedtuple
from io import BytesIO
import pkgutil

import matplotlib.pyplot as plt
import numpy as np


EmbeddingParams = namedtuple('EmbeddingParams', ['mean', 'stdev'])

def _clipped_normal_scale(vec, source_params, target_params, clip_number_stdevs=2):
    '''Scales the provided `vec`, which is assumed to come from a normal distribution
    represented by `source_params`, such that it matches the distribution represented
    by `target_params`, and clip extremes that are greater than `clip_number_stdevs`
    away.

    `source_params`, `target_params` should all be based on data with the
    same number of dimensions as `vec`, or use a single dimension to leverage
    broadcasting.

    Parameters
    ----------
    vec : numpy array
        The source vector that should be adjusted.
    source_params : EmbeddingParams
        Representation of distribution of source dataset.
    target_params : EmbeddingParams
        Representation of distribution of target dataset.
    clip_number_stdevs : int
        Number of standard deviations to trim from extremes.

    Returns
    -------
    Vector adjusted from being from the source distribution to the target distribution.
    '''
    normal_source = (vec - source_params.mean)/source_params.stdev

    # First clip the very big extremes
    # Since it's already normalized, just use the number of stdevs
    normal_source = np.clip(normal_source, -clip_number_stdevs, clip_number_stdevs)

    # Then line up with the new one
    return normal_source * target_params.stdev + target_params.mean


_face_components = [
    # chin
    {
        'slice_start': 0,
        'slice_end': 41,
        'closed': False,
    },
    # nose
    {
        'slice_start': 41,
        'slice_end': 58,
        'closed': False,
    },
    # lips
    #{
    #    'slice_start': 58,
    #    'slice_end': 86,
    #    'closed': True,
    #},
    # mouth
    {
        'slice_start': 86,
        'slice_end': 114,
        'closed': True,
    },
    # right_eye
    {
        'slice_start': 114,
        'slice_end': 134,
        'closed': True,
    },
    # left_eye
    {
        'slice_start': 134,
        'slice_end': 154,
        'closed': True,
    },
    # right_eyebrow
    {
        'slice_start': 154,
        'slice_end': 174,
        'closed': True,
    },
    # left_eyebrow
    {
        'slice_start': 174,
        'slice_end': 194,
        'closed': True,
    },
]


class LinearVec2Face(object):
    '''Creates an object that can illustrate a fixed-size numpy array
    representing a given data point.

    Starting with a source dataset where each data point is a vector of
    size `dims`, the object can be used as follows. If one vector is `vec`,
    then it can be plotted with:

        v2f = Vec2Face(dims=dims)
        v2f.draw_vec(vec)

    For more varied faces, normalize the source dataset (i.e. it should have
    a mean of 0 and standard deviation of 1).

    Parameters
    ----------
    dims: int
        Number of dimensions of the data points.
    '''
    params = np.load(BytesIO(pkgutil.get_data(__name__, 'data/linear.npz')))

    def __init__(self, dims):
        self.dims = dims

        # Check if dims is supported
        if 'd{}_w'.format(self.dims) not in self.params:
            supported_dims = self.list_supported_dims()
            raise NotImplementedError(
                '''{classname} does not support dims={dims}. Supported dims are: {supported_dims}'''.format(
                    classname=self.__class__.__name__,
                    dims=dims,
                    supported_dims=', '.join(map(str, sorted(supported_dims)))
                )
            )

        self._weights = self.params['d{}_w'.format(self.dims)]
        self._bias = self.params['d{}_b'.format(self.dims)]
        self._source_params = EmbeddingParams(0, 1)
        self._target_params = EmbeddingParams(
            mean=self.params['d{}_mean'.format(self.dims)],
            stdev=self.params['d{}_std'.format(self.dims)]
        )

    @classmethod
    def list_supported_dims(cls):
        '''Return a list of implemented values for dims.'''
        return [
            int(k.split('_')[0][1:])  # Extract number from 'd#_w'
            for k in cls.params.keys()
            if k.endswith('_w')  # Arbitrarily grab weights params
        ]

    def learn_source_params(self, source_data_sample):
        '''Set the source_params to automatically normalize the source data in
        future calls to draw_vec.

        Parameters
        ----------
        source_data_sample : 2D array
            A 2D array of size N x dims containing a sample of the source data.
        '''
        if source_data_sample.shape[1] != self.dims:
            raise ValueError(
                '`source_data_sample` should have shape (?, {dims})'.format(self.dims))

        self._source_params = EmbeddingParams(
            mean=source_data_sample.mean(axis=0),
            stdev=source_data_sample.std(axis=0)
        )

    def _draw_face(self, data, ax, scale=1, offset_x=0, offset_y=0):
        '''Draw face given data.'''
        offset = np.vstack([offset_x, offset_y]).T

        def draw_line(start, end):
            adjust_data = data[start:end] * scale + offset
            return ax.plot(*adjust_data.T, '-k')

        def draw_closed_line(start, end):
            closed_shape = np.vstack([data[start:end], data[start].reshape(1, 2)])
            adjust_data = closed_shape * scale + offset
            ax.plot(*adjust_data.T, '-k')

        for component in _face_components:
            draw_func = draw_closed_line if component['closed'] else draw_line
            draw_func(component['slice_start'], component['slice_end'])

        # Make sure the axis is configured to look nice.
        ax.axis('equal')
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.axis('off')

    def data(self, vec):
        '''Transform the provided vector into the data needed to illustrate the
        data point.

        Parameters
        ----------
        vec : 1D np.array
            The vector that should be illustrated

        Returns
        -------
        A np.array that can be used with `_draw_face`.
        '''
        if len(vec.shape) != 1 or vec.shape[0] != self.dims:
            raise ValueError(
                '''\
Dims of input vector ({input_shape}) doesn't match configured dims ({dims},)'''.format(
    input_shape=vec.shape[0],
    dims=self.dims)
            )

        scaled_vec = _clipped_normal_scale(vec, self._source_params, self._target_params)
        return (scaled_vec @ self._weights + self._bias).reshape(-1, 2)

    def draw_vec(self, vec, ax=None, scale=1, offset_x=0, offset_y=0):
        '''Illustrate the provided data point.

        Parameters
        ----------
        vec : numpy array
            The vector that should be illustrated
        ax : optional matplotlib axis object
            If None, use plt.gca().
        scale : optional float
            How much to scale the output drawing. Useful if illustrating multiple
            data points on the same ax.
        offset_x : optional float
            How much to translate on the x-axis. Useful if illustrating multiple
            data points on the same ax.
        offset_y : optional float
            How much to translate on the y-axis. Useful if illustrating multiple
            data points on the same ax.
        '''
        if ax is None:
            ax = plt.gca()

        data = self.data(vec)
        self._draw_face(data, ax, scale=scale, offset_x=offset_x, offset_y=offset_y)


# Main public interface
Vec2Face = LinearVec2Face  # Alias Vec2Face
