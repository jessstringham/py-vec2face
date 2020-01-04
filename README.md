
# Vec2Face

Functions that illustrate a vector as a drawing of a face.

The functions are derived from the [Helen dataset](http://www.ifp.illinois.edu/~vuongle2/helen/) [1]. This project is inspired by [Chernoff Faces](https://en.wikipedia.org/wiki/Chernoff_face). The approach for creating the functions is explained in [this blog post](https://jessicastringham.net/2019/11/06/learning-chernoff-faces/)

In theory, vectors near each other will have similar faces.

## Usage

The package includes pretrained functions for illustrating data points.

Initialize by providing the number of dimensions of the data points in the source dataset.

```python
v2f = Vec2Face(dims=16)
v2f.draw_vec(vec)
```

## Dev

    pip install -e .
    pip install -r requirements-dev.txt


[1] Interactive Facial Feature Localization Vuong Le, Jonathan Brandt, Zhe Lin, Lubomir Boudev, Thomas S. Huang. ECCV2012
