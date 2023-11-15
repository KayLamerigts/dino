# Dino
Lightweight Dicom Numpy Operations in pure Python


# Development

```bash
$ pip install .
$ black .
$ isort .
$ python -m unittest discover -s .
$ python testing/watch_tests.py dino
```

# Terms

spacing/scale
orientation/rotation
position/translation

size

(Oblique) Cartesian Coordinate System: It's a Cartesian coordinate system where the axes are perpendicular to each other (orthogonal) but are not necessarily aligned with the standard Cartesian axes (x, y, z).


affine_matrix @ [*point, 1]
