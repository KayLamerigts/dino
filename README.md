# Dino
Lightweight Dicom Numpy Operations in pure Python

# Usage

```python
import dino as dn

slices = load_pydicom_slice_from_series()
image = dn.create_image(slices)
```

# Development

```bash
$ pip install .
$ black .
$ isort .
$ python -m unittest discover -s .
$ python testing/watchdog_dev.py
```

# Terms

(Oblique) Cartesian Coordinate System: It's a Cartesian coordinate system where the axes are perpendicular to each other (orthogonal) but are not necessarily aligned with the standard Cartesian axes (x, y, z).

- affine_matrix @ [*point, 1]
