import numpy as np

def load_affine(slices):
    first_header = slices[0]

    # Position
    origin = np.array(first_header.ImagePositionPatient)
    endpoint = np.array(slices[-1].ImagePositionPatient)
    z_positions = [header.ImagePositionPatient[2] for header in slices]

    # Scale
    spacing_x, spacing_y = first_header.PixelSpacing
    spacing_z = np.abs(np.median(np.diff(z_positions)))
    spacing = np.diag([spacing_x, spacing_y, spacing_z])

    # Rotation
    orientation_x = np.array(first_header.ImageOrientationPatient[:3])
    orientation_y = np.array(first_header.ImageOrientationPatient[3:])
    orientation_z = ((endpoint - origin) / (len(slices) - 1)) / spacing_z
    orientation = np.array([orientation_x, orientation_y, orientation_z])

    # Affine
    rotation_scale = np.dot(orientation, spacing)
    homogeneous_row = np.array([0, 0, 0, 1]).reshape(1, 4)
    affine = np.r_[np.c_[rotation_scale, origin], homogeneous_row]

    return affine
