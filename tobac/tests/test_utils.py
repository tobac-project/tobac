import numpy as np
import tobac.utils as tb_utils


def test_spectral_filtering():
    """Testing tobac.utils.spectral_filtering with random test data."""

    # generate 3D data with random values
    random_data = np.random.rand(100, 300, 500)

    # define grid spacing [m] and minimum and maximum wavelength [km]
    dxy = 4000
    lambda_min, lambda_max = 400, 1000

    # use spectral filtering function on random data
    transfer_function, filtered_data = tb_utils.spectral_filtering(
        dxy, random_data, lambda_min, lambda_max, return_transfer_function=True
    )

    # a few checks on the output
    wavelengths = transfer_function[0]
    # first element in wavelengths-space is inf because normalized wavelengths are 0 here
    assert wavelengths[0, 0] == np.inf
    # the first elements should correspond to twice the distance of the corresponding axis (in km)
    assert wavelengths[1, 0] == (dxy / 1000) * random_data.shape[-2] * 2
    assert wavelengths[0, 1] == (dxy / 1000) * random_data.shape[-1] * 2

    # filtered/ smoothed field
    assert (filtered_data.max() - filtered_data.min()) < (
        random_data.max() - random_data.min()
    )
