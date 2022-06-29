import numpy as np
import tobac.utils as tb_utils
from scipy import fft

def test_spectral_filtering():
    """Testing tobac.utils.spectral_filtering with random test data that contains a wave signal."""

    # set wavelengths for filtering and grid spacing
    dxy = 4000
    lambda_min  = 400*1000
    lambda_max = 1000*1000

    # get wavelengths for domain 
    matrix= np.zeros((200,100))
    Ni = matrix.shape[-2]
    Nj = matrix.shape[-1]
    m, n = np.meshgrid(np.arange(Ni), np.arange(Nj), indexing="ij")
    alpha = np.sqrt(m**2 / Ni**2 + n**2 / Nj**2)
    lambda_mn = 2 * dxy / alpha

    # seed wave signal that lies within wavelength range for filtering
    signal_min = np.where(lambda_mn[0] < lambda_min)[0].min()
    signal_idx = np.random.randint(signal_min, matrix.shape[-1])
    matrix[0,signal_idx] = 1
    wave_data = fft.idctn(matrix)

    # use spectral filtering function on random wave data
    transfer_function, filtered_data = tb_utils.spectral_filtering(
        dxy, wave_data, lambda_min, lambda_max, return_transfer_function=True
    )

    # a few checks on the output
    wavelengths = transfer_function[0]
    # first element in wavelengths-space is inf because normalized wavelengths are 0 here
    assert wavelengths[0, 0] == np.inf
    # the first elements should correspond to twice the distance of the corresponding axis (in m)
    # this is because the maximum spatial scale is half a wavelength through the domain
    assert wavelengths[1, 0] == (dxy) * wave_data.shape[-2] * 2
    assert wavelengths[0, 1] == (dxy) * wave_data.shape[-1] * 2

    # check that filtered/ smoothed field exhibits smaller range of values
    assert (filtered_data.max() - filtered_data.min()) < (
        random_data.max() - random_data.min()
    )

    # because the randomly generated wave lies outside of range that is set for filtering,
    # make sure that the filtering results in the disappearance of this signal 
    assert abs( np.floor(np.log10(filtered_data.mean())) - np.floor(np.log10(random_data.mean()))  )   >= 1 

