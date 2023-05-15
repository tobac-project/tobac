import pytest
import numpy as np
import tobac.utils.periodic_boundaries as pbc_utils
import tobac.testing as tb_test


def test_calc_distance_coords_pbc():
    """Tests ```tobac.utils.calc_distance_coords_pbc```
    Currently tests:
    two points in normal space
    Periodicity along hdim_1, hdim_2, and corners
    """
    import numpy as np

    # Test first two points in normal space with varying PBC conditions
    for PBC_condition in ["none", "hdim_1", "hdim_2", "both"]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 0)), np.array((0, 0, 0)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(0)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 0)), np.array((0, 0, 1)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(1)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0)), np.array((0, 1)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(1)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 0)), np.array((3, 3, 1)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(4.3588989, rel=1e-3)

    # Now test two points that will be closer along the hdim_1 boundary for cases without PBCs
    for PBC_condition in ["hdim_1", "both"]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 0)), np.array((0, 9, 0)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(1)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 9, 0)), np.array((0, 0, 0)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(1)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((8, 0)), np.array((0, 0)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(2)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((4, 0, 4)), np.array((3, 7, 3)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(3.3166247)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((4, 0, 4)), np.array((3, 7, 3)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(3.3166247)

    # Test the same points, except without PBCs
    for PBC_condition in ["none", "hdim_2"]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 0)), np.array((0, 9, 0)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(9)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 9, 0)), np.array((0, 0, 0)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(9)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((8, 0)), np.array((0, 0)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(8)

    # Now test two points that will be closer along the hdim_2 boundary for cases without PBCs
    for PBC_condition in ["hdim_2", "both"]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 0)), np.array((0, 0, 9)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(1)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 9)), np.array((0, 0, 0)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(1)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 8)), np.array((0, 0)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(2)

    # Test the same points, except without PBCs
    for PBC_condition in ["none", "hdim_1"]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 0)), np.array((0, 0, 9)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(9)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 9)), np.array((0, 0, 0)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(9)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 8)), np.array((0, 0)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(8)

    # Test points that will be closer for the both
    PBC_condition = "both"
    assert pbc_utils.calc_distance_coords_pbc(
        np.array((0, 9, 9)), np.array((0, 0, 0)), 0, 10, 0, 10, PBC_condition
    ) == pytest.approx(1.4142135, rel=1e-3)
    assert pbc_utils.calc_distance_coords_pbc(
        np.array((0, 0, 9)), np.array((0, 9, 0)), 0, 10, 0, 10, PBC_condition
    ) == pytest.approx(1.4142135, rel=1e-3)

    # Test the corner points for no PBCs
    PBC_condition = "none"
    assert pbc_utils.calc_distance_coords_pbc(
        np.array((0, 9, 9)), np.array((0, 0, 0)), 0, 10, 0, 10, PBC_condition
    ) == pytest.approx(12.727922, rel=1e-3)
    assert pbc_utils.calc_distance_coords_pbc(
        np.array((0, 0, 9)), np.array((0, 9, 0)), 0, 10, 0, 10, PBC_condition
    ) == pytest.approx(12.727922, rel=1e-3)

    # Test the corner points for hdim_1 and hdim_2
    for PBC_condition in ["hdim_1", "hdim_2"]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 9, 9)), np.array((0, 0, 0)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(9.055385)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 9)), np.array((0, 9, 0)), 0, 10, 0, 10, PBC_condition
        ) == pytest.approx(9.055385)


@pytest.mark.parametrize(
    "loc_1, loc_2, bounds, PBC_flag, expected_dist",
    [
        ((0, 0, 0), (0, 0, 9), (0, 10, 0, 10), "both", 1),
    ],
)
def test_calc_distance_coords_pbc_param(loc_1, loc_2, bounds, PBC_flag, expected_dist):
    """Tests ```tobac.utils.calc_distance_coords_pbc``` in a parameterized way

    Parameters
    ----------
    loc_1: tuple
        First point location, either in 2D or 3D space (assumed z, h1, h2)
    loc_2: tuple
        Second point location, either in 2D or 3D space (assumed z, h1, h2)
    bounds: tuple
        hdim_1/hdim_2 bounds as (h1_min, h1_max, h2_min, h2_max)
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions
    expected_dist: float
        Expected distance between the two points
    """
    import numpy as np

    assert pbc_utils.calc_distance_coords_pbc(
        np.array(loc_1),
        np.array(loc_2),
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        PBC_flag,
    ) == pytest.approx(expected_dist)


def test_get_pbc_coordinates():
    """Tests tobac.util.get_pbc_coordinates.
    Currently runs the following tests:
    For an invalid PBC_flag, we raise an error
    For PBC_flag of 'none', we truncate the box and give a valid box.

    """

    with pytest.raises(ValueError):
        pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, 1, 4, "c")

    # Test PBC_flag of none

    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, 1, 4, "none") == [
        (1, 4, 1, 4),
    ]
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, -1, 4, 1, 4, "none") == [
        (0, 4, 1, 4),
    ]
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 12, 1, 4, "none") == [
        (1, 10, 1, 4),
    ]
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 12, -1, 4, "none") == [
        (1, 10, 0, 4),
    ]

    # Test PBC_flag with hdim_1
    # Simple case, no PBC overlapping
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, 1, 4, "hdim_1") == [
        (1, 4, 1, 4),
    ]
    # PBC going on the min side
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, -1, 4, 1, 4, "hdim_1") == [
        (0, 4, 1, 4),
        (9, 10, 1, 4),
    ]
    # PBC going on the min side; should be truncated in hdim_2.
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, -1, 4, -1, 4, "hdim_1") == [
        (0, 4, 0, 4),
        (9, 10, 0, 4),
    ]
    # PBC going on the max side only
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 4, 12, 1, 4, "hdim_1") == [
        (4, 10, 1, 4),
        (0, 2, 1, 4),
    ]
    # PBC overlapping
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, -4, 12, 1, 4, "hdim_1") == [
        (0, 10, 1, 4),
    ]

    # Test PBC_flag with hdim_2
    # Simple case, no PBC overlapping
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, 1, 4, "hdim_2") == [
        (1, 4, 1, 4),
    ]
    # PBC going on the min side
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, -1, 4, "hdim_2") == [
        (1, 4, 0, 4),
        (1, 4, 9, 10),
    ]
    # PBC going on the min side with truncation in hdim_1
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, -4, 4, -1, 4, "hdim_2") == [
        (0, 4, 0, 4),
        (0, 4, 9, 10),
    ]
    # PBC going on the max side
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, 4, 12, "hdim_2") == [
        (1, 4, 4, 10),
        (1, 4, 0, 2),
    ]
    # PBC overlapping
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, -4, 12, "hdim_2") == [
        (1, 4, 0, 10),
    ]

    # Test PBC_flag with both
    # Simple case, no PBC overlapping
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, 1, 4, "both") == [
        (1, 4, 1, 4),
    ]
    # hdim_1 only testing
    # PBC on the min side of hdim_1 only
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, -1, 4, 1, 4, "both") == [
        (0, 4, 1, 4),
        (9, 10, 1, 4),
    ]
    # PBC on the max side of hdim_1 only
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 4, 12, 1, 4, "both") == [
        (4, 10, 1, 4),
        (0, 2, 1, 4),
    ]
    # PBC overlapping on max side of hdim_1 only
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, -4, 12, 1, 4, "both") == [
        (0, 10, 1, 4),
    ]
    # hdim_2 only testing
    # PBC on the min side of hdim_2 only
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, -1, 4, "both") == [
        (1, 4, 0, 4),
        (1, 4, 9, 10),
    ]
    # PBC on the max side of hdim_2 only
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, 4, 12, "both") == [
        (1, 4, 4, 10),
        (1, 4, 0, 2),
    ]
    #  PBC overlapping on max side of hdim_2 only
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, -4, 12, "both") == [
        (1, 4, 0, 10),
    ]
    # hdim_1 and hdim_2 testing simultaneous
    # both larger than the actual domain
    assert pbc_utils.get_pbc_coordinates(0, 10, 0, 10, -1, 12, -4, 14, "both") == [
        (0, 10, 0, 10),
    ]
    # min in hdim_1 and hdim_2
    assert tb_test.lists_equal_without_order(
        pbc_utils.get_pbc_coordinates(0, 10, 0, 10, -3, 3, -4, 2, "both"),
        [(0, 3, 0, 2), (0, 3, 6, 10), (7, 10, 6, 10), (7, 10, 0, 2)],
    )
    # max in hdim_1, min in hdim_2
    assert tb_test.lists_equal_without_order(
        pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 5, 12, -4, 2, "both"),
        [(5, 10, 0, 2), (5, 10, 6, 10), (0, 2, 6, 10), (0, 2, 0, 2)],
    )
    # max in hdim_1 and hdim_2
    assert tb_test.lists_equal_without_order(
        pbc_utils.get_pbc_coordinates(0, 10, 0, 10, 5, 12, 7, 15, "both"),
        [(5, 10, 7, 10), (5, 10, 0, 5), (0, 2, 0, 5), (0, 2, 7, 10)],
    )
    # min in hdim_1, max in hdim_2
    assert tb_test.lists_equal_without_order(
        pbc_utils.get_pbc_coordinates(0, 10, 0, 10, -3, 3, 7, 15, "both"),
        [(0, 3, 7, 10), (0, 3, 0, 5), (7, 10, 0, 5), (7, 10, 7, 10)],
    )


def test_weighted_circmean() -> None:
    """
    Test that weighted_circmean gives the expected results compared to scipy.stats.circmean
    """
    from scipy.stats import circmean

    values = np.arange(0, 12)
    weights = np.ones([12])
    high, low = 5, 0
    assert pbc_utils.weighted_circmean(
        values, weights, high=high, low=low
    ) == pytest.approx(circmean(values, high=high, low=low))

    # Test if weights are set to values other than 1
    assert pbc_utils.weighted_circmean(
        values, weights / 5, high=high, low=low
    ) == pytest.approx(circmean(values, high=high, low=low))
    assert pbc_utils.weighted_circmean(
        values, weights * 7, high=high, low=low
    ) == pytest.approx(circmean(values, high=high, low=low))

    # set some weights to zero
    weights[[4, 7, 9]] = 0
    assert pbc_utils.weighted_circmean(
        values, weights, high=high, low=low
    ) == pytest.approx(circmean(values[weights.astype(bool)], high=high, low=low))

    # Set some non-zero weights
    weights[[4, 7]] = 2
    weights[9] = 3
    duplicated_values = np.concatenate([values, [4, 7, 9, 9]])
    assert pbc_utils.weighted_circmean(
        values, weights, high=high, low=low
    ) == pytest.approx(circmean(duplicated_values, high=high, low=low))
