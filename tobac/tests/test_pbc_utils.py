import pytest
import numpy as np
import tobac.utils.periodic_boundaries as pbc_utils
import tobac.testing as tb_test


def test_calc_distance_coords_pbc_2D():
    """Tests ```tobac.utils.calc_distance_coords_pbc```
    Currently tests:
    two points in normal space
    Periodicity along hdim_1, hdim_2, and corners
    """
    import numpy as np

    # Test first two points in normal space with varying PBC conditions
    for max_dims in [
        np.array([0, 0]),
        np.array([10, 0]),
        np.array([0, 10]),
        np.array([10, 10]),
    ]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0)), np.array((0, 0)), max_dims
        ) == pytest.approx(0)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0)), np.array((0, 1)), max_dims
        ) == pytest.approx(1)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0)), np.array((3, 1)), max_dims
        ) == pytest.approx(10**0.5, rel=1e-3)

    # Now test two points that will be closer along the hdim_1 boundary for cases without PBCs
    for max_dims in [np.array([10, 0]), np.array([10, 10])]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0)), np.array((9, 0)), max_dims
        ) == pytest.approx(1)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((9, 0)), np.array((0, 0)), max_dims
        ) == pytest.approx(1)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 4)), np.array((7, 3)), max_dims
        ) == pytest.approx(10**0.5)

    # Test the same points, except without PBCs
    for max_dims in [np.array([0, 0]), np.array([0, 10])]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0)), np.array((9, 0)), max_dims
        ) == pytest.approx(9)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((9, 0)), np.array((0, 0)), max_dims
        ) == pytest.approx(9)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 4)), np.array((7, 3)), max_dims
        ) == pytest.approx(50**0.5)

    # Now test two points that will be closer along the hdim_2 boundary for cases without PBCs
    for max_dims in [np.array([0, 10]), np.array([10, 10])]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0)), np.array((0, 9)), max_dims
        ) == pytest.approx(1)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 9)), np.array((0, 0)), max_dims
        ) == pytest.approx(1)

    # Test the same points, except without PBCs
    for max_dims in [np.array([0, 0]), np.array([10, 0])]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0)), np.array((0, 9)), max_dims
        ) == pytest.approx(9)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 9)), np.array((0, 0)), max_dims
        ) == pytest.approx(9)

    # Test points that will be closer for the both
    max_dims = np.array([10, 10])
    assert pbc_utils.calc_distance_coords_pbc(
        np.array((9, 9)), np.array((0, 0)), max_dims
    ) == pytest.approx(1.4142135, rel=1e-3)
    assert pbc_utils.calc_distance_coords_pbc(
        np.array((0, 9)), np.array((9, 0)), max_dims
    ) == pytest.approx(1.4142135, rel=1e-3)

    # Test the corner points for no PBCs
    max_dims = np.array([0, 0])
    assert pbc_utils.calc_distance_coords_pbc(
        np.array((9, 9)), np.array((0, 0)), max_dims
    ) == pytest.approx(12.727922, rel=1e-3)
    assert pbc_utils.calc_distance_coords_pbc(
        np.array((0, 9)), np.array((9, 0)), max_dims
    ) == pytest.approx(12.727922, rel=1e-3)

    # Test the corner points for hdim_1 and hdim_2
    for max_dims in [np.array([10, 0]), np.array([0, 10])]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((9, 9)), np.array((0, 0)), max_dims
        ) == pytest.approx(9.055385)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 9)), np.array((9, 0)), max_dims
        ) == pytest.approx(9.055385)


def test_calc_distance_coords_pbc_3D():
    """Tests ```tobac.utils.calc_distance_coords_pbc```
    Currently tests:
    two points in normal space
    Periodicity along hdim_1, hdim_2, and corners
    """
    import numpy as np

    # Test first two points in normal space with varying PBC conditions
    for max_dims in [
        np.array([0, 0, 0]),
        np.array([0, 10, 0]),
        np.array([0, 0, 10]),
        np.array([0, 10, 10]),
    ]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 0)), np.array((0, 0, 0)), max_dims
        ) == pytest.approx(0)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 0)), np.array((0, 0, 1)), max_dims
        ) == pytest.approx(1)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 0)), np.array((3, 3, 1)), max_dims
        ) == pytest.approx(4.3588989, rel=1e-3)

    # Now test two points that will be closer along the hdim_1 boundary for cases without PBCs
    for max_dims in [np.array([0, 10, 0]), np.array([0, 10, 10])]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 0)), np.array((0, 9, 0)), max_dims
        ) == pytest.approx(1)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 9, 0)), np.array((0, 0, 0)), max_dims
        ) == pytest.approx(1)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((4, 0, 4)), np.array((3, 7, 3)), max_dims
        ) == pytest.approx(3.3166247)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((4, 0, 4)), np.array((3, 7, 3)), max_dims
        ) == pytest.approx(3.3166247)

    # Test the same points, except without PBCs
    for max_dims in [np.array([0, 0, 0]), np.array([0, 0, 10])]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 0)), np.array((0, 9, 0)), max_dims
        ) == pytest.approx(9)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 9, 0)), np.array((0, 0, 0)), max_dims
        ) == pytest.approx(9)

    # Now test two points that will be closer along the hdim_2 boundary for cases without PBCs
    for max_dims in [np.array([0, 0, 10]), np.array([0, 10, 10])]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 0)), np.array((0, 0, 9)), max_dims
        ) == pytest.approx(1)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 9)), np.array((0, 0, 0)), max_dims
        ) == pytest.approx(1)

    # Test the same points, except without PBCs
    for max_dims in [np.array([0, 0, 0]), np.array([0, 10, 0])]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 0)), np.array((0, 0, 9)), max_dims
        ) == pytest.approx(9)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 9)), np.array((0, 0, 0)), max_dims
        ) == pytest.approx(9)

    # Test points that will be closer for the both
    max_dims = np.array([0, 10, 10])
    assert pbc_utils.calc_distance_coords_pbc(
        np.array((0, 9, 9)), np.array((0, 0, 0)), max_dims
    ) == pytest.approx(1.4142135, rel=1e-3)
    assert pbc_utils.calc_distance_coords_pbc(
        np.array((0, 0, 9)), np.array((0, 9, 0)), max_dims
    ) == pytest.approx(1.4142135, rel=1e-3)

    # Test the corner points for no PBCs
    max_dims = np.array([0, 0, 0])
    assert pbc_utils.calc_distance_coords_pbc(
        np.array((0, 9, 9)), np.array((0, 0, 0)), max_dims
    ) == pytest.approx(12.727922, rel=1e-3)
    assert pbc_utils.calc_distance_coords_pbc(
        np.array((0, 0, 9)), np.array((0, 9, 0)), max_dims
    ) == pytest.approx(12.727922, rel=1e-3)

    # Test the corner points for hdim_1 and hdim_2
    for max_dims in [np.array([0, 10, 0]), np.array([0, 0, 10])]:
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 9, 9)), np.array((0, 0, 0)), max_dims
        ) == pytest.approx(9.055385)
        assert pbc_utils.calc_distance_coords_pbc(
            np.array((0, 0, 9)), np.array((0, 9, 0)), max_dims
        ) == pytest.approx(9.055385)


def test_invalid_limit_names():
    """
    Test that invalid_limit_names will correctly return the names of keywords that equal None
    """
    assert pbc_utils.invalid_limit_names() == []
    assert pbc_utils.invalid_limit_names(a=0, b=0) == []
    assert pbc_utils.invalid_limit_names(a=None, b=0) == ["a"]
    assert pbc_utils.invalid_limit_names(a=0, b=None) == ["b"]
    assert pbc_utils.invalid_limit_names(a=None, b=None) == ["a", "b"]


def test_validate_pbc_dims():
    """
    Test validate_pbc_dims works correctly and raise exceptions appropriately
    """
    # Assert that size is only returned if PBCs are required in that dimension
    assert pbc_utils.validate_pbc_dims(0, 10, 0, 15, "none") == (0, 0)
    assert pbc_utils.validate_pbc_dims(0, 10, 0, 15, "hdim_1") == (10, 0)
    assert pbc_utils.validate_pbc_dims(0, 10, 0, 15, "hdim_2") == (0, 15)
    assert pbc_utils.validate_pbc_dims(0, 10, 0, 15, "both") == (10, 15)

    # Assert that giving None for limits of dimensions that are not PBCs is handled correctly
    assert pbc_utils.validate_pbc_dims(None, None, None, None, "none") == (0, 0)
    assert pbc_utils.validate_pbc_dims(0, 10, None, None, "hdim_1") == (10, 0)
    assert pbc_utils.validate_pbc_dims(None, None, 0, 15, "hdim_2") == (0, 15)

    # Test that an invalid option for PBC_flag raises the correct error
    with pytest.raises(pbc_utils.PBCflagError):
        _ = pbc_utils.validate_pbc_dims(0, 10, 0, 15, "invalid_pbc_option")

    # Test that providing None for the limits of a PBC dimension raises the correct error
    with pytest.raises(pbc_utils.PBCLimitError):
        _ = pbc_utils.validate_pbc_dims(None, None, 0, 15, "hdim_1")
    with pytest.raises(pbc_utils.PBCLimitError):
        _ = pbc_utils.validate_pbc_dims(None, None, None, 15, "hdim_2")
    with pytest.raises(pbc_utils.PBCLimitError):
        _ = pbc_utils.validate_pbc_dims(0, None, 0, None, "both")


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
    h1_size, h2_size = pbc_utils.validate_pbc_dims(
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        PBC_flag,
    )

    if len(loc_1) == 3:
        max_dims = np.array([0, h1_size, h2_size])
    else:
        max_dims = np.array([h1_size, h2_size])

    assert pbc_utils.calc_distance_coords_pbc(
        np.array(loc_1),
        np.array(loc_2),
        max_dims,
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


def test_build_distance_function_3D():
    """Tests ```pbc_utils.build_distance_function```
    Currently tests:
    that this produces an object that is suitable to call from trackpy
    """

    test_func = pbc_utils.build_distance_function(0, 10, 0, 10, "both", True)
    assert test_func(np.array((0, 9, 9)), np.array((0, 0, 0))) == pytest.approx(
        1.4142135
    )

    test_func = pbc_utils.build_distance_function(0, 10, None, None, "hdim_1", True)
    assert test_func(np.array((0, 9, 9)), np.array((0, 0, 9))) == pytest.approx(1)

    test_func = pbc_utils.build_distance_function(None, None, 0, 10, "hdim_2", True)
    assert test_func(np.array((0, 9, 9)), np.array((0, 9, 0))) == pytest.approx(1)

    test_func = pbc_utils.build_distance_function(None, None, None, None, "none", True)
    assert test_func(np.array((0, 9, 9)), np.array((0, 0, 0))) == pytest.approx(
        (2 * 81) ** 0.5
    )


def test_build_distance_function_2D():
    """Tests ```pbc_utils.build_distance_function```
    Currently tests:
    that this produces an object that is suitable to call from trackpy
    """

    test_func = pbc_utils.build_distance_function(0, 10, 0, 10, "both", False)
    assert test_func(np.array((9, 9)), np.array((0, 0))) == pytest.approx(1.4142135)

    test_func = pbc_utils.build_distance_function(0, 10, None, None, "hdim_1", False)
    assert test_func(np.array((9, 9)), np.array((0, 9))) == pytest.approx(1)

    test_func = pbc_utils.build_distance_function(None, None, 0, 10, "hdim_2", False)
    assert test_func(np.array((9, 9)), np.array((9, 0))) == pytest.approx(1)

    test_func = pbc_utils.build_distance_function(None, None, None, None, "none", False)
    assert test_func(np.array((9, 9)), np.array((0, 0))) == pytest.approx(
        (2 * 81) ** 0.5
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
