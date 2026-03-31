import numpy as np
from grid import Grid, UniformCartesianGrid

def test_get_axis():
    # Create a grid with 2 dimensions and 3 points in each dimension
    grid = UniformCartesianGrid(2,  limits=[[0, 1], [0, 1]], nptx=[3, 3])

    # Call the get_axis method
    axis = grid.get_axis()

    # Check that the axis is a list of 2 arrays, each with 3 values
    assert len(axis) == 2
    for a in axis:
        assert len(a) == 3

    # Check that the values of the axis are correct
    expected_axis = [
        np.array([0, 0.3333333333333333, 0.6666666666666666]),
        np.array([0, 0.3333333333333333, 0.6666666666666666])
    ]
    for i in range(len(axis)):
        assert np.allclose(axis[i], expected_axis[i])

    # same test for a 1D grid

    grid = UniformCartesianGrid(1,  limits=[0, 1], nptx=3)

    # Call the get_axis method
    axis = grid.get_axis()

    # Check that the axis is a list of 2 arrays, each with 3 values
    assert len(axis) ==1

    # Check that the values of the axis are correct
    expected_axis = [
        np.array([[0, 0.3333333333333333, 0.6666666666666666]])
    ]
    for i in range(len(axis)):
        assert np.allclose(axis[i], expected_axis[i])

def test_get_x():
    grid=UniformCartesianGrid(2,  limits=[[0, 1], [0, 1]], nptx=[3, 3])

    x=grid.get_x()
    assert x.shape==(9,2)
    assert np.allclose(x,[[0,0],[0, 0.33333333],[0, 0.66666667],[0.33333333, 0],[0.33333333,0.33333333],[0.33333333, 0.66666667],[0.66666667,0],[0.66666667, 0.33333333],[0.66666667,0.66666667]])

    # same test for a 1D grid

    grid=UniformCartesianGrid(1,  limits=[0, 1], nptx=3)

    x=grid.get_x()
    assert x.shape==(3,1)
    assert np.allclose(x,[[0],[0.33333333],[0.66666667]])

def test_get_dx():
    grid=UniformCartesianGrid(2,  limits=[[0, 1], [0, 1]], nptx=[3, 3])

    dx=grid.get_dx()
    assert np.allclose(dx,[0.3333333333333333, 0.3333333333333333])

    # same test for a 1D grid

    grid=UniformCartesianGrid(1,  limits=[0, 1], nptx=3)

    dx=grid.get_dx()
    assert dx==[0.3333333333333333]
def test_get_dV():
    grid=UniformCartesianGrid(2,  limits=[[0, 1], [0, 1]], nptx=[3, 3])

    dV=grid.get_dV()
    assert np.allclose(dV[:], 0.3333333333333333**2)

    # same test for a 1D grid

    grid=UniformCartesianGrid(1,  limits=[0, 1], nptx=3)

    dV=grid.get_dV()
    assert np.allclose(dV[:], 0.3333333333333333)

def test_get_inGrid():
    def filter(x):
        x_test=np.zeros(x.shape[0], dtype=bool)
        for n, ix in enumerate(x):
            x_test[n]=ix[0]>0.5 and ix[1]<0.5 
        return x_test

    def filter1d(x):
        x_test=np.zeros(x.shape[0], dtype=bool)
        for n, ix in enumerate(x):
            x_test[n]=ix[0]>0.5 
        return x_test
    
    grid=UniformCartesianGrid(2,  limits=[[0, 1], [0, 1]], nptx=[3, 3], filter=filter)

    inGrid=grid.get_inGrid()

    assert np.array_equal(inGrid, np.array([False, False, False, False, False, False, True, True, False]))

    #  the same test for a 1D grid

    grid=UniformCartesianGrid(1,  limits=[0, 1], nptx=3, filter=filter1d)

    inGrid=grid.get_inGrid()

    assert np.array_equal(inGrid, np.array([False, False, True]))   

def test_subSet():
    grid=UniformCartesianGrid(2,  limits=[[0, 1], [0, 1]], nptx=[3, 3])

    subSet=grid.subSet([slice(None, None, 2), slice(None, None, 2)])

    assert subSet.shape==(4,2)
    assert np.allclose(subSet,[[0,0],[0.0,0.66666667],[0.66666667,0.0],[0.66666667,0.66666667]])

    subSet=grid.subSet([slice(None, 1, 1), slice(None, None, 1)])

    assert np.allclose(subSet,[[0,0],[0,0.33333333],[0,0.66666667]])

    # same test for a 1D grid

    grid=UniformCartesianGrid(1,  limits=[0, 1], nptx=3)

    subSet=grid.subSet([slice(None, None, 2)])

    assert subSet.shape==(2,1)
    assert np.allclose(subSet,[[0],[0.66666667]])

def test_subSet_using_filter():
    def filter(x):
        x_test=np.zeros(x.shape[0], dtype=bool)
        for n, ix in enumerate(x):
            x_test[n]=ix[0]>0.5 and ix[1]<0.5 
        
        if x.shape[0]==1:
            x_test=x_test[0]
        
        return x_test
    
    def filter1d(x):
        x_test=np.zeros(x.shape[0], dtype=bool)
        for n, ix in enumerate(x):
            x_test[n]=ix[0]>0.5

        return x_test

    grid=UniformCartesianGrid(2,  limits=[[0, 1], [0, 1]], nptx=[3, 3], filter=filter)

    subSet=grid.subSet([slice(None, None, 2), slice(None, None, 2)])

    assert subSet.shape==(1,2)
    # assert np.allclose(subSet,[[0,0],[0.0,0.66666667],[0.66666667,0.0],[0.66666667,0.66666667]])
    assert np.allclose(subSet,[[0.66666667,0.0]])

    # test for a 1D grid

    grid=UniformCartesianGrid(1,  limits=[0, 1], nptx=3, filter=filter1d)

    subSet=grid.subSet([slice(None, None, 2)])

    assert subSet.shape==(1,1)
    assert np.allclose(subSet,[[0.66666667]])
