import numpy as np
from grid import Grid, UniformCylindricalGrid

def test_get_axis():
    grid=UniformCylindricalGrid(3,  limits=[[0, 1], [0, 2*np.pi], [0,1]], nptx=[4, 4, 4])

    axis=grid.get_axis()

    assert np.allclose(axis[0], [0, 0.25, 0.5, 0.75])
    assert np.allclose(axis[1], [0, 0.5*np.pi, 1*np.pi, 1.5*np.pi])
    assert np.allclose(axis[2], [0, 0.25, 0.5, 0.75])

    grid=UniformCylindricalGrid(3,  limits=[[0, 1], [0, 2*np.pi], [0,1]], origin=(0.1, 0.05*np.pi,0),nptx=[4, 4, 4])

    axis=grid.get_axis()

    assert np.allclose(axis[0], [0.1, 0.35, 0.6, 0.85])
    assert np.allclose(axis[1], [0.05*np.pi, 0.5*np.pi+0.05*np.pi, 1*np.pi+0.05*np.pi, 1.5*np.pi+0.05*np.pi])
    assert np.allclose(axis[2], [0, 0.25, 0.5, 0.75])

    # same test for a 1D grid

    grid=UniformCylindricalGrid(1,  limits=[0,1], nptx=4)

    axis=grid.get_axis()

    assert np.allclose(axis[0], [0, 0.25, 0.5, 0.75])

def test_get_x():
    grid=UniformCylindricalGrid(3,  limits=[[0, 1], [0, 2*np.pi], [0,1]], nptx=[2, 2, 2])

    x=grid.get_x()

    assert np.allclose(x[0], [0, 0, 0])
    assert np.allclose(x[1], [0, 0, 0.5])
    assert np.allclose(x[2], [0, np.pi, 0])
    assert np.allclose(x[3], [0, np.pi, 0.5])

    assert np.allclose(x[4], [0.5, 0, 0])
    assert np.allclose(x[5], [0.5, 0, 0.5])
    assert np.allclose(x[6], [0.5, np.pi, 0])
    assert np.allclose(x[7], [0.5, np.pi, 0.5])

    # same test for a 1D grid

    grid=UniformCylindricalGrid(1,  limits=[0,1], nptx=4)

    x=grid.get_x()

    assert np.allclose(x, [[0],[0.25],[0.5],[0.75]])

def test_dx():
    grid=UniformCylindricalGrid(3,  limits=[[0, 1], [0, 2*np.pi], [0,1]], nptx=[2, 2, 2])

    dx=grid.get_dx()

    assert np.allclose(dx, [0.5, np.pi , 0.5])

    # same test for a 1D grid

    grid=UniformCylindricalGrid(1,  limits=[0,1], nptx=4)

    dx=grid.get_dx()

    assert np.allclose(dx, [[0.25]])

def test_dV():
    grid=UniformCylindricalGrid(3,  limits=[[0, 1], [0, 2*np.pi], [0,1]], nptx=[2, 2, 2])

    dV=grid.get_dV()
    dx=grid.get_dx()

    dprod=dx[0]*dx[1]*dx[2]

    assert np.allclose(dV, [0, 0, 0, 0, dprod*0.5, dprod*0.5, dprod*0.5, dprod*0.5])

    # same test for a 1D grid

    grid=UniformCylindricalGrid(1,  limits=[0,1], nptx=4)

    dV=grid.get_dV()
    dx=grid.get_dx()

    dprod=dx[0]

    assert np.allclose(dV, [dprod*0, dprod*0.25, dprod*0.5, dprod*0.75])

def test_inGrid():

    def filter(x):  
        x_test=np.zeros(x.shape[0], dtype=bool)
        for n, ix in enumerate(x):
            x_test[n]=ix[0]>0.5 and ix[1]<np.pi 

        return x_test
    
    def filter2(x):  
        x_test=np.zeros(x.shape[0], dtype=bool)
        for n, ix in enumerate(x):
            x_test[n]=ix[0]>0.5 and ix[1]>np.pi 
        return x_test
    
    def filter_1D(x):  
        x_test=np.zeros(x.shape[0], dtype=bool)
        for n, ix in enumerate(x):
            x_test[n]=ix[0]>0.5

        return x_test
    
    def filter2_1D(x):  
        x_test=np.zeros(x.shape[0], dtype=bool)
        for n, ix in enumerate(x):
            x_test[n]=ix[0]<0.5

        return x_test
    
    grid=UniformCylindricalGrid(3,  limits=[[0, 1], [0, 2*np.pi], [0,1]], nptx=[3, 3, 3], filter=filter)

    inGrid=grid.get_inGrid()

    assert np.allclose(inGrid, [False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
        True,  True,  True,  True,  True,  True, False, False, False])

    grid=UniformCylindricalGrid(3,  limits=[[0, 1], [0, 2*np.pi], [0,1]], nptx=[3, 3, 3], filter=filter2)

    inGrid=grid.get_inGrid()

    assert np.allclose(inGrid, [False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
        False,  False,  False,  False,  False,  False, True, True, True])

    # same test for a 1D grid

    grid=UniformCylindricalGrid(1,  limits=[0,1], nptx=4, filter=filter_1D)

    inGrid=grid.get_inGrid()

    assert np.allclose(inGrid, [False, False, False, True])

    grid=UniformCylindricalGrid(1,  limits=[0,1], nptx=4, filter=filter2_1D)

    inGrid=grid.get_inGrid()

    assert np.allclose(inGrid, [True, True, False, False])



def test_toCartesian():
    grid=UniformCylindricalGrid(3,  limits=[[0, 1], [0, np.pi], [0,1]], nptx=[2, 2, 2])

    cartGrid=grid.toCartesian()

    assert np.allclose(cartGrid, [[0,0,0], [0,0,0.5],
                                  [0,0,0], [0,0,0.5 ], 
                                  [0.5,0,0], [0.5,0,0.5],
                                  [0,0.5,0], [0.0,0.5,0.5]])

    # same test for a 1D grid

    grid=UniformCylindricalGrid(1,  limits=[0,1], nptx=4)

    cartGrid=grid.toCartesian()

    assert np.allclose(cartGrid, [[0], [0.25], [0.5], [0.75]])

def test_subSet():
    grid=UniformCylindricalGrid(2,  limits=[[0, 1], [0, 2*np.pi]], nptx=[3,3])

    subSet=grid.subSet([slice(0, 2), slice(None, None, 2)])

    assert np.allclose(subSet,[[0,0],[0.0,4.1887902],[0.33333333,0.0],[0.33333333,4.1887902]])

    #same test for a 1D grid

    grid=UniformCylindricalGrid(1,  limits=[0,1], nptx=3)

    subSet=grid.subSet([slice(None, None, 2)])

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

    grid=UniformCylindricalGrid(2,  limits=[[0, 1], [0, 2*np.pi]], nptx=[3, 3], filter=filter)

    subSet=grid.subSet([slice(None, None, 2), slice(None, None, 2)])

    assert subSet.shape==(1,2)
    assert np.allclose(subSet,[[0.66666667,0.0]])

    # test for a 1D grid

    grid=UniformCylindricalGrid(1,  limits=[0, 1], nptx=3, filter=filter1d)

    subSet=grid.subSet([slice(None, None, 2)])

    assert subSet.shape==(1,1)
    assert np.allclose(subSet,[[0.66666667]])
