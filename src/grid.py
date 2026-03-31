import numpy as np

def polygonfilter(x, filter_args:dict):
    def cross2d(a, b): # auxiliar funtion to define the cross product of 2D vectors
        return a[0]*b[1] - a[1]*b[0]
    radius=filter_args['radius']
    nsides=filter_args['nsides']
    phi=np.arange(0, 2*np.pi, 2*np.pi/nsides)
    polygon_vertices=np.column_stack((radius*np.sin(phi), radius*np.cos(phi)))
    polygon_edges= np.array([polygon_vertices[i]-polygon_vertices[i-1] for i in range (6)])
    x_test=np.zeros(x.shape[0], dtype=bool)
    for n, ix in enumerate(x):
        test=[cross2d(polygon_edges[i],(ix-polygon_vertices[i]))<=1e-15 for i in range(6)]
        x_test[n]=np.all(test)
    return x_test

class Grid:
    def __init__(self, ndim: int, limits: list, nptx: list, origin: tuple, filter: callable, filter_args:dict={}):
        '''
        filter: function that returns a boolean array. Exmaple:
            import numpy as np
            def filter(x):
                x_test=np.zeros(x.shape[0], dtype=bool)
                for n, ix in enumerate(x):
                    x_test[n]=ix[0]>0.5 and ix[1]<0.5 
                return x_test

            x=np.array([[0.6, 0.6],[0.7,0.0]])
            print(f' {x=} ---> {filter(x)=}')
        '''
        self.ndim = ndim
        if origin is not None and len(origin)!=ndim:
            raise ValueError("origin must have the same number of elements as ndim")
        
        self.origin=origin
  
        if ndim==1:
            self.limits=[limits]
            self.nptx = [nptx]
            self.npts= nptx
        else:
            self.limits = limits
            self.nptx = nptx #number of points in each dimension
            self.npts = np.prod(nptx) #total number of points
        
        self.filter = filter
        self.filter_args=filter_args

        self.x=np.empty((self.npts,self.ndim), dtype=float) #array (npts,ndim) of all coordinates 
        self.axis=np.empty(self.ndim, dtype=float) #list of nptx arrays of axis coordinates
        self.dx = np.empty_like(self.axis)
        self.dV = np.empty_like(self.x)
        self.inGrid = np.empty((self.npts,self.ndim), dtype=bool)

    def __repr__(self):
        info=f"# \n"
        info+=f"# {self.__class__.__name__}:  id= {id(self):x} \n"
        info+=f"# \t ndim={self.ndim}  \t  origin={self.origin} \n"
        info+=f"# \t limits={self.limits} \n"
        info+=f"# \t nptx={self.nptx} \n"
        info+=f"# \t filter={self.filter}  \t  filter_args={self.filter_args}"
        return info
    
    def get_x(self):
        raise NotImplementedError
    
    def get_axis(self):
        raise NotImplementedError

    def get_dV(self):
        raise NotImplementedError

    def get_dx(self):
        raise NotImplementedError
    
    def get_inGrid(self):
        raise NotImplementedError
    
    def toCartesian(self):
        raise NotImplementedError
        
    
class UniformCartesianGrid(Grid):
    def __init__(self, ndim: int, limits: list, nptx: list, origin: tuple=None, filter:callable=None, filter_args:dict={}):
        """
        Initialize a UniformCartesianGrid object.

        Parameters:
        ndim : int
            Number of dimensions.
        limits : list
            List of tuples, each containing the lower and upper limits of a dimension.
        nptx : list
            Number of points in each dimension.
        origin : tuple, optional
            Origin of the grid, defaults to (0,)*ndim.
        filter : callable, optional
            Function that takes an array of coordinates and returns a boolean array, defaults to None.

        """
        
        if origin is None:
            origin=(0,)*ndim
        
        super().__init__(ndim, limits, nptx, origin=origin, filter=filter, filter_args=filter_args)

        self.axis=self.get_axis()
        self.cartesian=True
        self.x=self.get_x()
        self.dx=self.get_dx()
        self.dV=self.get_dV()
        self.V=np.sum(self.dV)
        self.inGrid=self.get_inGrid()


        

    def get_axis(self):
        axis = []
        for i in range(self.ndim):
            axis.append(np.linspace(self.limits[i][0]+self.origin[i], self.limits[i][1]+self.origin[i], self.nptx[i], endpoint=False))
        return axis
    
    def get_x(self):
        X=np.meshgrid(*self.axis, indexing="ij")
        x=np.zeros((self.npts,self.ndim), dtype=np.float64)
        for i in range(self.ndim):
            x[:,i]=X[i].ravel(order='C')  
        return x
     
    def get_dx(self):
        dx = [self.axis[idim][1]-self.axis[idim][0] for idim in range(self.ndim)]
        return dx
    
    def get_dV(self):
        dV=np.array([np.prod(self.dx)])
        return np.broadcast_to(dV, self.x.shape[0]) # In this way the returned array has minimum memory (only memory space for one element)
                                                 # but one has to be aware that broadcasted arrays are not mutable (only read-only)
    
    def get_inGrid(self, x:np.ndarray=None, filter:callable=None, filter_args:dict={}):
        if x is None:
            x=self.x
        if filter is None:
            filter=self.filter
        if filter_args == {}:
            filter_args=self.filter_args

        if filter is None:
            inGrid=np.ones(x.shape[0],dtype=bool)
        else:
            if filter_args == {}:
                inGrid=filter(x)
            else:
                inGrid=filter(x, filter_args)
        return inGrid

    def toCartesian(self):
        return self.x
    
    def subSet(self, slices: list[slice] | slice = None,filter:callable=None, filter_args:dict={} ):
        """
        Returns a subset of the grid's points based on the given slices and filter.

        Parameters:
        slices : list[slice] | slice, optional
            List of slices or a single slice to select a subset of points.
            If None, the entire grid is selected.
        filter : callable, optional
            Function that takes an array of coordinates and returns a boolean array, defaults to None.

        Returns:
        x_subset : numpy.ndarray
            Subset of the grid's points.
        """
        if slices is None:
            x_subset= self.x[self.inGrid]
            if filter is not None:
                x_subset=x_subset[filter(x_subset, filter_args)]
            return x_subset
        
        if type(slices) is slice:
            slices=[slices]
        elif type(slices) is not list:
            raise ValueError("slices must be a slice or a list of slices")
        
        xArray=self.x.reshape(*self.nptx,self.ndim)
        x_subset=xArray[*slices].reshape(-1,self.ndim)

        inGrid=self.get_inGrid(x_subset,filter=filter, filter_args=filter_args)
        print(f'{inGrid=}')

        x_subset=x_subset[inGrid]

        return x_subset
            
class UniformCylindricalGrid(Grid):
    def __init__(self, ndim: int, limits: list, nptx: list, origin: tuple=None, offset=None, cartesian=False, filter:callable=None,
                 filter_args:dict={}):
        # coordinate x[1] refers to rho, x[2] to phi and x[>=3] cartesian

        super().__init__(ndim, limits, nptx, origin=origin, filter=filter, filter_args=filter_args)

        # check limits and origin to be compatible with cylindrical coordinates
        if origin is None:
            self.origin=(0,)*ndim
        elif origin[0]<0:
                raise ValueError("origin[0] must be >=0")
        elif origin[0]+limits[0][0]<0:
                raise ValueError("lower rho limit (origin[0]+limits[0][0]) must be >=0")
        elif origin[0]+limits[0][1]<0:
                raise ValueError("upper rho limit (origin[0]+limits[0][1]) must be >=0")
        
        if offset is not None and cartesian is False:
            raise ValueError("offset must be None if cartesian is False")
        
        self.offset=offset
        
        if ndim>1:
            if self.limits[1][1]-self.limits[1][0]>2*np.pi:
                raise ValueError("phi limits expand over an intervale greater than 2pi")
             
        self.axis=self.get_axis()
        self.cartesian=cartesian
        self.x=self.get_x()

        self.dx=self.get_dx()
        self.dV=self.get_dV()
        self.V=np.sum(self.dV)

        self.inGrid=self.get_inGrid()

        if cartesian:
            self.x=self.toCartesian()
        

    def get_axis(self):
        axis = []
        for i in range(self.ndim):
            axis.append(np.linspace(self.limits[i][0]+self.origin[i], self.limits[i][1]+self.origin[i], self.nptx[i], endpoint=False))
        return axis
    
    def get_x(self):
        X=np.meshgrid(*self.axis, indexing="ij")
        x=np.zeros((self.npts,self.ndim),dtype=np.float64)
        for i in range(self.ndim):
            x[:,i]=X[i].ravel(order='C')  
        return x
     
    def get_dx(self):
        dx=[]
        for idim in range(self.ndim):
            dx.append(self.axis[idim][1]-self.axis[idim][0])
        return dx
    
    def get_dV(self):
        dV=np.prod(self.dx)*self.x[:,0]
        return dV
    
    def get_inGrid(self, x:np.ndarray=None, filter:callable=None, filter_args:dict={}):
        if x is None:
            x=self.x
        if filter is None:
            filter=self.filter
        if filter_args == {}:
            filter_args=self.filter_args

        if filter is None:
            inGrid=np.ones(x.shape[0],dtype=bool)
        else:
            if filter_args == {}:
                inGrid=filter(x)
            else:
                inGrid=filter(x, filter_args)
        return inGrid

    def toCartesian(self):
        if self.ndim==1:
            return self.x
        cart_x0=self.x[:,0]*np.cos(self.x[:,1]) # x
        cart_x1=self.x[:,0]*np.sin(self.x[:,1]) # y
        cart_x=self.x.copy()
        cart_x[:,0]=cart_x0
        cart_x[:,1]=cart_x1
        if self.offset is not None:
            cart_x+=self.offset
        return cart_x
    
    def subSet(self, slices: list[slice] | slice = None,filter:callable=None, filter_args:dict={}):
        """
        Returns a subset of the grid's points based on the given slices and filter.

        Parameters:
        slices : list[slice] | slice, optional
            List of slices or a single slice to select a subset of points.
            If None, the entire grid is selected.
        filter : callable, optional
            Function that takes an array of coordinates and returns a boolean array, defaults to None.

        Returns:
        x_subset : numpy.ndarray
            Subset of the grid's points.
        """
        if slices is None:
            x_subset= self.x[self.inGrid]
            if filter is not None:
                x_subset=x_subset[filter(x_subset, filter_args)]
            return x_subset
        
        if type(slices) is slice:
            slices=[slices]
        elif type(slices) is not list:
            raise ValueError("slices must be a slice or a list of slices")
        
        xArray=self.x.reshape(*self.nptx,self.ndim)
        x_subset=xArray[*slices].reshape(-1,self.ndim)

        inGrid=self.get_inGrid(x_subset,filter=filter, filter_args=filter_args)

        x_subset=x_subset[inGrid]

        return x_subset
     
class FonGrid:
    def __init__(self, fun: callable, grid: Grid, restricted: bool = False, **parameters):
        self.grid=grid
        self.fun=fun
        self.restricted=restricted
        self.parameters=parameters

        if restricted:
            self.val=self.fun(self.grid.x[self.grid.inGrid], **self.parameters)
        else:
            self.val=self.fun(self.grid.x, **self.parameters)


