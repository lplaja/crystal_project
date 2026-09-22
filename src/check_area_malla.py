import sys; sys.path.append("/home/lplaja/crystal_project/src")
import numpy as np, grid as gr

ggr = gr.UniformCartesianGrid(2, [[-0.75, 0.75], [-0.75, 0.75]], [500, 500], origin=(0.0, 0),
                              filter=gr.polygonfilter, filter_args={'nsides': 6, 'radius': 2/3})
area_malla = np.sum(ggr.dV[ggr.inGrid])
area_BZ = 1/abs(2.1304224930*(-1.23) - 1.23*2.1304224930)      # 1/|a1 x a2|, en 1/A^2
print(area_malla, area_BZ, area_malla/area_BZ)
