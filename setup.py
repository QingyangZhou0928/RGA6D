from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='RGA6D',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='rga.ext',
            sources=[
                'models/extensions/extra/cloud/cloud.cpp',
                'models/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'models/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'models/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'models/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'models/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
