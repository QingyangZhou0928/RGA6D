from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='RCCR_py_intra',
    ext_modules=[
        CppExtension(
            name='RCCR_py_intra',
            sources=[
                'intra_bindings.cpp',
                'intra_registration.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-fopenmp']
            },
            extra_link_args=['-fopenmp'],
            include_dirs=[
                '/usr/local/include',                 
                '/usr/local/include/pcl-1.12',      
                '/usr/local/include/eigen3',        
                '.',                               
            ],
            libraries=[
                'pcl_common',
                'pcl_kdtree',
                'pcl_octree',
                'pcl_search',
                'pcl_sample_consensus',
                'pcl_filters',
                'pcl_io',
                'pcl_features',
                'pcl_registration',
                'igraph',
                'blas',
                'lapack',
                'gfortran'
            ],
            library_dirs=[
                '/usr/local/lib',
                '/usr/lib/x86_64-linux-gnu'
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)