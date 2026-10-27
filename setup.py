# Author: Romain E. Lacoste
# License: BSD-3-Clause

import os
import subprocess

import numpy
from setuptools import Extension, find_packages, setup


def ensure_init_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__init__.py' not in filenames:
            init_file_path = os.path.join(dirpath, '__init__.py')
            with open(init_file_path, 'w') as f:
                f.write("# init file for package\n")

def run_swig(interface_file, output_file, include_dirs):
    swig_cmd = [
        'swig',
        '-python',  # Generate Python bindings
        '-c++',     # The code is C++
        *[f'-I{d}' for d in include_dirs],  # Include directories
        '-o', output_file,  # Output file for SWIG-generated C++ wrapper code
        interface_file  # Input SWIG interface file
    ]

    try:
        subprocess.check_call(swig_cmd)
    except subprocess.CalledProcessError as e:
        print(f"SWIG command failed for {interface_file} with error:", e)
        raise

def create_extension(module_name, module_dir, source_files):
    # Define directory paths
    interface_dir = os.path.normpath(os.path.join('lib', 'interface', 'sparklen', module_dir))
    src_dir = os.path.normpath(os.path.join('lib', 'src', module_dir))
    build_dir = os.path.normpath(os.path.join('sparklen', module_dir, 'build'))

    # Define file paths
    interface_file = os.path.normpath(os.path.join(interface_dir, f'{module_name}_module.i'))
    wrapper_file = os.path.normpath(os.path.join(build_dir, f'{module_name}_wrap.cxx'))

    # Ensure build directory exists
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    # Define include directories
    include_dirs = [
        os.path.normpath(os.path.join('lib', 'include')),
        os.path.normpath(os.path.join('lib', 'include', 'sparklen')),
        os.path.normpath(os.path.join('lib', 'interface', 'sparklen')),
        os.path.normpath(os.path.join('lib', 'interface'))
    ]

    # Run SWIG to generate wrapper code
    run_swig(interface_file, wrapper_file, include_dirs)

    # Determine compile arguments based on platform
    extra_compile_args = ['-std=c++11'] if os.name != 'nt' else ['/std:c++11']

    # Define source files for the extension
    sources = [wrapper_file] + [os.path.normpath(os.path.join(src_dir, src)) for src in source_files]

    # Return the Extension object
    return Extension(
        name=f"sparklen.{module_dir.replace('/', '.')}.build._{module_name}",
        sources=sources,
        include_dirs=include_dirs + [numpy.get_include()],
        extra_compile_args=extra_compile_args,
        language='c++'
    )

# Create extensions for module1 and module2 with multiple source files
array_extension = create_extension(
    module_name='array',
    module_dir='array',
    source_files=[
    ]
)

hawkes_model_extension = create_extension(
    module_name='hawkes_model',
    module_dir='hawkes/model',
    source_files=[
        'model_hawkes_exp_least_squares_single.cpp',
        'model_hawkes_exp_least_squares.cpp',
        'model_hawkes_exp_log_likelihood_single.cpp',
        'model_hawkes_exp_log_likelihood.cpp'
    ]
)

prox_extention = create_extension(
    module_name='prox',
    module_dir='prox',
    source_files=[
        'prox_zero.cpp',
        'prox_l1.cpp',
        'prox_l2.cpp',
        'prox_elastic_net.cpp'
    ]
)

ext_sparklen_modules = [
    array_extension,
    hawkes_model_extension,
    prox_extention
]

setup(
    ext_modules=ext_sparklen_modules,
    packages=find_packages(),
    include_package_data=True,
  )

ensure_init_files('sparklen')
