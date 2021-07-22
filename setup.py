from setuptools import setup, find_packages

setup(
    name="holy_chaos",
    # packages = ["holy_chaos"],
    packages=find_packages(exclude=["docs", "tests*"]),
    install_requires=[ #minimal set of requireement necessaru for the package to run
        "bokeh",
        "ipdb",
        "dask[complete]",
        "dask-ms[xarray]",
        "datashader",
        "future"
        "matplotlib",
        "numpy >= 1.15",
        "python-casacore",
        "xarray",
        "numcodecs", #for dask experimental zarr
        "zarr" # for dask expermental zarr
    ]
    # package_data={"sample": ["package_data.dat"]} 
    # #additionaldata necessary for the package to run. Different from manifest,
    # used for licence or readme which must be included but are not necesarily code
    #scripts = specify a script that is executable
    # entry_points={"console_scripts": name:python_module:function_torun} 
    # Define a script available in the path and 
    # maps it to a module  of a method

)
