from setuptools import setup

scripts = ["scripts/emc_prepare_starting_model.py",
           "scripts/emc_make_sparse.py"]

setup(name="pyemc",
      version="0.1",
      author="Tomas Ekeberg",
      packages=["pyemc"],
      package_data={"pyemc": ["cuda/header.cu",
                              "cuda/calculate_responsabilities_cuda.cu",
                              "cuda/calculate_scaling_cuda.cu",
                              "cuda/emc_cuda.cu",
                              "cuda/update_slices_cuda.cu"]},
      include_package_data=True,
      scripts=scripts)



