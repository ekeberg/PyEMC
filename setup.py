from setuptools import setup


scripts = ["scripts/emc_prepare_starting_model.py",
           "scripts/emc_make_sparse.py",
           "scripts/emc_assemble.py"]


setup(name="pyemc",
      version="0.1",
      author="Tomas Ekeberg",
      packages=["pyemc"],
      package_data={"pyemc": ["cuda/header.cu",
                              "cuda/calculate_responsabilities_cuda.cu",
                              "cuda/calculate_scaling_cuda.cu",
                              "cuda/emc_cuda.cu",
                              "cuda/update_slices_cuda.cu",
                              "cuda/tools.cu"]},
      include_package_data=True,
      scripts=scripts)
# import sys
# from setuptools import setup, Extension
# from setuptools.command.build_ext import build_ext

# import pybind11


# scripts = ["scripts/emc_prepare_starting_model.py",
#            "scripts/emc_make_sparse.py",
#            "scripts/emc_assemble.py"]

# def cpp_flag(compiler):
#     if has_flag(compiler, "-std=c++14"):
#         return "-std=c++14"
#     elif has_flag(compiler, "-std=c++11"):
#         return "-std=c++11"
#     else:
#         raise RuntimeError("Unsupported compiler: at least C++11 support is needed")

# class BuildExt(build_ext):

#     c_opts = {
#         "msvc": ["/EHsc"],
#         "unix": []
#     }
    
#     if sys.platform == "darwin":
#         c_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]

#     def build_extensions(self):
#         ct = self.compiler.compiler_type
#         opts = self.c_opts.get(ct, [])
#         if ct == "unix":
#             opts.append(f'-DVERSION_INFO="{self.distribution.get_version()}"')
#             opts.append(cpp_flag(self.compiler))
                        

# ext_modules = [
#     Extension("calculate_responsabilities",
#               ["pyemc/cpu/calculate_responsabilities_cpu.cpp"],
#               include_dirs=[pybind11.get_include(False),
#                             pybind11.get_include(True)],
#               language="c++")]



# setup(name="pyemc",
#       version="0.1",
#       author="Tomas Ekeberg",
#       packages=["pyemc"],
#       package_data={"pyemc": ["cuda/header.cu",
#                               "cuda/calculate_responsabilities_cuda.cu",
#                               "cuda/calculate_scaling_cuda.cu",
#                               "cuda/emc_cuda.cu",
#                               "cuda/update_slices_cuda.cu",
#                               "cuda/tools.cu"]},
#       ext_modules=ext_modules,
#       cmdclass={"build_ext", BuildExt},
#       include_package_data=True,
#       scripts=scripts)
