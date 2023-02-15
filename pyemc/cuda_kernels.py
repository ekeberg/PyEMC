import os
import sys
import cupy


def import_cuda_file(file_name, kernel_names, absolute_path=False):
    threads_code = f"const int NTHREADS = {_NTHREADS};"
    cuda_files_dir = os.path.join(os.path.split(__file__)[0], "cuda")
    header_file = "header.cu"
    with open(os.path.join(cuda_files_dir, header_file), "r") as file_handle:
        header_source = file_handle.read()
    if not absolute_path:
        file_name = os.path.join(cuda_files_dir, file_name)
    with open(file_name, "r") as file_handle:
        main_source = file_handle.read()
    combined_source = "\n".join((header_source, threads_code, main_source))
    module = cupy.RawModule(code=combined_source,
                            options=("--std=c++11", ),
                            name_expressions=kernel_names)
    module.compile(log_stream=sys.stdout)
    kernels = {}
    for this_name in kernel_names:
        kernels[this_name] = module.get_function(this_name)
    return kernels



def import_kernels():
    emc_kernels = import_cuda_file(
        "emc_cuda.cu",
        ["kernel_expand_model",
         "kernel_insert_slices",
         "kernel_expand_model_2d",
         "kernel_insert_slices_2d",
         "kernel_rotate_model"])
    respons_kernels = import_cuda_file(
        "calculate_responsabilities_cuda.cu",
        ["kernel_sum_slices",
         "kernel_calculate_responsabilities_poisson",
         "kernel_calculate_responsabilities_poisson_scaling",
         "kernel_calculate_responsabilities_poisson_per_pattern_scaling",
         "kernel_calculate_responsabilities_poisson_sparse",
         "kernel_calculate_responsabilities_poisson_sparse_scaling",
         "kernel_calculate_responsabilities_poisson_sparse_per_pattern_"
         "scaling",
         "kernel_calculate_responsabilities_poisson_sparser",
         "kernel_calculate_responsabilities_poisson_sparser_scaling",
         "kernel_calculate_responsabilities_gaussian",
         "kernel_calculate_responsabilities_gaussian_scaling",
         "kernel_calculate_responsabilities_gaussian_per_pattern_scaling"])
    scaling_kernels = import_cuda_file(
        "calculate_scaling_cuda.cu",
        ["kernel_calculate_scaling_poisson",
         "kernel_calculate_scaling_poisson_sparse",
         "kernel_calculate_scaling_poisson_sparser",
         "kernel_calculate_scaling_per_pattern_poisson",
         "kernel_calculate_scaling_per_pattern_poisson_sparse"])
    slices_kernels = import_cuda_file(
        "update_slices_cuda.cu",
        ["kernel_normalize_slices",
         "kernel_update_slices<int>",
         "kernel_update_slices<float>",
         "kernel_update_slices_scaling<int>",
         "kernel_update_slices_scaling<float>",
         "kernel_update_slices_per_pattern_scaling<int>",
         "kernel_update_slices_per_pattern_scaling<float>",
         "kernel_update_slices_sparse",
         "kernel_update_slices_sparse_scaling",
         "kernel_update_slices_sparse_per_pattern_scaling",
         "kernel_update_slices_sparser",
         "kernel_update_slices_sparser_scaling"])
    tools_kernels = import_cuda_file(
        "tools.cu",
        ["kernel_blur_model"])
    kernels = {**emc_kernels,
               **respons_kernels,
               **scaling_kernels,
               **slices_kernels,
               **tools_kernels}
    return kernels


kernels = import_kernels()
