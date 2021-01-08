
extern "C" __global__ void kernel_calculate_scaling_poisson(const float *const patterns,
							    const float *const slices,
							    float *const scaling,
							    const int number_of_pixels) {
  const int index_pattern = blockIdx.x;
  const int index_slice = blockIdx.y;
  const int number_of_patterns = gridDim.x;

  const float *const pattern = &patterns[number_of_pixels*index_pattern];
  const float *const slice = &slices[number_of_pixels*index_slice];

  float sum_slice = 0.;
  float sum_pattern = 0.;
  for (int index = threadIdx.x; index < number_of_pixels; index += blockDim.x) {
    if (pattern[index] >= 0. && slice[index] >= 0.) {
      sum_slice += slice[index];
      sum_pattern += pattern[index];
    }
  }

  __shared__ float sum_slice_cache[NTHREADS];
  __shared__ float sum_pattern_cache[NTHREADS];  
  sum_slice_cache[threadIdx.x] = sum_slice;
  sum_pattern_cache[threadIdx.x] = sum_pattern;
  inblock_reduce(sum_slice_cache);
  inblock_reduce(sum_pattern_cache);

  if (threadIdx.x == 0) {
    if (sum_pattern_cache[0] > 0) {
      scaling[index_slice*number_of_patterns + index_pattern] = sum_slice_cache[0] / sum_pattern_cache[0];
    } else {
      scaling[index_slice*number_of_patterns + index_pattern] = 1.;
    }
  }
}


extern "C" __global__ void kernel_calculate_scaling_poisson_sparse(const int *const pattern_start_indices,
								   const int *const pattern_indices,
								   const int *const pattern_values,
								   const float *const slices,
								   float *const scaling,
								   const int number_of_pixels) {
  const int index_pattern = blockIdx.x;
  const int index_slice = blockIdx.y;
  const int number_of_patterns = gridDim.x;

  //const float *const pattern = &patterns[number_of_pixels*index_pattern];
  const float *const slice = &slices[number_of_pixels*index_slice];

  const int this_start_index = pattern_start_indices[index_pattern];
  const int this_end_index = pattern_start_indices[index_pattern+1];

  float sum_slice = 0.;
  int sum_pattern = 0;

  for (int index = this_start_index+threadIdx.x; index < this_end_index; index += blockDim.x) {
    if (slice[pattern_indices[index]]) {
      sum_pattern += pattern_values[index];
    }
  }

  for (int index = threadIdx.x; index < number_of_pixels; index += blockDim.x) {
    if (slice[index] >= 0.) {
      sum_slice += slice[index];
    }
  }

  __shared__ float sum_slice_cache[NTHREADS];
  __shared__ float sum_pattern_cache[NTHREADS];  
  sum_slice_cache[threadIdx.x] = sum_slice;
  sum_pattern_cache[threadIdx.x] = (float) sum_pattern;
  inblock_reduce(sum_slice_cache);
  inblock_reduce(sum_pattern_cache);

  if (threadIdx.x == 0 ) {
    if (sum_pattern_cache[0] > 0) {
      scaling[index_slice*number_of_patterns + index_pattern] = sum_slice_cache[0] / sum_pattern_cache[0];
    } else {
      scaling[index_slice*number_of_patterns + index_pattern] = 1.0;
    }
  }
}


extern "C" __global__ void kernel_calculate_scaling_per_pattern_poisson(const float *const patterns,
									const float *const slices,
									const float *const responsabilities,
									float *const scaling,
									const int number_of_pixels,
									const int number_of_rotations) {
  const int index_pattern = blockIdx.x;
  const int number_of_patterns = gridDim.x;

  const float *const pattern = &patterns[number_of_pixels*index_pattern];

  float sum_nominator = 0.;
  float sum_denominator = 0.;
  for (int index_slice = threadIdx.x; index_slice < number_of_rotations; index_slice += blockDim.x) {
    const float *const slice = &slices[number_of_pixels*index_slice];
    for (int index = 0; index < number_of_pixels; index++) {
      if (pattern[index] >= 0. && slice[index] >= 0.) {
	sum_nominator += responsabilities[index_slice*number_of_patterns + index_pattern] * slice[index];
	sum_denominator += responsabilities[index_slice*number_of_patterns + index_pattern] * pattern[index];
      }
    }
  }

  __shared__ float sum_nominator_cache[NTHREADS];
  __shared__ float sum_denominator_cache[NTHREADS];
  sum_nominator_cache[threadIdx.x] = sum_nominator;
  sum_denominator_cache[threadIdx.x] = sum_denominator;
  inblock_reduce(sum_nominator_cache);
  inblock_reduce(sum_denominator_cache);

  if (threadIdx.x == 0) {
    if (sum_denominator_cache[0] > 0) {
      scaling[index_pattern] = sum_nominator_cache[0] / sum_denominator_cache[0];
    } else {
      scaling[index_pattern] = 1.;
    }
  }
}


extern "C" __global__ void kernel_calculate_scaling_per_pattern_poisson_sparse(const int *const pattern_start_indices,
									       const int *const pattern_indices,
									       const float *const pattern_values,
									       const float *const slices,
									       const float *const responsabilities,
									       float *const scaling,
									       const int number_of_pixels,
									       const int number_of_rotations) {
  const int index_pattern = blockIdx.x;
  const int number_of_patterns = gridDim.x;

  const int this_start_index = pattern_start_indices[index_pattern];
  const int this_end_index = pattern_start_indices[index_pattern+1];

  float sum_nominator = 0.;
  float sum_denominator = 0.;
  for (int index_slice = threadIdx.x; index_slice < number_of_rotations; index_slice += blockDim.x) {
    const float *const slice = &slices[number_of_pixels*index_slice];
    for (int index = this_start_index; index < this_end_index; index++) {
      if (slice[pattern_indices[index]] >= 0.) {
	sum_denominator += responsabilities[index_slice*number_of_patterns + index_pattern] * pattern_values[index];
      }
    }
    for (int index = 0; index < number_of_pixels; index++) {
      if (slice[pattern_indices[index]] >= 0.) {
	sum_nominator += responsabilities[index_slice*number_of_patterns + index_pattern] * slice[index];
      }
    }
  }

  __shared__ float sum_nominator_cache[NTHREADS];
  __shared__ float sum_denominator_cache[NTHREADS];
  sum_nominator_cache[threadIdx.x] = sum_nominator;
  sum_denominator_cache[threadIdx.x] = sum_denominator;
  inblock_reduce(sum_nominator_cache);
  inblock_reduce(sum_denominator_cache);

  if (threadIdx.x == 0) {
    if (sum_denominator_cache[0] > 0) {
      scaling[index_pattern] = sum_nominator_cache[0] / sum_denominator_cache[0];
    } else {
      scaling[index_pattern] = 1.;
    }
  }
}
