
extern "C" __global__ void kernel_calculate_responsabilities_poisson(const int* const patterns,
								     const float *const slices,
								     const int number_of_pixels,
								     float *const responsabilities,
								     const float *const log_factorial_table)
{
  __shared__ float sum_cache[NTHREADS];
  
  const int index_pattern = blockIdx.x;
  const int index_slice = blockIdx.y;
  const int number_of_patterns = gridDim.x;
  
  const int *const pattern = &patterns[number_of_pixels*index_pattern];
  const float *const slice = &slices[number_of_pixels*index_slice];
  
  float sum = 0.;
  for (int index = threadIdx.x; index < number_of_pixels; index += blockDim.x) {
    if (pattern[index] >= 0 && slice[index] > 0.) {
      /* sum += ((-slice[index]) + */
      /* 	      ((int) pattern[index]) * logf(slice[index]) - */
      /* 	      log_factorial_table[(int) pattern[index]]); */
      /* sum += ((-slice[index]) + */
      /* 	      ((int) pattern[index]) * logf(slice[index]) - */
      /* 	      log_factorial_table[(int) pattern[index]]); */
      sum += ((-slice[index]) +
      	      pattern[index] * logf(slice[index]) -
      	      log_factorial_table[pattern[index]]);
    }
  }
  sum_cache[threadIdx.x] = sum;

  inblock_reduce(sum_cache);
  if (threadIdx.x == 0) {
    responsabilities[index_slice*number_of_patterns + index_pattern] = sum_cache[0];
  }
}


extern "C" __global__ void kernel_calculate_responsabilities_poisson_scaling(const int *const patterns,
									     const float *const slices,
									     const int number_of_pixels,
									     const float *const scalings,
									     float *const responsabilities,
									     const float *const log_factorial_table)
{
  __shared__ float sum_cache[NTHREADS];

  const int index_pattern = blockIdx.x;
  const int index_slice = blockIdx.y;
  const int number_of_patterns = gridDim.x;
  
  const int *const pattern = &patterns[number_of_pixels*index_pattern];
  const float *const slice = &slices[number_of_pixels*index_slice];
  const float scaling = scalings[index_slice*number_of_patterns + index_pattern];
  
  float sum = 0.;
  for (int index = threadIdx.x; index < number_of_pixels; index += blockDim.x) {
    if (pattern[index] >= 0. && slice[index] > 0.) {
      sum += ((-slice[index]/scaling) +
	      pattern[index] * logf(slice[index]/scaling) -
	      log_factorial_table[pattern[index]]);
    }
  }
  sum_cache[threadIdx.x] = sum;

  inblock_reduce(sum_cache);
  if (threadIdx.x == 0) {
    responsabilities[index_slice*number_of_patterns + index_pattern] = sum_cache[0];
  }
}

extern "C" __global__ void kernel_sum_slices(const float *const slices,
				  const int number_of_pixels,
				  float *const slice_sums)
{
  __shared__ float sum_cache[NTHREADS];
  
  const int index_slice = blockIdx.x;
  const float *const slice = &slices[number_of_pixels*index_slice];

  float sum = 0.;
  for (int index_pixel = threadIdx.x; index_pixel < number_of_pixels; index_pixel += blockDim.x) {
    if (slice[index_pixel] > 0.) {
      sum += slice[index_pixel];
    }
  }
  sum_cache[threadIdx.x] = sum;
  inblock_reduce(sum_cache);
  if (threadIdx.x == 0) {
    slice_sums[index_slice] = sum_cache[0];
  }
}


extern "C"__global__ void kernel_calculate_responsabilities_poisson_per_pattern_scaling(const int *const patterns,
											const float *const slices,
											const int number_of_pixels,
											const float *const scalings,
											float *const responsabilities,
											const float *const log_factorial_table)
{
  __shared__ float sum_cache[NTHREADS];

  const int index_pattern = blockIdx.x;
  const int index_slice = blockIdx.y;
  const int number_of_patterns = gridDim.x;
  
  const int *const pattern = &patterns[number_of_pixels*index_pattern];
  const float *const slice = &slices[number_of_pixels*index_slice];
  const float scaling = scalings[index_pattern];
  
  float sum = 0.;
  for (int index = threadIdx.x; index < number_of_pixels; index += blockDim.x) {
    if (pattern[index] >= 0. && slice[index] > 0.) {
      sum += ((-slice[index]/scaling) +
	      pattern[index] * logf(slice[index]/scaling) -
	      log_factorial_table[pattern[index]]);
    }
  }
  sum_cache[threadIdx.x] = sum;

  inblock_reduce(sum_cache);
  if (threadIdx.x == 0) {
    responsabilities[index_slice*number_of_patterns + index_pattern] = sum_cache[0];
  }
}

extern "C" __global__ void kernel_calculate_responsabilities_sparse(const int *const pattern_start_indices,
								    const int *const pattern_indices,
								    const int *const pattern_values,
								    const float *const slices,
								    const int number_of_pixels,
								    float *const responsabilities,
								    const float *const slice_sums,
								    const float *const log_factorial_table)
{
  __shared__ float sum_cache[NTHREADS];

  const int number_of_patterns = gridDim.x;
  const int index_pattern = blockIdx.x;
  const int index_slice = blockIdx.y;
  const float *const slice = &slices[number_of_pixels*index_slice];
  
  int index_pixel;
  float sum = 0.;
  for (int index = pattern_start_indices[index_pattern]+threadIdx.x;
       index < pattern_start_indices[index_pattern+1];
       index += blockDim.x) {
    index_pixel = pattern_indices[index];
    if (slice[index_pixel] > 0.) {
      sum += (pattern_values[index] *
      	      logf(slice[index_pixel]) -
      	      log_factorial_table[pattern_values[index]]);
    }
  }
  sum_cache[threadIdx.x] = sum;
  inblock_reduce(sum_cache);
  if (threadIdx.x == 0) {
    responsabilities[index_slice*number_of_patterns + index_pattern] = -slice_sums[index_slice] + sum_cache[0];
  }
}


extern "C" __global__ void kernel_calculate_responsabilities_sparse_scaling(const int *const pattern_start_indices,
									    const int *const pattern_indices,
									    const int *const pattern_values,
									    const float *const slices,
									    const int number_of_pixels,
									    const float *const scaling,
									    float *const responsabilities,
									    const float *const slice_sums,
									    const float *const log_factorial_table)
{
  __shared__ float sum_cache[NTHREADS];

  const int number_of_patterns = gridDim.x;
  const int index_pattern = blockIdx.x;
  const int index_slice = blockIdx.y;
  const float *const slice = &slices[number_of_pixels*index_slice];
  const float this_scaling = scaling[index_slice*number_of_patterns + index_pattern];

  
  int index_pixel;
  float sum = 0.;
  int this_pattern_start = pattern_start_indices[index_pattern];
  int this_pattern_end = pattern_start_indices[index_pattern+1];
  for (int index = this_pattern_start+threadIdx.x;
       index < this_pattern_end;
       index += blockDim.x) {
    index_pixel = pattern_indices[index];
    if (slice[index_pixel] > 0.) {
      sum += pattern_values[index] * logf(slice[index_pixel]/this_scaling) - log_factorial_table[pattern_values[index]];
    }
  }
  sum_cache[threadIdx.x] = sum;
  inblock_reduce(sum_cache);
  if (threadIdx.x == 0) {
    responsabilities[index_slice*number_of_patterns + index_pattern] = -slice_sums[index_slice]/this_scaling + sum_cache[0];
  }
}


extern "C" __global__ void kernel_calculate_responsabilities_sparse_per_pattern_scaling(const int *const pattern_start_indices,
											const int *const pattern_indices,
											const int *const pattern_values,
											const float *const slices,
											const int number_of_pixels,
											const float *const scaling,
											float *const responsabilities,
											const float *const slice_sums,
											const float *const log_factorial_table)
{
  __shared__ float sum_cache[NTHREADS];

  const int number_of_patterns = gridDim.x;
  const int index_pattern = blockIdx.x;
  const int index_slice = blockIdx.y;
  const float *const slice = &slices[number_of_pixels*index_slice];
  const float this_scaling = scaling[index_pattern];

  
  int index_pixel;
  float sum = 0.;
  for (int index = pattern_start_indices[index_pattern]+threadIdx.x; index < pattern_start_indices[index_pattern+1]; index += blockDim.x) {
    index_pixel = pattern_indices[index];
    if (slice[index_pixel] > 0.) {
      sum += pattern_values[index] * logf(slice[index_pixel]/this_scaling) - log_factorial_table[pattern_values[index]];
    }
  }
  sum_cache[threadIdx.x] = sum;
  inblock_reduce(sum_cache);
  if (threadIdx.x == 0) {
    responsabilities[index_slice*number_of_patterns + index_pattern] = -slice_sums[index_slice]/this_scaling + sum_cache[0];
  }
}


extern "C" __global__ void kernel_calculate_responsabilities_sparser_scaling(const int *const pattern_start_indices,
									     const int *const pattern_indices,
									     const int *const pattern_values,
									     const int *const pattern_ones_start_indices,
									     const int *const pattern_ones_indices,
									     const float *const slices,
									     const int number_of_pixels,
									     const float *const scaling,
									     float *const responsabilities,
									     const float *const slice_sums,
									     const float *const log_factorial_table)
{
  __shared__ float sum_cache[NTHREADS];

  const int number_of_patterns = gridDim.x;
  const int index_pattern = blockIdx.x;
  const int index_slice = blockIdx.y;
  const float *const slice = &slices[number_of_pixels*index_slice];
  const float this_scaling = scaling[index_slice*number_of_patterns + index_pattern];

  
  int index_pixel;
  float sum = 0.;
  int this_pattern_start = pattern_start_indices[index_pattern];
  int this_pattern_end = pattern_start_indices[index_pattern+1];
  for (int index = this_pattern_start+threadIdx.x;
       index < this_pattern_end;
       index += blockDim.x) {
    index_pixel = pattern_indices[index];
    if (slice[index_pixel] > 0.) {
      sum += pattern_values[index] * logf(slice[index_pixel]/this_scaling) - log_factorial_table[pattern_values[index]];
    }
  }

  int this_pattern_ones_start = pattern_ones_start_indices[index_pattern];
  int this_pattern_ones_end = pattern_ones_start_indices[index_pattern+1];
  for (int index = this_pattern_ones_start+threadIdx.x;
       index < this_pattern_ones_end;
       index += blockDim.x) {
    index_pixel = pattern_ones_indices[index];
    if (slice[index_pixel] > 0.) {
      sum += logf(slice[index_pixel]/this_scaling);
    }
  }
  
  sum_cache[threadIdx.x] = sum;
  inblock_reduce(sum_cache);
  if (threadIdx.x == 0) {
    responsabilities[index_slice*number_of_patterns + index_pattern] = -slice_sums[index_slice]/this_scaling + sum_cache[0];
  }
}



/* __global__ void kernel_calculate_surprise_estimate_and_variance(const float *const slices, */
/* 								float * const surprise_expectation, */
/* 								float * const surprise_variance, */
/* 								const int number_of_pixels, */
/* 								const int k_max) { */
/*   const int index_slice = blockIdx.x; */
/*   const float *const slice = &slices[number_of_pixels*index_slice]; */
/*   float * const this_surprise_expectation = &surprise_expectation[number_of_pixels*index_slice]; */
/*   float * const this_surprise_variance = &surprise_variance[number_of_pixels*index_slice]; */

/*   for (int index = threadIdx.x; index < number_of_pixels; index += blockDim.x) { */
/*     if (slice[index] > 0.) { */
/*       float log_factorial = 0; */
/*       this_surprise_expectation[index] = 0; */
/*       this_surprise_variance[index] = 0; */
/*       for (int k = 0; k < k_max; k++) { */
/* 	float log_factor = k*logf(slice[index]) - slice[index] - log_factorial; */

/* 	this_surprise_expectation[index] -= (expf(log_factor) * */
/* 					     (k*logf(slice[index])-slice[index]- */
/* 					      log_factorial)); */
/* 	this_surprise_variance[index] += (expf(log_factor) * */
/* 					  pow(k*logf(slice[index])-slice[index]- */
/* 					      log_factorial, 2)); */

/* 	/\* if (index == 128*30+64 && index_slice == 0) { *\/ */
/* 	/\*   printf("%f\n", this_surprise_expectation[index]); *\/ */
/* 	/\*   printf("%f\n", this_surprise_variance[index]); *\/ */
/* 	/\*   printf("%f, %i, %f, %f, %f\n", expf(log_factor), k, logf(slice[index]), slice[index], log_factorial); *\/ */
/* 	/\*   printf("\n"); *\/ */
/* 	/\* } *\/ */
/* 	log_factorial += logf(k+1); */
/*       } */
/*       this_surprise_expectation[index] -= pow(this_surprise_variance[index], 2); */
/*       /\* if (index == 128*30+64 && index_slice == 0) { *\/ */
/*       /\* 	printf("%f\n", this_surprise_expectation[index]); *\/ */
/*       /\* } *\/ */
/*     } */
/*   } */
/* } */
								

/* __global__ void kernel_calculate_surprise(const float* const patterns, */
/* 					  const float *const slices, */
/* 					  const int number_of_pixels, */
/* 					  float *const responsabilities, */
/* 					  const float *const log_factorial_table, */
/* 					  const float *const surprise_expectation, */
/* 					  const float *const surprise_variance) */
/* {  */
/*   __shared__ float sum_surprise_cache[NTHREADS]; */
/*   __shared__ float sum_surprise_expectation_cache[NTHREADS]; */
/*   __shared__ float sum_surprise_variance_cache[NTHREADS]; */

/*   const int index_pattern = blockIdx.x; */
/*   const int index_slice = blockIdx.y; */
/*   const int number_of_patterns = gridDim.x; */
  
/*   const float *const pattern = &patterns[number_of_pixels*index_pattern]; */
/*   const float *const slice = &slices[number_of_pixels*index_slice]; */

/*   const float * const this_surprise_expectation = &surprise_expectation[number_of_pixels*index_slice]; */
/*   const float * const this_surprise_variance = &surprise_variance[number_of_pixels*index_slice]; */
  
/*   float sum_surprise = 0.; */
/*   float sum_surprise_expectation = 0.; */
/*   float sum_surprise_variance = 0.; */
/*   for (int index = threadIdx.x; index < number_of_pixels; index += blockDim.x) { */
/*     if (pattern[index] >= 0. && slice[index] > 0.) { */
/*       sum_surprise -= -(pattern[index]*logf(slice[index]) - slice[index] - log_factorial_table[(int) pattern[index]]); */
/*       sum_surprise_expectation += this_surprise_expectation[index]; */
/*       sum_surprise_variance += this_surprise_variance[index]; */
/*     } */
/*   } */
/*   sum_surprise_cache[threadIdx.x] = sum_surprise; */
/*   sum_surprise_expectation_cache[threadIdx.x] = sum_surprise_expectation; */
/*   sum_surprise_variance_cache[threadIdx.x] = sum_surprise_variance; */

/*   inblock_reduce(sum_surprise_cache); */
/*   inblock_reduce(sum_surprise_expectation_cache); */
/*   inblock_reduce(sum_surprise_variance_cache); */
/*   if (threadIdx.x == 0) { */
/*     responsabilities[index_slice*number_of_patterns + index_pattern] = */
/*       (sum_surprise_cache[0] - sum_surprise_expectation_cache[0]) / sum_surprise_variance_cache[0]; */
/*   } */
/* } */


/* __global__ void kernel_calculate_surprise_single(const float* const patterns, */
/* 						 const float *const slices, */
/* 						 const int number_of_pixels, */
/* 						 float *const surprise, */
/* 						 const float *const log_factorial_table, */
/* 						 const int k_max) */
/* { */
/*   __shared__ float sum_surprise_cache[NTHREADS]; */
/*   __shared__ float sum_surprise_expectation_cache[NTHREADS]; */
/*   __shared__ float sum_surprise_variance_cache[NTHREADS]; */

/*   const int index_pattern = blockIdx.x; */
  
/*   const float *const pattern = &patterns[number_of_pixels*index_pattern]; */
/*   const float *const slice = &slices[number_of_pixels*index_pattern]; */
  
/*   float sum_surprise = 0.; */
/*   float sum_surprise_expectation = 0.; */
/*   float sum_surprise_variance = 0.; */
/*   for (int index = threadIdx.x; index < number_of_pixels; index += blockDim.x) { */
/*     if (pattern[index] >= 0. && slice[index] > 0.) { */
/*       float surprise; */
/*       float surprise_expectation = 0; */
/*       float surprise_variance = 0; */
/*       float log_factorial = 0; */
/*       for (int k = 0; k < k_max; k++) { */
/* 	float log_factor = k*logf(slice[index]) - slice[index] - log_factorial; */
/* 	surprise_expectation -= (expf(log_factor) * */
/* 				 (k*logf(slice[index])-slice[index]- */
/* 				  log_factorial)); */
/* 	surprise_variance += (expf(log_factor) * */
/* 			      pow(k*logf(slice[index])-slice[index]- */
/* 				  log_factorial, 2)); */

/* 	log_factorial += logf(k+1); */
	
/*       } */
/*       surprise_variance -= pow(surprise_expectation, 2); */
/*       surprise = -(pattern[index]*logf(slice[index]) - slice[index] - log_factorial_table[(int) pattern[index]]); */

/*       sum_surprise += surprise; */
/*       sum_surprise_expectation += surprise_expectation; */
/*       sum_surprise_variance += surprise_variance; */
/*     } */
/*   } */
/*   sum_surprise_cache[threadIdx.x] = sum_surprise; */
/*   sum_surprise_expectation_cache[threadIdx.x] = sum_surprise_expectation; */
/*   sum_surprise_variance_cache[threadIdx.x] = sum_surprise_variance; */

/*   inblock_reduce(sum_surprise_cache); */
/*   inblock_reduce(sum_surprise_expectation_cache); */
/*   inblock_reduce(sum_surprise_variance_cache); */
/*   if (threadIdx.x == 0) { */
/*     surprise[index_pattern] = */
/*       (sum_surprise_cache[0] - sum_surprise_expectation_cache[0]) / sum_surprise_variance_cache[0]; */
/*     if (index_pattern == 0) { */
/*       printf("%f = (%f - %f) / %f\n", surprise[index_pattern], sum_surprise_cache[0], */
/* 	     sum_surprise_expectation_cache[0], sum_surprise_variance_cache[0]); */
/*     } */
/*   } */
/* } */
