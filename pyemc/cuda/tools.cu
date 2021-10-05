__device__ void foo() // Dummy function used to suppress NTHREADS warning
{
  printf("%i\n", NTHREADS);
}

__global__ void kernel_blur_model(float *const model_out,
				  const float *const model_in,
				  const float sigma,
				  const int cutoff) {
   const int model_x = gridDim.x;
   const int model_y = gridDim.y;
   const int model_z = blockDim.x;

   const int x = blockIdx.x;
   const int y = blockIdx.y;
   const int z = threadIdx.x;

   float sum = 0;
   float weight = 0;
   float this_weight;

   if (model_in[x*model_y*model_z + y*model_z + z] < 0) {
     model_out[x*model_y*model_z + y*model_z + z] = -1;
     // printf("(%d, %d, %d) return\n", x, y, z);
     // return;
   } else {
     // printf("(%d, %d, %d) do full\n", x, y, z);
   
     for (int this_x = x-cutoff; this_x <= x+cutoff; this_x++) {
       if (this_x < 0 || this_x >= model_x) continue;
       for (int this_y = y-cutoff; this_y <= y+cutoff; this_y++) {
	 if (this_y < 0 || this_y >= model_y) continue;
	 for (int this_z = z-cutoff; this_z <= z+cutoff; this_z++) {
	   if (this_z < 0 || this_z >= model_z) continue;

	   if (model_in[this_x*model_y*model_z + this_y*model_z + this_z] < 0) continue;
	 
	   this_weight = exp( -(powf(this_x-x, 2) + powf(this_y-y, 2) + powf(this_z-z, 2)) / (2*powf(sigma, 2)) );
	   /* printf("(%d, %d, %d) (%d, %d, %d) weight is %g = exp(-%g / %g)\n", */
	   /* 	  x, y, z, this_x, this_y, this_z, this_weight, */
	   /* 	  (powf(this_x-x, 2) + powf(this_y-y, 2) + powf(this_z-z, 2)), */
	   /* 	  (2*powf(sigma, 2))); */
	   sum += model_in[this_x*model_y*model_z + this_y*model_z + this_z] * this_weight;
	   weight += this_weight;
	 }
       }
     }

     if (weight > 0) {
       model_out[x*model_y*model_z + y*model_z + z] = sum / weight;
       //printf("(%d, %d, %d) weight is positive (%g)\n", x, y, z, weight);
     } else {
       model_out[x*model_y*model_z + y*model_z + z] = -1;
       //printf("(%d, %d, %d) weight is zero (%g)\n", x, y, z, weight);
     }
   }

   /* printf("sigma = %f, cutoff = %d\n", sigma, cutoff); */
}
