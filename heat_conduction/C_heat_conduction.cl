//-------------------------------------------------------------
//
//  Heat Conduction Kernel
//
//-------------------------------------------------------------

#define I2D(num, c, r) ((r)*(num)+(c)) // Indexing into a 1D array from 2D space

__kernel void step_kernel_mod(
					int ni, 
					int nj, 
					float fact, 
					__global float* temp_in, 
					__global float* temp_out) 
{
	int i00, im10, ip10, i0m1, i0p1;
	float d2tdx2, d2tdy2;
  
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;

	if(i < ni-1 && j < nj-1) {
		// find indices into linear memory for central point and neighbours
		i00 = I2D(ni, i, j);
		im10 = I2D(ni, i-1, j); 
		ip10 = I2D(ni, i+1, j);
		i0m1 = I2D(ni, i, j-1);
		i0p1 = I2D(ni, i, j+1);

		// evaluate derivatives
		d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
		d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

		// update temperatures
		temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
	  }
}