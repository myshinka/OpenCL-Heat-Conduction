//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix library for the multiplication driver
//
//  PURPOSE: This is a simple set of functions to manipulate
//           matrices used with the multiplcation driver.
//
//  USAGE:   The matrices are square and the order is
//           set as a defined constant, ORDER.
//
//  HISTORY: Written by Tim Mattson, August 2010
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//           Ported to C by Tom Deakin, 2013
//			 Modified by me, 2023
//
//------------------------------------------------------------------------------

#include "heat_sim.h"

//------------------------------------------------------------------------------
//
//	Referential function for calculating heat transfer to be run on the CPU
//
//------------------------------------------------------------------------------

void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;


  // loop over all points in domain (except boundary)
  for ( int j=1; j < nj-1; j++ ) {
    for ( int i=1; i < ni-1; i++ ) {
      // find indices into linear memory
      // for central point and neighbours
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
}

//------------------------------------------------------------------------------
//
//  Function to initialize matrices with random data
//
//------------------------------------------------------------------------------
void initmat(int size, float *temp1, float *temp2)
{
	for( int i = 0; i < size; ++i) {
		temp1[i] = (float)rand()/(float)(RAND_MAX/100.0f);
		temp2[i] = 0;
  }
}

//------------------------------------------------------------------------------
//
//  Function to analyze and output results
//
//------------------------------------------------------------------------------
void results(int ni, int nj, float *temp, float *temp_ref)
{

	float maxError = 0;

	for( int i = 0; i < ni*nj; ++i ) {
		if (abs(temp[i]-temp_ref[i]) > maxError) { maxError = abs(temp[i]-temp_ref[i]); }
	}

	// Check and see if our maxError is greater than an error bound
	if (maxError > TOL)
		printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
	else
		printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);
}