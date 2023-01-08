//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix library include file (function prototypes)
//
//  HISTORY: Written by Tim Mattson, August 2010 
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//           Ported by Tom Deakin, July 2013
//			 Modified by me, January 2023
//
//------------------------------------------------------------------------------

#ifndef __MATRIX_LIB_HDR
#define __MATRIX_LIB_HDR

//------------------------------------------------------------------------------
//
//	Referential function for calculating heat transfer to be run on the CPU
//
//------------------------------------------------------------------------------
void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out);

//------------------------------------------------------------------------------
//
//  Function to initialize matrices with random data
//
//------------------------------------------------------------------------------
void initmat(int size, float *temp1_ref, float *temp2_ref);

//------------------------------------------------------------------------------
//
//  Function to analyze and output results 
//
//------------------------------------------------------------------------------
void results(int ni, int nj, float *temp1, float *temp1_ref);
    
#endif
