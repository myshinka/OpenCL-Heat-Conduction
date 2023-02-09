//------------------------------------------------------------------------------
//
//  Include fle for the Matrix Multiply test harness
//
//  HISTORY: Written by Tim Mattson, August 2010
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//           Ported to C by Tom Deakin, July 2013
//			 Modified by me, January 2023
//
//------------------------------------------------------------------------------

#ifndef __MULT_HDR
#define __MULT_HDR

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "matrix_lib.h"

//------------------------------------------------------------------------------
//  functions from ../C_Common
//------------------------------------------------------------------------------
extern int    output_device_info(cl_device_id );
extern double wtime();   // returns time since some fixed past point (wtime.c)

//------------------------------------------------------------------------------
//  Constants
//------------------------------------------------------------------------------
#define WIDTH    320      // Order of the square matrices
#define HEIGHT	 320
#define TOL      (0.0005) // tolerance used in floating point comparisons
#define DIM      2        // Max dim for NDRange
#define COUNT    30       // number of times to do each multiplication
#define SUCCESS  1
#define FAILURE  0
#define I2D(num, c, r) ((r)*(num)+(c)) // Indexing into a 1D array from 2D space

#endif
