#include "heat_sim.h"
#include "matrix_lib.h"
#include "err_code.h"
#include "device_picker.h"

char * getKernelSource(char *filename);

int main(int argc, char *argv[])
{
    float *temp1_ref, *temp2_ref, *temp; // reference matrices in host memory
	float *temp_out;			  // host output matrix for GPU
	
    int size;               // number of elements in each matrix

    cl_mem temp1, temp2, temp_tmp;   // matrices in device memory

    double start_time;      // starting time
    double run_time;        // run time
	float transfer;
	float tfac = 8.418e-5; // thermal diffusivity of silver

    char * kernelsource;    // kernel source string

    cl_int err;             // error code returned from OpenCL calls
    cl_device_id     device;        // compute device id
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        kernel;        // compute kernel

    int ni = WIDTH;
	int nj = HEIGHT;
	int tSteps = COUNT;
	bool saveData = 0;
	
//--------------------------------------------------------------------------------
// Check flags for custom input and allocate memory
//--------------------------------------------------------------------------------
	
	for (int i = 1; i < argc; i++) {
		
		if (strcmp(argv[i], "help") == 0 || strcmp(argv[i], "?") == 0) {
			printf("      -mW= MatWidth (Width of matrices, default 320)\n");
			printf("      -mH= MatHeight (Height of matrices, default 320)\n");
			printf("      -tS= TimeSteps (Number of time steps, default 30)\n");
			printf("      -sF (Save numeric data to heat_con.csv)\n");

			return 0;
		}
	  
		if (strcmp(argv[i], "-mW=") == 0) ni = atoi(argv[i+1]);
		if (strcmp(argv[i], "-mH=") == 0) nj = atoi(argv[i+1]);
		if (strcmp(argv[i], "-tS=") == 0) tSteps = atoi(argv[i+1]);
		if (strcmp(argv[i], "-sF") == 0) saveData = 1;
	}
	
	size = ni * nj;

    temp1_ref = (float *)malloc(size * sizeof(float));
    temp2_ref = (float *)malloc(size * sizeof(float));
	temp_out = (float *)malloc(size * sizeof(float));
	
	transfer = 2 * sizeof(float) * size / 1024;
	
//--------------------------------------------------------------------------------
// Create a context, queue and device
//--------------------------------------------------------------------------------

    cl_uint deviceIndex = 0;
    parseArguments(argc, argv, &deviceIndex);

    // Get list of devices
    cl_device_id devices[MAX_DEVICES];
    unsigned numDevices = getDeviceList(devices);

    // Check device index in range
    if (deviceIndex >= numDevices)
    {
      printf("Invalid device index (try '--list')\n");
      return EXIT_FAILURE;
    }

    device = devices[deviceIndex];

    char name[MAX_INFO_STRING];
    getDeviceName(device, name);
    printf("\nUsing OpenCL device: %s\n", name);

    // Create a compute context
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    commands = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Creating command queue");

//--------------------------------------------------------------------------------
// Initialise matrices, setup the buffers and write them into global memory
//--------------------------------------------------------------------------------
	initmat(size, temp1_ref, temp_out, temp2_ref);
	
    temp1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * size, temp_out, &err);
    checkError(err, "Creating buffer temp1");
    temp2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * size, temp2_ref, &err);
    checkError(err, "Creating buffer temp2");
	
//--------------------------------------------------------------------------------
// Run host reference version
//--------------------------------------------------------------------------------
    printf("\n===== Executing %d times host CPU version, order %d x %d ======\n", tSteps, ni, nj);
	
	//remove previous file
	if(saveData == 1) remove("heat_con.csv");
	
	start_time = wtime();
	
    for (int i = 0; i < tSteps; i++) {
		if(saveData == 0)
			step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);
		else
			step_kernel_file(ni, nj, tfac, temp1_ref, temp2_ref);
		
		// swap temperature pointer
		temp = temp1_ref;
		temp1_ref = temp2_ref;
		temp2_ref = temp;
		
	}
	
    run_time  = wtime() - start_time;
	
	if(saveData == 1) printf("Numeric data saved to heat_con.csv\n");
	
	printf("Overall CPU preformance: %.3f miliseconds, transfer %.0f kB.\n",
	run_time*1000, transfer);
	
	run_time = 0;

//--------------------------------------------------------------------------------
// Run GPU version
//--------------------------------------------------------------------------------
    kernelsource = getKernelSource("C_heat_conduction.cl");
    // Create the comput program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & kernelsource, NULL, &err);
    checkError(err, "Creating program with C_heat_conduction.cl");
    free(kernelsource);
    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from the program
    kernel = clCreateKernel(program, "step_kernel_mod", &err);
    if (!kernel || err != CL_SUCCESS)
    checkError(err, "Creating kernel with C_heat_conduction.cl");

    printf("\n===== Executing %d times device GPU version, order %d x %d ======\n",
		tSteps, ni, nj);
		
    for (int i = 0; i < tSteps; i++)
    {
		   
        err =  clSetKernelArg(kernel, 0, sizeof(int),    &ni);
        err |= clSetKernelArg(kernel, 1, sizeof(int),    &nj);
        err |= clSetKernelArg(kernel, 2, sizeof(float),  &tfac);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &temp1);
        err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &temp2);

        checkError(err, "Setting kernel args");

        start_time = wtime();

        // Execute the kernel
		const size_t global[2] = {ni-1, nj-1};
        err = clEnqueueNDRangeKernel(
            commands,
            kernel,
            2, NULL,
            global, 0,
            0, NULL, NULL);
        checkError(err, "Enqueueing kernel");

        err = clFinish(commands);
        checkError(err, "Waiting for kernel to finish");

        run_time += (wtime() - start_time) * 1000;
		
		// swap temperature pointers
		temp_tmp = temp1;
		temp1 = temp2;
		temp2 = temp_tmp;	
		
    } // end for loop
	
	err = clEnqueueReadBuffer(
            commands, temp1, CL_TRUE, 0,
            sizeof(float) * size, temp_out,
            0, NULL, NULL);
        checkError(err, "Reading back temp2");
	
	results(ni, nj, temp_out, temp1_ref);
	printf("Overall GPU performance: %.3f miliseconds, transfer %.0f kB. \n\n",
	run_time, transfer);

//--------------------------------------------------------------------------------
// Clean up
//--------------------------------------------------------------------------------
	free(temp1_ref);
    free(temp2_ref);
	free(temp_out);
	
	clReleaseMemObject(temp1);
    clReleaseMemObject(temp2);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return EXIT_SUCCESS;
}


char * getKernelSource(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error: Could not open kernel source file\n");
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    int len = ftell(file) + 1;
    rewind(file);

    char *source = (char *)calloc(sizeof(char), len);
    if (!source)
    {
        fprintf(stderr, "Error: Could not allocate memory for source string\n");
        exit(EXIT_FAILURE);
    }
    fread(source, sizeof(char), len, file);
    fclose(file);
    return source;
}
