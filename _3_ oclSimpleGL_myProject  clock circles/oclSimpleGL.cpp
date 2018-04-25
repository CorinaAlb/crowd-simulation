/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    This example demonstrates how to use the OpenCL/OpenGL interoperability to
    dynamically modify a vertex buffer using a OpenCL kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Create an OpenCL memory object from the vertex buffer object
    3. Acquire the VBO for writing from OpenCL
    4. Run OpenCL kernel to modify the vertex positions
    5. Release the VBO for returning ownership to OpenGL
    6. Render the results using OpenGL

    Host code
*/

#define UNIX
#define GL_INTEROP

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics Includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenGL/OpenGL.h>
    #include <GLUT/glut.h>
#else
    #include <GL/freeglut.h>
    #ifdef UNIX
       #include <GL/glx.h>
    #endif
#endif

// Includes
#include <memory>
#include <iostream>
#include <cassert>
#include <vector>
#include <fstream>

// Utilities, OpenCL and system includes
#include <oclUtils.h>
#include <shrQATest.h>

#if defined (__APPLE__) || defined(MACOSX)
   #define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
   #define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#endif

// Constants, defines, typedefs and global declarations
//*****************************************************************************
#define REFRESH_DELAY	  10 //ms

// Rendering window vars
const unsigned int window_width = 512;
const unsigned int window_height = 512;
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

const unsigned int no_circles = 3;
const unsigned int total_no_points = 36;

/* vertices coordinates definition */
GLfloat center_positons[6] = {
	-0.5, 0.0,
	0.0, 0.0,
	0.5, 0.0
};

GLfloat radius[6] = {
    0.2, 0.2,
	0.2, 0.3,
	0.25, 0.25
};

GLfloat vertices_on_circle[72] = {
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0
};

GLfloat mask[36] = {
    0, NAN,
    NAN, NAN,
    NAN, NAN,
    NAN, 0,
    NAN, NAN,
    NAN, NAN,
    0, NAN,
    NAN, NAN,
    NAN, NAN,
    NAN, NAN,
    NAN, NAN,
    NAN, NAN,
    NAN, NAN,
    NAN, NAN,
    NAN, NAN,
    NAN, NAN,
    NAN, NAN,
    NAN, NAN
};

GLfloat centers_mask[6] = {
    1, 0,
	1, 1,
	0, 1
};

/* vertices colors definition */
GLfloat vertices_colors[144] = {
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0
};

GLint no_points_on_circle[3] = {
    12, 12, 12
};

// OpenCL vars
cl_platform_id cpPlatform;
cl_context cxGPUContext;
cl_device_id* cdDevices;
cl_uint uiDevCount;
cl_command_queue cqCommandQueue;

cl_kernel ckKernel_draw;
cl_kernel ckKernel_color_change;
cl_kernel ckKernel_centers;
cl_kernel ckKernel_inc_radius;
cl_kernel ckKernel_dec_radius;
cl_kernel ckKernel_no_points_on_circle;

cl_mem vbo_cl_positions;
cl_mem vbo_cl_centers;
cl_mem vbo_cl_radius;
cl_mem vbo_cl_colors;
cl_mem vbo_cl_mask;
cl_mem vbo_cl_centers_mask;
cl_mem vbo_cl_no_points_on_circle;

cl_program cpProgram;
cl_int ciErrNum;

char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL = NULL;             // Buffer to hold source for compilation
size_t szGlobalWorkSize[] = {3, 1};
size_t szGlobalWorkSizeColors[] = {3, 1};
const char* cExecutableName = NULL;

// vbo variables
GLuint vbo_positions;
GLuint vbo_centers;
GLuint vbo_radius;
GLuint vbo_colors;
GLuint vbo_mask;
GLuint vbo_centers_mask;
GLuint vbo_points_on_circle;

int iGLUTWindowHandle = 0;          // handle to the GLUT window

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

// Sim and Auto-Verification parameters
float anim = 0.0;
int iFrameCount = 0;                // FPS count for averaging
int iFrameTrigger = 90;             // FPS trigger for sampling
int iFramesPerSec = 0;              // frames per second
int iTestSets = 3;
int g_Index = 0;
shrBOOL bQATest = shrFALSE;
shrBOOL bNoPrompt = shrFALSE;

int *pArgc = NULL;
char **pArgv = NULL;

bool increment = true;

// Forward Function declarations
//*****************************************************************************
// OpenCL functionality
void runKernel();
void saveResultOpenCL(int argc, const char** argv, const GLuint& vbo);

// GL functionality
void InitGL(int* argc, char** argv);
void createVBOPositions(GLuint* vbo);
void createVBOColors(GLuint* vbo);
void createVBOCenters(GLuint* vbo);
void createVBORadius(GLuint* vbo);
void createVBOMask(GLuint* vbo);
void createVBOCentersMask(GLuint* vbo);
void createVBONoPointsOnCircle(GLuint* vbo);

void DisplayGL();
void KeyboardGL(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Helpers
void TestNoGL();
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;

// Main program
//*****************************************************************************
int main(int argc, char** argv)
{
    pArgc = &argc;
    pArgv = argv;

    // start logs
    shrQAStart(argc, argv);
    cExecutableName = argv[0];
    shrSetLogFileName ("oclSimpleGL.txt");
    shrLog("%s Starting...\n\n", argv[0]);

    // check command line args
    if (argc > 1)
    {
        bQATest   = shrCheckCmdLineFlag(argc, (const char**)argv, "qatest");
        bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");
    }

    // Initialize OpenGL items (if not No-GL QA test)
    shrLog("%sInitGL...\n\n", bQATest ? "Skipping " : "Calling ");
    if(!bQATest)
    {
        InitGL(&argc, argv);
    }

    //Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Get the number of GPU devices available to the platform
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiDevCount);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Create the device list
    cdDevices = new cl_device_id [uiDevCount];
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiDevCount, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Get device requested on command line, if any
    unsigned int uiDeviceUsed = 0;
    unsigned int uiEndDev = uiDevCount - 1;
    if(shrGetCmdLineArgumentu(argc, (const char**)argv, "device", &uiDeviceUsed ))
    {
      uiDeviceUsed = CLAMP(uiDeviceUsed, 0, uiEndDev);
      uiEndDev = uiDeviceUsed;
    }

    // Check if the requested device (or any of the devices if none requested) supports context sharing with OpenGL
    if(!bQATest)
    {
        bool bSharingSupported = false;
        for(unsigned int i = uiDeviceUsed; (!bSharingSupported && (i <= uiEndDev)); ++i)
        {
            size_t extensionSize;
            ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize );
            oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
            if(extensionSize > 0)
            {
                char* extensions = (char*)malloc(extensionSize);
                ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, extensionSize, extensions, &extensionSize);
                oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
                std::string stdDevString(extensions);
                free(extensions);

                size_t szOldPos = 0;
                size_t szSpacePos = stdDevString.find(' ', szOldPos); // extensions string is space delimited
                while (szSpacePos != stdDevString.npos)
                {
                    if( strcmp(GL_SHARING_EXTENSION, stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str()) == 0 )
                    {
                        // Device supports context sharing with OpenGL
                        uiDeviceUsed = i;
                        bSharingSupported = true;
                        break;
                    }
                    do
                    {
                        szOldPos = szSpacePos + 1;
                        szSpacePos = stdDevString.find(' ', szOldPos);
                    }
                    while (szSpacePos == szOldPos);
                }
            }
        }

        shrLog("%s...\n\n", bSharingSupported ? "Using CL-GL Interop" : "No device found that supports CL/GL context sharing");
        oclCheckErrorEX(bSharingSupported, true, pCleanup);

        // Define OS-specific context properties and create the OpenCL context
        #if defined (__APPLE__)
            CGLContextObj kCGLContext = CGLGetCurrentContext();
            CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
            cl_context_properties props[] =
            {
                CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup,
                0
            };
            cxGPUContext = clCreateContext(props, 0,0, NULL, NULL, &ciErrNum);
        #else
            #ifdef UNIX
                cl_context_properties props[] =
                {
                    CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
                    CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
                    CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform,
                    0
                };
                cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);
            #else // Win32
                cl_context_properties props[] =
                {
                    CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
                    CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
                    CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform,
                    0
                };
                cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);
            #endif
        #endif
    }
    else
    {
        cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 0};
        cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);
    }

    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Log device used (reconciled for requested requested and/or CL-GL interop capable devices, as applies)
    shrLog("Device # %u, ", uiDeviceUsed);
    oclPrintDevName(LOGBOTH, cdDevices[uiDeviceUsed]);
    shrLog("\n");

    // create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[uiDeviceUsed], 0, &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Program Setup
    size_t program_length;
    cPathAndName = shrFindFilePath("simpleGL.cl", argv[0]);
    shrCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &program_length);
    shrCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);

    // create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1,
					  (const char **) &cSourceCL, &program_length, &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // build the program
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclSimpleGL.ptx");
        Cleanup(EXIT_FAILURE);
    }

    // create the kernel
    ckKernel_draw = clCreateKernel(cpProgram, "draw_circle", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ckKernel_color_change = clCreateKernel(cpProgram, "change_circle_color", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ckKernel_centers = clCreateKernel(cpProgram, "move_circle_center", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ckKernel_inc_radius = clCreateKernel(cpProgram, "increment_radius", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ckKernel_dec_radius = clCreateKernel(cpProgram, "decrement_radius", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
//    ckKernel_no_points_on_circle = clCreateKernel(cpProgram, "points_on_circle", &ciErrNum);
//    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // create VBO (if using standard GL or CL-GL interop), otherwise create Cl buffer
    createVBOPositions(&vbo_positions);
    createVBOCenters(&vbo_centers);
    createVBORadius(&vbo_radius);
    createVBOColors(&vbo_colors);
    createVBOMask(&vbo_mask);
    createVBOCentersMask(&vbo_centers_mask);
    createVBONoPointsOnCircle(&vbo_points_on_circle);

    //clEnqueueWriteBuffer(cqCommandQueue, vbo_cl, CL_TRUE, 0, 6 * sizeof(float), vertices_position, 0, NULL, NULL);

    // set the args values
    ciErrNum  = clSetKernelArg(ckKernel_draw, 0, sizeof(cl_mem), (void *) &vbo_cl_centers);
    ciErrNum |= clSetKernelArg(ckKernel_draw, 1, sizeof(cl_mem), (void *) &vbo_cl_radius);
    ciErrNum |= clSetKernelArg(ckKernel_draw, 2, sizeof(cl_mem), (void *) &vbo_cl_positions);
    //ciErrNum |= clSetKernelArg(ckKernel_draw, 3, sizeof(cl_mem), (void *) &vbo_cl_no_points_on_circle);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

     // set the args values
    ciErrNum  = clSetKernelArg(ckKernel_color_change, 0, sizeof(cl_mem), (void *) &vbo_cl_radius);
    ciErrNum |= clSetKernelArg(ckKernel_color_change, 1, sizeof(cl_mem), (void *) &vbo_cl_positions);
    ciErrNum |= clSetKernelArg(ckKernel_color_change, 2, sizeof(cl_mem), (void *) &vbo_cl_colors);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // set the args values
    ciErrNum  = clSetKernelArg(ckKernel_centers, 0, sizeof(cl_mem), &vbo_cl_centers);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // set the args values
    ciErrNum  = clSetKernelArg(ckKernel_inc_radius, 0, sizeof(cl_mem), (void *) &vbo_cl_radius);
    ciErrNum |= clSetKernelArg(ckKernel_inc_radius, 1, sizeof(cl_mem), (void *) &vbo_cl_centers_mask);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    ciErrNum  = clSetKernelArg(ckKernel_dec_radius, 0, sizeof(cl_mem), (void *) &vbo_cl_radius);
    ciErrNum |= clSetKernelArg(ckKernel_dec_radius, 1, sizeof(cl_mem), (void *) &vbo_cl_centers_mask);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

//    ciErrNum  = clSetKernelArg(ckKernel_no_points_on_circle, 0, sizeof(cl_mem), (void *) &vbo_cl_radius);
//    ciErrNum |= clSetKernelArg(ckKernel_no_points_on_circle, 1, sizeof(cl_mem), (void *) &vbo_cl_no_points_on_circle);
//    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // init timer 1 for fps measurement
    shrDeltaT(1);

    // Start main GLUT rendering loop for processing and rendering,
	// or otherwise run No-GL Q/A test sequence
    shrLog("\n%s...\n", bQATest ? "No-GL test sequence" : "Standard GL Loop");
    if(!bQATest)
    {
        glutMainLoop();
    }

    // Normally unused return path
    Cleanup(EXIT_SUCCESS);
}


// Initialize GL
//*****************************************************************************
void InitGL(int* argc, char** argv)
{
    // initialize GLUT
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - window_width/2,
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - window_height/2);
    glutInitWindowSize(window_width, window_height);
    iGLUTWindowHandle = glutCreateWindow("OpenCL/GL Interop (VBO)");
#if !(defined (__APPLE__) || defined(MACOSX))
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif

    // register GLUT callback functions
    glutDisplayFunc(DisplayGL);
    glutKeyboardFunc(KeyboardGL);
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);

	// initialize necessary OpenGL extensions
    glewInit();
    GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object");
    shrCheckErrorEX(bGLEW, shrTRUE, pCleanup);

    // default initialization
    glClearColor(0.0, 0.5, 1.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    return;
}

// Run the OpenCL part of the computation
//*****************************************************************************
void runKernel()
{
    ciErrNum = CL_SUCCESS;

#ifdef GL_INTEROP
    // map OpenGL buffer object for writing from OpenCL
    glFinish();
    ciErrNum  = clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_positions, 0, 0, 0 );
    ciErrNum |= clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_centers, 0, 0, 0 );
    ciErrNum |= clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_colors, 0, 0, 0 );
    ciErrNum |= clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_radius, 0, 0, 0 );
    ciErrNum |= clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_mask, 0, 0, 0 );
    ciErrNum |= clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_centers_mask, 0, 0, 0 );
    //ciErrNum |= clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_no_points_on_circle, 0, 0, 0 );

    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
#endif

    // Set arg 3 and execute the kernel
    //ciErrNum = clSetKernelArg(ckKernel, 1, sizeof(float), &anim);
    //ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_centers, 1, NULL, szGlobalWorkSize, NULL, 0, 0, 0 );
    //ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_no_points_on_circle, 1, NULL, szGlobalWorkSizeColors, NULL, 0, 0, 0 );
    //shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_color_change, 1, NULL, szGlobalWorkSizeColors, NULL, 0, 0, 0 );
    if (increment)
    {
        ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_inc_radius, 1, NULL, szGlobalWorkSize, NULL, 0, 0, 0 );
    } else {
        ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_dec_radius, 1, NULL, szGlobalWorkSize, NULL, 0, 0, 0 );
    }
    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_draw, 1, NULL, szGlobalWorkSize, NULL, 0, 0, 0 );
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

#ifdef GL_INTEROP
    // unmap buffer object
    ciErrNum  = clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_positions, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_centers, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_colors, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_radius, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_mask, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_centers_mask, 0, 0, 0 );
    //ciErrNum  = clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_no_points_on_circle, 0, 0, 0 );
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    clFinish(cqCommandQueue);
#else

    // Explicit Copy
    // map the PBO to copy data from the CL buffer via host
    glBindBufferARB(GL_ARRAY_BUFFER, vbo);

    // map the buffer object into client's memory
    void* ptr = glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY_ARB);

    ciErrNum = clEnqueueReadBuffer(cqCommandQueue, vbo_cl, CL_TRUE, 0, sizeof(float) * 4 * mesh_height * mesh_width, ptr, 0, NULL, NULL);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    glUnmapBufferARB(GL_ARRAY_BUFFER);
#endif
}

// Create VBO
//*****************************************************************************
void createVBOPositions(GLuint* vbo)
{
    // create VBO
    unsigned int size = total_no_points * 2 * sizeof(float);
    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, vertices_on_circle, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_positions = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_positions = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }

}

void createVBOCenters(GLuint* vbo)
{
    // create VBO
    unsigned int size = no_circles * 2 * sizeof(float);
    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, center_positons, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_centers = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_centers = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }

}

void createVBORadius(GLuint* vbo)
{
    // create VBO
    unsigned int size = no_circles * 2 * sizeof(float);
    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, radius, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_radius = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_radius = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_radius = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }

}

void createVBOColors(GLuint* vbo)
{
    // create VBO
    unsigned int size = total_no_points * 4 * sizeof(float);
    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, vertices_colors, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_colors = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_colors = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_colors = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }

}

void createVBOMask(GLuint* vbo)
{
    // create VBO
    unsigned int size = total_no_points * 2 * sizeof(float);
    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, mask, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_mask = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_mask = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_mask = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }

}

void createVBOCentersMask(GLuint* vbo)
{
    // create VBO
    unsigned int size = no_circles * 2 * sizeof(int);
    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, centers_mask, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_centers_mask = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
2222222222222222222222222222222222222222221            vbo_cl_mask = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_centers_mask = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }

}

void createVBONoPointsOnCircle(GLuint* vbo)
{
    // create VBO
    unsigned int size = no_circles * 2 * sizeof(float);
    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, no_points_on_circle, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_no_points_on_circle = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_radius = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_no_points_on_circle = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }

}
// Display callback
//*****************************************************************************
void DisplayGL()
{
    // run OpenCL kernel to generate vertex positions
    runKernel();

    // clear graphics then render from the vbo
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_PROGRAM_POINT_SIZE_EXT);

    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_positions);
    glVertexPointer(2, GL_FLOAT, 0, 0);

    glEnableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
    glColorPointer(4, GL_FLOAT, 0, 0);

    glPointSize(10);
    glDrawArrays(GL_POINTS, 0, total_no_points);

    glDisableClientState(GL_VERTEX_ARRAY);

    // flip backbuffer to screen
    glutSwapBuffers();
}

void timerEvent(int value)
{
    glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

// Keyboard events handler
//*****************************************************************************
void KeyboardGL(unsigned char key, int x, int y)
{
    switch(key)
    {
         case 033: // octal equivalent of the Escape key
            glutLeaveMainLoop();
            break;
        case 'd':
            increment = false;
            break;
        case 'i':
            increment = true;
            break;
    }
}

// Function to clean up and exit
//*****************************************************************************
void Cleanup(int iExitCode)
{
    // Cleanup allocated objects
    shrLog("\nStarting Cleanup...\n\n");
    if(ckKernel_draw)       clReleaseKernel(ckKernel_draw);
    if(ckKernel_centers)       clReleaseKernel(ckKernel_centers);
    if(cpProgram)      clReleaseProgram(cpProgram);
    if(cqCommandQueue) clReleaseCommandQueue(cqCommandQueue);
    if(vbo_positions)
    {
        glBindBuffer(1, vbo_positions);
        glDeleteBuffers(1, &vbo_positions);
        vbo_positions = 0;
    }
    if(vbo_cl_positions)clReleaseMemObject(vbo_cl_positions);
    if(vbo_centers)
    {
        glBindBuffer(1, vbo_centers);
        glDeleteBuffers(1, &vbo_centers);
        vbo_centers = 0;
    }
    if(vbo_cl_centers)clReleaseMemObject(vbo_cl_centers);
    if(vbo_colors)
    {
        glBindBuffer(1, vbo_colors);
        glDeleteBuffers(1, &vbo_colors);
        vbo_colors = 0;
    }
    if(vbo_cl_colors)clReleaseMemObject(vbo_cl_colors);
    if(vbo_radius)
    {
        glBindBuffer(1, vbo_radius);
        glDeleteBuffers(1, &vbo_radius);
        vbo_radius = 0;
    }
    if(vbo_cl_radius)clReleaseMemObject(vbo_cl_radius);

    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
    if(cdDevices)delete(cdDevices);

    // finalize logs and leave
    shrQAFinish2(bQATest, *pArgc, (const char **)pArgv, (iExitCode == 0) ? QA_PASSED : QA_FAILED );
    if (bQATest || bNoPrompt)
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\n", cExecutableName);
    }
    else
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\nPress <Enter> to Quit\n", cExecutableName);
        #ifdef WIN32
            getchar();
        #endif
    }
    exit (iExitCode);
}
