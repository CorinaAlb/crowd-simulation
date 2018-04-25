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
#include <string>

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

#define NO_POINTS_ON_CIRCLE     360

// Rendering window vars
const unsigned int window_width = 1200;
const unsigned int window_height = 1200;
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

// OpenCL vars
cl_platform_id cpPlatform;
cl_context cxGPUContext;
cl_device_id* cdDevices;
cl_uint uiDevCount;
cl_command_queue cqCommandQueue;

cl_kernel ckKernel_draw;
cl_kernel ckKernel_draw_using_mask;
cl_kernel ckKernel_color_change;
cl_kernel ckKernel_analize_circle_color;
cl_kernel ckKernel_centers;
cl_kernel ckKernel_modify_radius;
cl_kernel ckKernel_change_speed_mask_sign;
cl_kernel ckKernel_influence_circle_mask;

cl_mem vbo_cl_positions;
cl_mem vbo_cl_centers;
cl_mem vbo_cl_radius;
cl_mem vbo_cl_speed_mask;
cl_mem vbo_cl_colors;
cl_mem vbo_cl_centers_mask;
cl_mem vbo_cl_colors_mask;
cl_mem vbo_cl_influence_circles_mask;

cl_program cpProgram;
cl_int ciErrNum;

char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL = NULL;             // Buffer to hold source for compilation
const char* cExecutableName = NULL;

// vbo variables
GLuint vbo_positions;
GLuint vbo_centers;
GLuint vbo_radius_mask;
GLuint vbo_speed_mask;
GLuint vbo_radius;
GLuint vbo_colors;
GLuint vbo_mask;
GLuint vbo_centers_mask;
GLuint vbo_colors_mask;
GLuint vbo_influence_circle_mask;

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

// Forward Function declarations
//*****************************************************************************
// OpenCL functionality
void runKernel();
void saveResultOpenCL(int argc, const char** argv, const GLuint& vbo);

// GL functionality
void InitGL(int* argc, char** argv);
void createVBOTotalPoints(GLuint* vbo);
void createVBOColors(GLuint* vbo);
void createVBOCenters(GLuint* vbo);
void createVBORadius(GLuint* vbo);
void createVBORadiusMask(GLuint* vbo);
void createVBOCentersMask(GLuint* vbo);
void createVBOSpeedMask(GLuint* vbo);
void createVBOSpeedSignMask(GLuint* vbo);
void createVBOColorsMask(GLuint* vbo);
void createVBOInfluenceCircleMask(GLuint* vbo);

void DisplayGL();
void KeyboardGL(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Helpers
void TestNoGL();
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;

void mappingStage();

bool increment = true;
int circle_index = -1;

int no_circles = 0;
int influence_function_no = 0;

int index_influence_function = 0;

GLfloat* circles_center_position;
GLfloat* circles_colors;
GLfloat* circles_radius;
GLfloat* circles_speed_mask;
GLint* draw_circles_mask;
GLint* pulse_circles_mask;
GLint* influence_circles_mask;

GLfloat* all_points;
GLfloat* vertices_colors;

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

    mappingStage();

    // create the kernel
    ckKernel_draw = clCreateKernel(cpProgram, "draw_all_circles", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ckKernel_draw_using_mask = clCreateKernel(cpProgram, "draw_circles_using_mask", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ckKernel_analize_circle_color = clCreateKernel(cpProgram, "analize_circle_color", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ckKernel_color_change = clCreateKernel(cpProgram, "circle_expansion_color_change", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ckKernel_centers = clCreateKernel(cpProgram, "move_circle_center", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ckKernel_modify_radius = clCreateKernel(cpProgram, "modify_radius", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ckKernel_change_speed_mask_sign = clCreateKernel(cpProgram, "change_speed_mask_sign", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ckKernel_influence_circle_mask = clCreateKernel(cpProgram, "influence_circles_mask", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // create VBO (if using standard GL or CL-GL interop), otherwise create Cl buffer
    createVBOTotalPoints(&vbo_positions);
    createVBOCenters(&vbo_centers);
    createVBORadius(&vbo_radius);
    createVBOColors(&vbo_colors);
    createVBOCentersMask(&vbo_centers_mask);
    createVBOColorsMask(&vbo_colors_mask);
    createVBOSpeedMask(&vbo_speed_mask);
    createVBOInfluenceCircleMask(&vbo_influence_circle_mask);

    // set the args values
    ciErrNum  = clSetKernelArg(ckKernel_draw, 0, sizeof(cl_mem), (void *) &vbo_cl_centers);
    ciErrNum |= clSetKernelArg(ckKernel_draw, 1, sizeof(cl_mem), (void *) &vbo_cl_radius);
    ciErrNum |= clSetKernelArg(ckKernel_draw, 2, sizeof(cl_mem), (void *) &vbo_cl_positions);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // set the args values
    ciErrNum  = clSetKernelArg(ckKernel_draw_using_mask, 0, sizeof(cl_mem), (void *) &vbo_cl_centers);
    ciErrNum |= clSetKernelArg(ckKernel_draw_using_mask, 1, sizeof(cl_mem), (void *) &vbo_cl_radius);
    ciErrNum |= clSetKernelArg(ckKernel_draw_using_mask, 2, sizeof(cl_mem), (void *) &vbo_cl_positions);
    ciErrNum |= clSetKernelArg(ckKernel_draw_using_mask, 3, sizeof(cl_mem), (void *) &vbo_cl_centers_mask);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

     // set the args values
    ciErrNum  = clSetKernelArg(ckKernel_color_change, 0, sizeof(cl_mem), (void *) &vbo_cl_radius);
    ciErrNum |= clSetKernelArg(ckKernel_color_change, 1, sizeof(cl_mem), (void *) &vbo_cl_positions);
    ciErrNum |= clSetKernelArg(ckKernel_color_change, 2, sizeof(cl_mem), (void *) &vbo_cl_colors);
    ciErrNum |= clSetKernelArg(ckKernel_color_change, 3, sizeof(cl_mem), (void *) &vbo_cl_colors_mask);
    ciErrNum |= clSetKernelArg(ckKernel_color_change, 4, sizeof(cl_mem), (void *) &vbo_cl_centers);
    ciErrNum |= clSetKernelArg(ckKernel_color_change, 5, sizeof(cl_mem), (void *) &vbo_cl_centers_mask);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum  = clSetKernelArg(ckKernel_analize_circle_color, 0, sizeof(cl_mem), (void *) &vbo_cl_radius);
    ciErrNum |= clSetKernelArg(ckKernel_analize_circle_color, 1, sizeof(cl_mem), (void *) &vbo_cl_positions);
    ciErrNum |= clSetKernelArg(ckKernel_analize_circle_color, 2, sizeof(cl_mem), (void *) &vbo_cl_colors);
    ciErrNum |= clSetKernelArg(ckKernel_analize_circle_color, 3, sizeof(cl_mem), (void *) &vbo_cl_colors_mask);
    ciErrNum |= clSetKernelArg(ckKernel_analize_circle_color, 4, sizeof(cl_mem), (void *) &vbo_cl_centers);
    ciErrNum |= clSetKernelArg(ckKernel_analize_circle_color, 5, sizeof(cl_mem), (void *) &vbo_cl_centers_mask);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    ciErrNum |= clSetKernelArg(ckKernel_influence_circle_mask, 0, sizeof(cl_mem), (void *) &vbo_cl_influence_circles_mask);
    ciErrNum |= clSetKernelArg(ckKernel_influence_circle_mask, 1, sizeof(cl_mem), (void *) &vbo_cl_colors);
    ciErrNum |= clSetKernelArg(ckKernel_influence_circle_mask, 2, sizeof(cl_mem), (void *) &vbo_cl_colors_mask);
    ciErrNum |= clSetKernelArg(ckKernel_influence_circle_mask, 3, sizeof(cl_mem), (void *) &vbo_cl_radius);
    ciErrNum |= clSetKernelArg(ckKernel_influence_circle_mask, 4, sizeof(cl_mem), (void *) &vbo_cl_centers);
    ciErrNum |= clSetKernelArg(ckKernel_influence_circle_mask, 5, sizeof(cl_mem), (void *) &vbo_cl_positions);
    ciErrNum |= clSetKernelArg(ckKernel_influence_circle_mask, 6, sizeof(cl_mem), (void *) &vbo_cl_centers_mask);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // set the args values
    ciErrNum  = clSetKernelArg(ckKernel_centers, 0, sizeof(cl_mem), &vbo_cl_centers);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // set the args values
    ciErrNum  = clSetKernelArg(ckKernel_modify_radius, 0, sizeof(cl_mem), (void *) &vbo_cl_radius);
    ciErrNum |= clSetKernelArg(ckKernel_modify_radius, 1, sizeof(cl_mem), (void *) &vbo_cl_speed_mask);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // set the args values
    ciErrNum  = clSetKernelArg(ckKernel_change_speed_mask_sign, 0, sizeof(cl_mem), (void *) &vbo_cl_speed_mask);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

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
    glClearColor(0.65, 0.75, 0.85, 0.0);
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
    ciErrNum |= clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_centers_mask, 0, 0, 0 );
    ciErrNum |= clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_colors_mask, 0, 0, 0 );
    ciErrNum |= clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_speed_mask, 0, 0, 0 );
    ciErrNum |= clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_influence_circles_mask, 0, 0, 0 );
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
#endif

//    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_color_change, 1, NULL, szGlobalWorkSize, NULL, 0, 0, 0 );
//    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_analize_circle_color, 1, NULL, szGlobalWorkSize, NULL, 0, 0, 0 );
    size_t szGlobalWorkSizeInfluence[] = {influence_function_no, 1};
    size_t szGlobalWorkSize[] = {no_circles, 1};
    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_influence_circle_mask, 1, NULL, szGlobalWorkSizeInfluence, NULL, 0, 0, 0 );
    if (circle_index != -1)
    {
        ciErrNum |= clSetKernelArg(ckKernel_change_speed_mask_sign, 1, sizeof(int), &circle_index);
        ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_change_speed_mask_sign, 1, NULL, szGlobalWorkSize, NULL, 0, 0, 0 );
        circle_index = -1;
    }
    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_modify_radius, 1, NULL, szGlobalWorkSize, NULL, 0, 0, 0 );
    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_draw, 1, NULL, szGlobalWorkSize, NULL, 0, 0, 0 );
    //ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_draw_using_mask, 1, NULL, szGlobalWorkSize, NULL, 0, 0, 0 );
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

#ifdef GL_INTEROP
    // unmap buffer object
    ciErrNum  = clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_positions, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_centers, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_colors, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_radius, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_centers_mask, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_colors_mask, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_speed_mask, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_influence_circles_mask, 0, 0, 0 );
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

    glPointSize(6);
    glDrawArrays(GL_POINTS, 0, NO_POINTS_ON_CIRCLE * no_circles);

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
        case '0':
            circle_index = 0;
            break;
        case '1':
            circle_index = 1;
            break;
        case '2':
            circle_index = 2;
            break;
        case '3':
            circle_index = 3;
            break;
        case '4':
            circle_index = 4;
            break;
        case '5':
            circle_index = 5;
            break;
        case '6':
            circle_index = 6;
            break;
        case '7':
            circle_index = 7;
            break;
        case '8':
            circle_index = 8;
            break;
        case '9':
            circle_index = 9;
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

void mappingStage()
{
    std::string line;
    std::ifstream myfile("world.ads");

    if (myfile.is_open())
    {
        char *line_char;
        char *pointer_1;

        getline(myfile,line);

        if (line.compare(0, 10, "no_circles") == 0)
        {
            line_char = strdup(line.c_str());
            char *no_circles_char = strtok_r(line_char, ":", &pointer_1);
            no_circles_char = strtok_r(NULL, ":", &pointer_1);
            no_circles = atoi(no_circles_char);

            circles_center_position = new GLfloat [2 * no_circles];
            circles_colors = new GLfloat [4 * no_circles];
            circles_radius = new GLfloat [2 * no_circles];
            circles_speed_mask = new GLfloat [2 * no_circles];
            draw_circles_mask = new GLint [no_circles];
            pulse_circles_mask = new GLint [no_circles];

            for (int i=0; i < no_circles * 2; i++)
            {
                circles_speed_mask[i] = 0.0f;
                pulse_circles_mask[i/2] = 0;
                draw_circles_mask[i/2] = 0;
            }
        }

        int no_circles_check = no_circles;
        int index_position = -1;
        int index_color = -1;
        int index_radius = -1;

        while (getline(myfile, line))
        {
            // entities declaration
            if (line.compare(0, 6, "circle") == 0 && no_circles_check >= 0)
            {
                line_char = strdup(line.c_str());
                char *circle = strtok_r(line_char, ":| ", &pointer_1);

                if (circle == NULL)
                    break;

                circle = strtok_r(NULL, ":| ", &pointer_1);

                while (circle != NULL)
                {
                    if (strstr(circle, "position"))
                    {
                        circle = strtok_r(NULL, ":| ", &pointer_1);
                        circles_center_position[++index_position] = atof(circle);

                        circle = strtok_r(NULL, ":| ", &pointer_1);
                        circles_center_position[++index_position] = atof(circle);
                    }
                    if (strstr(circle, "color"))
                    {
                        circle = strtok_r(NULL, ":| ", &pointer_1);
                        circles_colors[++index_color] = atof(circle);
                        circle = strtok_r(NULL, ":| ", &pointer_1);
                        circles_colors[++index_color] = atof(circle);
                        circle = strtok_r(NULL, ":| ", &pointer_1);
                        circles_colors[++index_color] = atof(circle);
                        circle = strtok_r(NULL, ":| ", &pointer_1);
                        circles_colors[++index_color] = atof(circle);
                    }
                    if (strstr(circle, "radius"))
                    {
                        circle = strtok_r(NULL, ":| ", &pointer_1);
                        circles_radius[++index_radius] = atof(circle);
                        circle = strtok_r(NULL, ":| ", &pointer_1);
                        circles_radius[++index_radius] = atof(circle);
                    }
                    circle = strtok_r(NULL, ":| ", &pointer_1);
                }
                no_circles_check--;
            }
            // functions declaration => create masks for entities envolved
            else if (line.compare(0, 8, "function") == 0)
            {
                char *line_char = strdup(line.c_str());
                char *func = strtok_r(line_char, ": ", &pointer_1);

                if (func == NULL)
                    break;

                func = strtok_r(NULL, ": ", &pointer_1);

                // draw function
                if (strstr(func, "draw"))
                {
                    while (func != NULL)
                    {
                        func = strtok_r(NULL, "_ ", &pointer_1); // circle
                        func = strtok_r(NULL, "_ ", &pointer_1); // number

                        if (func != NULL)
                        {
                            draw_circles_mask[atoi(func) - 1] = 1;
                        }
                    }
                }
                // pulse function
                else if (strstr(func, "pulse"))
                {
                    while (func != NULL)
                    {
                        func = strtok_r(NULL, "_ ", &pointer_1); // circle
                        func = strtok_r(NULL, "_ ", &pointer_1); // number

                        int pulse_circle_index = -1;
                        if (func != NULL)
                        {
                            pulse_circle_index = atoi(func) -1;
                            pulse_circles_mask[pulse_circle_index] = 1;
                        }

                        func = strtok_r(NULL, ": ", &pointer_1); // speed

                        if (func != NULL)
                        {
                            func = strtok_r(NULL, ": ", &pointer_1); // value speed.x
                            circles_speed_mask[2 * pulse_circle_index] = atof(func);
                            func = strtok_r(NULL, ": ", &pointer_1); // value speed.y
                            circles_speed_mask[2 * pulse_circle_index + 1] = atof(func);
                        }
                    }
                }
                // influence function
                else if (strstr(func, "influence"))
                {
                    int master_circle;
                    int influenced_circle;

                    func = strtok_r(NULL, "_ ", &pointer_1); // circle
                    func = strtok_r(NULL, "_ ", &pointer_1); // number

                    if (func != NULL)
                    {
                        master_circle = atoi(func) - 1;
                    }

                    func = strtok_r(NULL, "_ ", &pointer_1); // circle
                    func = strtok_r(NULL, "_ ", &pointer_1); // number

                    if (func != NULL)
                    {
                        influenced_circle = atoi(func) - 1;
                    }

                    influence_circles_mask[index_influence_function] = master_circle;
                    index_influence_function++;
                    influence_circles_mask[index_influence_function] = influenced_circle;
                    index_influence_function++;
                }
            }
            // influence functions number
            else if (line.compare(0, 18, "influence_function") == 0)
            {
                char *line_char = strdup(line.c_str());
                char *func = strtok_r(line_char, ":", &pointer_1); // influence_function

                if (func == NULL)
                    break;

                func = strtok_r(NULL, ":", &pointer_1); // number
                influence_function_no = atoi(func);

                influence_circles_mask = new GLint [influence_function_no * 2];
            }
        }

        myfile.close();
    }
    else std::cout << "Unable to open file";
}

// Create VBO
//*****************************************************************************
void createVBOTotalPoints(GLuint* vbo)
{
    // create VBO
    printf("no circles :: %d\n", no_circles);
    unsigned int size = 360 * no_circles * 2 * sizeof(GLfloat);
    all_points = new GLfloat [size];

    for (int i=0; i < 360 * no_circles * 2; i++)
    {
            all_points[i] = 0.0f;
    }

    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, all_points, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_positions = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_positions = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
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
        glBufferData(GL_ARRAY_BUFFER, size, circles_center_position, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_centers = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_centers = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
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
        glBufferData(GL_ARRAY_BUFFER, size, circles_radius, GL_DYNAMIC_DRAW);

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
    unsigned int size = no_circles * 360 * 4 * sizeof(GLfloat);
    vertices_colors = new GLfloat [size];

    for (int i=0; i < 360 * no_circles * 4; i++)
    {
        vertices_colors[i] = 0.0f;
    }

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

void createVBOCentersMask(GLuint* vbo)
{
    // create VBO
    unsigned int size = no_circles * sizeof(GLint);

    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, draw_circles_mask, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_centers_mask = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_centers_mask = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
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

void createVBOInfluenceCircleMask(GLuint* vbo)
{
    // create VBO
    unsigned int size = influence_function_no * 2 * sizeof(GLint);

    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, influence_circles_mask, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_influence_circles_mask = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_influence_circles_mask = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_influence_circles_mask = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
}

void createVBOColorsMask(GLuint* vbo)
{
    // create VBO
    unsigned int size = no_circles * 4 * sizeof(float);
    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, circles_colors, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_colors_mask = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_colors_mask = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_colors_mask = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
}

void createVBOSpeedMask(GLuint* vbo)
{
    // create VBO
    unsigned int size = no_circles * 2 * sizeof(float);
    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, circles_speed_mask, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_speed_mask = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_speed_mask = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_speed_mask = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
}
