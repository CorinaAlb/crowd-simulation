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

#include <sys/time.h>

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
#define REFRESH_DELAY	    10 //ms

#define NO_PIECES           16

// Rendering window vars
const unsigned int window_width = 1000;
const unsigned int window_height = 1000;
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

// OpenCL vars
cl_platform_id cpPlatform;
cl_context cxGPUContext;
cl_device_id* cdDevices;
cl_uint uiDevCount;
cl_command_queue cqCommandQueue;

cl_kernel ckKernel_create_collision_map;
cl_kernel ckKernel_compute_velocity;
cl_kernel ckKernel_clean_collision_map;
cl_kernel ckKernel_update_target_positions;
cl_kernel ckKernel_move_to_target_path_faithful;
cl_kernel ckKernel_attraction;

cl_program cpProgram;
cl_int ciErrNum;
char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL = NULL;             // Buffer to hold source for compilation
const char* cExecutableName = NULL;

// vbo variables
GLuint vbo_obstacle_positions;
GLuint vbo_obstacle_colors;
GLuint vbo_positions;
GLuint vbo_old_positions;
GLuint vbo_velocity;
GLuint vbo_colors;
GLuint vbo_path_faithful;
GLuint vbo_gravitational_force;
GLuint vbo_attraction_map;

cl_mem vbo_cl_positions;
cl_mem vbo_cl_obstacle_positions;
cl_mem vbo_cl_obstacle_colors;
cl_mem vbo_cl_old_positions;
cl_mem vbo_cl_velocity;
cl_mem vbo_cl_colors;
cl_mem vbo_cl_collision_map;
cl_mem vbo_cl_path_faithful;
cl_mem vbo_cl_gravitational_force;
cl_mem vbo_cl_attraction_map;

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

// GL functionality
void InitGL(int* argc, char** argv);

void createVBOObstaclePositions(GLuint* vbo);
void createVBOObstacleColors(GLuint* vbo);
void createVBOPositions(GLuint* vbo);
void createVBOColors(GLuint* vbo);
void createVBOCollisionMap(GLuint* vbo);
void createVBOPathFaithful(GLuint* vbo);
void createVBOOldPositions(GLuint* vbo);
void createVBOVelocity(GLuint* vbo);
void createVBOGravitationalForce(GLuint* vbo);
void createVBOAttractionMap(GLuint* vbo);

void DisplayGL();
void timerEvent(int value);

// Helpers
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;

int no_points;
int no_attractions;
int no_obstacles;

GLfloat *obstacle_positions;
GLfloat *obstacle_colors;
GLfloat *points_position;
GLfloat *points_velocity;
GLfloat *points_old_position;
GLfloat *points_color;
GLfloat *path_faithful;
GLfloat *gravitational_force;

GLint *attraction_map;

GLfloat *pieces_coordinates_x = new GLfloat[4 * NO_PIECES];
GLfloat *pieces_coordinates_y = new GLfloat[4 * NO_PIECES];

struct timeval stop, start;

void init_world();
void init_pieces();

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

    init_world();
    init_pieces();

    ckKernel_move_to_target_path_faithful = clCreateKernel(cpProgram, "move_to_target_path_faithful", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ckKernel_compute_velocity = clCreateKernel(cpProgram, "compute_velocity", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    if (no_attractions > 0)
    {
        ckKernel_attraction = clCreateKernel(cpProgram, "attraction", &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }

    // create VBO (if using standard GL or CL-GL interop), otherwise create Cl buffer
    createVBOPositions(&vbo_positions);
    createVBOColors(&vbo_colors);
    createVBOPathFaithful(&vbo_path_faithful);
    createVBOOldPositions(&vbo_old_positions);
    createVBOVelocity(&vbo_velocity);
    createVBOGravitationalForce(&vbo_gravitational_force);
    createVBOObstaclePositions(&vbo_obstacle_positions);
    createVBOObstacleColors(&vbo_obstacle_colors);
    if (no_attractions > 0)
    {
        createVBOAttractionMap(&vbo_attraction_map);
    }

    ciErrNum  = clSetKernelArg(ckKernel_move_to_target_path_faithful, 0, sizeof(cl_mem), (void *) &vbo_cl_positions);
    ciErrNum |= clSetKernelArg(ckKernel_move_to_target_path_faithful, 1, sizeof(cl_mem), (void *) &vbo_cl_path_faithful);
    ciErrNum |= clSetKernelArg(ckKernel_move_to_target_path_faithful, 2, sizeof(cl_mem), (void *) &vbo_cl_velocity);
    ciErrNum |= clSetKernelArg(ckKernel_move_to_target_path_faithful, 3, sizeof(cl_mem), (void *) &vbo_cl_gravitational_force);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    ciErrNum  = clSetKernelArg(ckKernel_compute_velocity, 0, sizeof(cl_mem), (void *) &vbo_cl_positions);
    ciErrNum |= clSetKernelArg(ckKernel_compute_velocity, 1, sizeof(cl_mem), (void *) &vbo_cl_old_positions);
    ciErrNum |= clSetKernelArg(ckKernel_compute_velocity, 2, sizeof(cl_mem), (void *) &vbo_cl_velocity);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    if (no_attractions > 0)
    {
        ciErrNum  = clSetKernelArg(ckKernel_attraction, 0, sizeof(cl_mem), (void *) &vbo_cl_positions);
        ciErrNum  = clSetKernelArg(ckKernel_attraction, 1, sizeof(cl_mem), (void *) &vbo_cl_attraction_map);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }

    // If specified, compute and save off data for regression tests
    if(shrCheckCmdLineFlag(argc, (const char**) argv, "regression"))
    {
        // run OpenCL kernel once to generate vertex positions, then save results
        runKernel();
    }

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
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);

	// initialize necessary OpenGL extensions
    glewInit();
    GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object");
    shrCheckErrorEX(bGLEW, shrTRUE, pCleanup);

    // default initialization
    glClearColor(0.0, 0.25, 0.25, 0.5);
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
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum  = clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_old_positions, 0, 0, 0 );
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum  = clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_colors, 0, 0, 0 );
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum  = clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_path_faithful, 0, 0, 0 );
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum  = clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_gravitational_force, 0, 0, 0 );
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum  = clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_velocity, 0, 0, 0 );
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    if (no_attractions > 0)
    {
        ciErrNum  = clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl_attraction_map, 0, 0, 0 );
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
#endif
    size_t szGlobalWorkSize[] = {(size_t) no_points, 1};
    size_t szGlobalWorkSizeDouble[] = {(size_t) no_points, 2};
    size_t szGlobalWorkSizeAttraction[] = {(size_t) no_attractions, 1};
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_compute_velocity, 1, NULL, szGlobalWorkSize, NULL, 0, 0, 0 );
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_move_to_target_path_faithful, 1, NULL, szGlobalWorkSize, NULL, 0, 0, 0 );
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    if (no_attractions > 0)
    {
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_attraction, 1, NULL, szGlobalWorkSizeAttraction, NULL, 0, 0, 0 );
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
#ifdef GL_INTEROP
    // unmap buffer object
    ciErrNum  = clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_positions, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_old_positions, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_velocity, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_path_faithful, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_gravitational_force, 0, 0, 0 );
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_colors, 0, 0, 0 );
    if (no_attractions > 0)
    {
        ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl_attraction_map, 0, 0, 0 );
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
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

int time_increment = 0;
int time_sum = 0;
// Display callback
//*****************************************************************************
void DisplayGL()
{
    gettimeofday(&start, NULL);

    // run OpenCL kernel to generate vertex positions
    runKernel();
    time_increment++;

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
    glDrawArrays(GL_POINTS, 0, no_points);

    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_obstacle_positions);
    glVertexPointer(2, GL_FLOAT, 0, 0);

    glEnableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_obstacle_colors);
    glColorPointer(4, GL_FLOAT, 0, 0);

    glLineWidth(5);
    glDrawArrays(GL_LINES, 0, no_obstacles);

//    glColor3f(0.0f,0.0f,1.0f); //blue color

//    glBegin(GL_POLYGON);//begin drawing of polygon
//      glVertex3f(-0.5f,0.5f,0.0f);//first vertex
//      glVertex3f(0.5f,0.5f,0.0f);//second vertex
//      glVertex3f(1.0f,0.0f,0.0f);//third vertex
//      glVertex3f(0.5f,-0.5f,0.0f);//fourth vertex
//      glVertex3f(-0.5f,-0.5f,0.0f);//fifth vertex
//      glVertex3f(-1.0f,0.0f,0.0f);//sixth vertex
//    glEnd();//end drawing of polygon

    glDisableClientState(GL_VERTEX_ARRAY);

    // flip backbuffer to screen
    glutSwapBuffers();

    gettimeofday(&stop, NULL);

    time_sum += stop.tv_usec - start.tv_usec;

    if (time_increment == 100)
    {
        printf("took %lu\n", time_sum / 100);
        time_increment = 0;
        time_sum  = 0;
    }
}

float m_position_x = -0.8;
float m_position_y = -0.8;
float m_increment_position_x = 0.035;
float m_increment_position_y = 0.035;

void init_world()
{
    std::string line;
    std::ifstream myfile("world.ads");

    if (myfile.is_open())
    {
        char *line_char;
        char *pointer;
        int no_points_index = -1;
        int no_obstacles_index = -1;
        int no_old_points_index = -1;
        int no_attractions_index = -1;
        int no_colors_index = -1;
        int no_obstacles_colors_index = -1;
        int no_target_points_index = -1;

        getline(myfile, line);

        if (line.compare(0, 7, "no_dots") == 0)
        {
            line_char = strdup(line.c_str());
            strtok_r(line_char, ":", &pointer);
            char *no_points_char = strtok_r(NULL, ":", &pointer);
            no_points = atoi(no_points_char);

            points_position = new GLfloat [2 * no_points];
            points_old_position = new GLfloat [2 * no_points];
            points_color = new GLfloat [4 * no_points];
            path_faithful = new GLfloat [no_points];
            gravitational_force = new GLfloat [no_points];

            points_velocity = new GLfloat [2 * no_points];

            for (int i=0; i < 2 * no_points; i++)
            {
                points_velocity[i] =  0.0f;
            }

            printf("no points: %d\n", no_points);
        }

        getline(myfile, line);

        if (line.compare(0, 14, "no_attractions") == 0)
        {
            line_char = strdup(line.c_str());
            strtok_r(line_char, ":", &pointer);
            char *no_attractions_char = strtok_r(NULL, ":", &pointer);
            no_attractions = atoi(no_attractions_char);

            if (no_attractions > 0)
            {
                attraction_map = new GLint [2 * no_attractions];
            }

            printf("no attr: %d\n", no_attractions);
        }

        getline(myfile, line);

        if (line.compare(0, 12, "no_obstacles") == 0)
        {
            line_char = strdup(line.c_str());
            strtok_r(line_char, ":", &pointer);
            char *no_obstacles_char = strtok_r(NULL, ":", &pointer);
            no_obstacles = atoi(no_obstacles_char);

            if (no_obstacles > 0)
            {
                obstacle_positions = new GLfloat [2 * no_obstacles];
                obstacle_colors = new GLfloat [4 * no_obstacles];
            }

            printf("no obst: %d\n", no_obstacles);
        }

        for (int i=0; i < no_points + no_obstacles; i++)
        {
            getline(myfile, line);

            printf("check: %d\n", i);

            if (line.compare(0, 3, "dot") == 0)
            {
                line_char = strdup(line.c_str());
                char *info = strtok_r(line_char, "| ", &pointer);

                while (info != NULL)
                {
                    if (strstr(info, "old_pos"))
                    {
                        char *old_position_x_char = strtok_r(NULL, "| ", &pointer);
                        char *old_position_y_char = strtok_r(NULL, "| ", &pointer);

                        float old_position_x = atof(old_position_x_char);
                        float old_position_y = atof(old_position_y_char);

                        points_old_position[++no_old_points_index] = old_position_x;
                        points_old_position[++no_old_points_index] = old_position_y;
                    }
                    else if (strstr(info, "position"))
                    {
                        char *position_x_char = strtok_r(NULL, "| ", &pointer);
                        char *position_y_char = strtok_r(NULL, "| ", &pointer);

                        float position_x = atof(position_x_char);
                        float position_y = atof(position_y_char);

                        points_position[++no_points_index] = position_x;
                        points_position[++no_points_index] = position_y;
                    }
                    else if (strstr(info, "color"))
                    {
                        char *color_x_char = strtok_r(NULL, "| ", &pointer);
                        char *color_y_char = strtok_r(NULL, "| ", &pointer);
                        char *color_z_char = strtok_r(NULL, "| ", &pointer);
                        char *color_w_char = strtok_r(NULL, "| ", &pointer);

                        float color_x = atof(color_x_char);
                        float color_y = atof(color_y_char);
                        float color_z = atof(color_z_char);
                        float color_w = atof(color_w_char);

                        points_color[++no_colors_index] = color_x;
                        points_color[++no_colors_index] = color_y;
                        points_color[++no_colors_index] = color_z;
                        points_color[++no_colors_index] = color_w;
                    }
                    else if (strstr(info, "path_faithful"))
                    {
                        char *path_faithful_char = strtok_r(NULL, "| ", &pointer);

                        path_faithful[no_points_index / 2] = atoi(path_faithful_char);
                    }
                    else if (strstr(info, "gravitation"))
                    {
                        char *gravitational_force_char = strtok_r(NULL, "| ", &pointer);

                        gravitational_force[no_points_index / 2] = atoi(gravitational_force_char);
                    }
                    else if (strstr(info, "attracted_by"))
                    {
                        char *attracted_by_char = strtok_r(NULL, "| ", &pointer);

                        attraction_map[++no_attractions_index] = no_points_index / 2;
                        attraction_map[++no_attractions_index] = atoi(attracted_by_char);
                    }

                    info = strtok_r(NULL, "| ", &pointer);
                }
            }
            else if (line.compare(0, 8, "obstacle") == 0)
            {
                line_char = strdup(line.c_str());
                char *info = strtok_r(line_char, "| ", &pointer);

                while (info != NULL)
                {
                    if (strstr(info, "position"))
                    {
                        char *obstacle_position_x_char = strtok_r(NULL, "| ", &pointer);
                        char *obstacle_position_y_char = strtok_r(NULL, "| ", &pointer);

                        float obstacle_position_x = atof(obstacle_position_x_char);
                        float obstacle_position_y = atof(obstacle_position_y_char);

                        obstacle_positions[++no_obstacles_index] = obstacle_position_x;
                        obstacle_positions[++no_obstacles_index] = obstacle_position_y;
                    }
                    else if (strstr(info, "color"))
                    {
                        char *obstacle_color_x_char = strtok_r(NULL, " ", &pointer);
                        char *obstacle_color_y_char = strtok_r(NULL, " ", &pointer);
                        char *obstacle_color_z_char = strtok_r(NULL, " ", &pointer);
                        char *obstacle_color_w_char = strtok_r(NULL, " ", &pointer);

                        no_obstacles_colors_index += 1;
                        obstacle_colors[no_obstacles_colors_index] = atof(obstacle_color_x_char);
                        no_obstacles_colors_index += 1;
                        obstacle_colors[no_obstacles_colors_index] = atof(obstacle_color_y_char);
                        no_obstacles_colors_index += 1;
                        obstacle_colors[no_obstacles_colors_index] = atof(obstacle_color_z_char);
                        no_obstacles_colors_index += 1;
                        obstacle_colors[no_obstacles_colors_index] = atof(obstacle_color_w_char);
                    }

                    info = strtok_r(NULL, "| ", &pointer);
                }
            }
        }

        printf("DONE\n");
    }

    printf("POSITONS\n");
    for (int i=0; i<no_obstacles; i++)
    {
        printf("%f %f\n", obstacle_positions[2 * i], obstacle_positions[2 * i + 1]);
    }

//    int no_points_index = 1;
//    int no_old_points_index = 1;
//
//    for (int i=2; i<no_points; i++)
//    {
//        points_position[++no_points_index] = m_position_x;
//        points_position[++no_points_index] = m_position_y;
//
//        points_old_position[++no_old_points_index] = m_position_x;
//        points_old_position[++no_old_points_index] = m_position_y;
//
//        m_position_x += m_increment_position_x;
//
//        printf("%f  ", m_position_x);
//
//        if (fabs(m_position_x) > 0.85)
//        {
//            m_increment_position_x *= -1;
//            m_position_y += m_increment_position_y;
//        }
//        if (fabs(m_position_y) > 0.85)
//        {
//            m_increment_position_y *= -1;
//        }
//    }
}

void init_pieces()
{
    float start_x = -1.0f;
    float start_y = 1.0f;

    float step = 0.5;

    for (int i=0; i<16; i++)
    {
        pieces_coordinates_x[i * 4 + 0] = start_x;
        pieces_coordinates_y[i * 4 + 0] = start_y;

        // up right
        pieces_coordinates_x[i * 4 + 1] = start_x + step;
        pieces_coordinates_y[i * 4 + 1] = start_y;

        // down left
        pieces_coordinates_x[i * 4 + 2] = start_x;
        pieces_coordinates_y[i * 4 + 2] = start_y - step;

        // down right
        pieces_coordinates_x[i * 4 + 3] = start_x + step;
        pieces_coordinates_y[i * 4 + 3] = start_y - step;

        start_x += step;

        if (start_x == 1.0f)
        {
            start_y -= step;
            start_x = -1.0f;
        }
    }

//    printf("\nPIECES COORDINATES\n");
//
//    for (int i=0; i<16; i++)
//    {
//        printf("%f %f %f %f\n", pieces_coordinates_x[i*4+0], pieces_coordinates_x[i*4+1], pieces_coordinates_x[i*4+2], pieces_coordinates_x[i*4+3]);
//        printf("%f %f %f %f\n", pieces_coordinates_y[i*4+0], pieces_coordinates_y[i*4+1], pieces_coordinates_y[i*4+2], pieces_coordinates_y[i*4+3]);
//    }
}

void timerEvent(int value)
{
    glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

// Create VBO
//*****************************************************************************
void createVBOObstaclePositions(GLuint* vbo)
{
    // create VBO
    unsigned int size = no_obstacles * 2 * sizeof(GLfloat);

    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, obstacle_positions, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_obstacle_positions = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_obstacle_positions = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_obstacle_positions = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
}

void createVBOObstacleColors(GLuint* vbo)
{
    // create VBO
    unsigned int size = no_obstacles * 4 * sizeof(GLfloat);

    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, obstacle_colors, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_obstacle_colors = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_obstacle_colors = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_obstacle_colors = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
}

void createVBOPositions(GLuint* vbo)
{
    // create VBO
    unsigned int size = no_points * 2 * sizeof(GLfloat);

    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, points_position, GL_DYNAMIC_DRAW);

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

void createVBOOldPositions(GLuint* vbo)
{
    // create VBO
    unsigned int size = no_points * 2 * sizeof(GLfloat);

    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, points_old_position, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_old_positions = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_old_positions = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_old_positions = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
}

void createVBOVelocity(GLuint* vbo)
{
    // create VBO
    unsigned int size = no_points * 2 * sizeof(GLfloat);

    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, points_velocity, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_velocity = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_velocity = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_velocity = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
}

void createVBOPathFaithful(GLuint* vbo)
{
    // create VBO
    unsigned int size = no_points * sizeof(GLfloat);

    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, path_faithful, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_path_faithful = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_path_faithful = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_path_faithful = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
}

void createVBOAttractionMap(GLuint* vbo)
{
    unsigned int size = 2 * no_attractions * sizeof(GLint);

    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, attraction_map, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_attraction_map = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_attraction_map = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_attraction_map = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
}

void createVBOColors(GLuint* vbo)
{
    // create VBO
    unsigned int size = no_points * 4 * sizeof(GLfloat);
    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, points_color, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_colors = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
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

void createVBOGravitationalForce(GLuint* vbo)
{
    // create VBO
    unsigned int size = no_points * sizeof(GLfloat);

    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, gravitational_force, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl_gravitational_force = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl_gravitational_force = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        // create standard OpenCL mem buffer
        vbo_cl_gravitational_force = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
}
// Function to clean up and exit
//*****************************************************************************
void Cleanup(int iExitCode)
{
    // Cleanup allocated objects
    shrLog("\nStarting Cleanup...\n\n");
    if(ckKernel_move_to_target_path_faithful)       clReleaseKernel(ckKernel_move_to_target_path_faithful);
    if(ckKernel_create_collision_map)       clReleaseKernel(ckKernel_create_collision_map);
    if(ckKernel_clean_collision_map)       clReleaseKernel(ckKernel_clean_collision_map);
    if(ckKernel_compute_velocity)       clReleaseKernel(ckKernel_compute_velocity);

    if(cpProgram)      clReleaseProgram(cpProgram);
    if(cqCommandQueue) clReleaseCommandQueue(cqCommandQueue);

    if(vbo_obstacle_positions)
    {
        glBindBuffer(1, vbo_obstacle_positions);
        glDeleteBuffers(1, &vbo_obstacle_positions);
        vbo_obstacle_positions = 0;
    }
    if(vbo_cl_obstacle_positions)clReleaseMemObject(vbo_cl_obstacle_positions);

    if(vbo_obstacle_colors)
    {
        glBindBuffer(1, vbo_obstacle_colors);
        glDeleteBuffers(1, &vbo_obstacle_colors);
        vbo_obstacle_colors = 0;
    }
    if(vbo_cl_obstacle_colors)clReleaseMemObject(vbo_cl_obstacle_colors);

    if(vbo_positions)
    {
        glBindBuffer(1, vbo_positions);
        glDeleteBuffers(1, &vbo_positions);
        vbo_positions = 0;
    }
    if(vbo_cl_positions)clReleaseMemObject(vbo_cl_positions);

    if(vbo_old_positions)
    {
        glBindBuffer(1, vbo_old_positions);
        glDeleteBuffers(1, &vbo_old_positions);
        vbo_old_positions = 0;
    }
    if(vbo_cl_old_positions)clReleaseMemObject(vbo_cl_old_positions);

    if(vbo_velocity)
    {
        glBindBuffer(1, vbo_velocity);
        glDeleteBuffers(1, &vbo_velocity);
        vbo_velocity = 0;
    }
    if(vbo_cl_velocity)clReleaseMemObject(vbo_cl_velocity);

    if(vbo_colors)
    {
        glBindBuffer(1, vbo_colors);
        glDeleteBuffers(1, &vbo_colors);
        vbo_colors = 0;
    }
    if(vbo_cl_colors)clReleaseMemObject(vbo_cl_colors);

    if(vbo_path_faithful)
    {
        glBindBuffer(1, vbo_path_faithful);
        glDeleteBuffers(1, &vbo_path_faithful);
        vbo_path_faithful = 0;
    }
    if(vbo_cl_path_faithful)clReleaseMemObject(vbo_cl_path_faithful);

    if(vbo_gravitational_force)
    {
        glBindBuffer(1, vbo_gravitational_force);
        glDeleteBuffers(1, &vbo_gravitational_force);
        vbo_gravitational_force = 0;
    }
    if(vbo_cl_gravitational_force)clReleaseMemObject(vbo_cl_gravitational_force);

    if(vbo_attraction_map)
    {
        glBindBuffer(1, vbo_attraction_map);
        glDeleteBuffers(1, &vbo_attraction_map);
        vbo_attraction_map = 0;
    }
    if(vbo_cl_attraction_map)clReleaseMemObject(vbo_cl_attraction_map);

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
