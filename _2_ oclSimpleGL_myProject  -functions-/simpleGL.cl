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

/* This example demonstrates how to use the OpenCL/OpenGL bindings  */

#define POINTS_ON_CIRCLE    10

#define RADIUS_XY_INC       0.001f
#define RADIUS_X_INC        0.001f
#define RADIUS_Y_INC        0.001f

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__kernel void sine_wave(__global float4* pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sin(u*freq + time) * cos(v*freq + time) * 0.5f;

    // write output vertex
    pos[y*width+x] = (float4)(u, w, v, 1.0f);
}

__kernel void change_circle_color(float pos_x, float pos_y, __global float4* colors, __global float* mask)
{
    unsigned int gid = get_global_id(0);

    for (int i=0; i < POINTS_ON_CIRCLE; i++)
    {
        if (pos_x == pos_y)
        {
            colors[gid * POINTS_ON_CIRCLE + i].x = 00111111100000000000000000000000 && mask[gid * POINTS_ON_CIRCLE + i];
            colors[gid * POINTS_ON_CIRCLE + i].y = 00111111100000000000000000000000 && mask[gid * POINTS_ON_CIRCLE + i];
        }
    }
}

__kernel void draw_circle(__global float2* pos, __global float2* radius, __global float2* points_on_circle)
{
    unsigned int gid = get_global_id(0);

    for (int i=0; i < POINTS_ON_CIRCLE; i++)
    {
        float theta = 2.0f * 3.1415926f * i / 10;//get the current angle

        points_on_circle[gid * POINTS_ON_CIRCLE + i].x = pos[gid].x + radius[gid].x * cos(theta);
        points_on_circle[gid * POINTS_ON_CIRCLE + i].y = pos[gid].y + radius[gid].y * sin(theta);
    }
}

__kernel void move_circle_center(__global float2* pos)
{
    unsigned int gid = get_global_id(0);

//    sign_x[gid].x = (int) ((1 - pos[gid].x - 0.05) / (fabs(1 - pos[gid].x - 0.05))) * sign_x[gid].x;
//    sign_y[gid].y = (int) ((1 - pos[gid].y - 0.05) / (fabs(1 - pos[gid].y - 0.05))) * sign_x[gid].y;

//    pos[gid].x = pos[gid].x + 0.01 * sign_x[gid].x;
//    pos[gid].y = pos[gid].y + 0.01 * sign_y[gid].y;

    float sign_x = 1 - pos[gid].x - 0.05;
    float sign_y = 1 - pos[gid].y - 0.05;

    pos[gid].x = pos[gid].x + 0.01 * sign_x;
    pos[gid].y = pos[gid].y + 0.01 * sign_y;
}

__kernel void increment_radius(__global float2* radius_xy, __global float2* center_mask)
{
    unsigned int gid = get_global_id(0);

    radius_xy[gid].x = radius_xy[gid].x + RADIUS_XY_INC * center_mask[gid].x;
    radius_xy[gid].y = radius_xy[gid].y + RADIUS_XY_INC * center_mask[gid].y;
}

__kernel void decrement_radius(__global float2* radius_xy, __global float2* mask)
{
    unsigned int gid = get_global_id(0);

    radius_xy[gid].x = radius_xy[gid].x - RADIUS_XY_INC * mask[gid].x;
    radius_xy[gid].y = radius_xy[gid].y - RADIUS_XY_INC * mask[gid].y;
}
