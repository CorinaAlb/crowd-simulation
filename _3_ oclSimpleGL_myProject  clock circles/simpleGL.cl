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

#define POINTS_ON_CIRCLE    12

#define RADIUS_XY_INC       0.001f
#define RADIUS_X_INC        0.001f
#define RADIUS_Y_INC        0.001f

__kernel void change_circle_color(__global float2* radius_xy, __global float2* points_on_circle, __global float4* colors)
{
    unsigned int gid = get_global_id(0);

    for (int i=0; i < POINTS_ON_CIRCLE; i++)
    {
        colors[i].x = (fabs(points_on_circle[19].x) < fabs(points_on_circle[i].x)) * 1.0f;
        colors[i].y = (fabs(points_on_circle[19].x) < fabs(points_on_circle[i].x)) * 0.0f;
        colors[i].z = (fabs(points_on_circle[19].x) < fabs(points_on_circle[i].x)) * 0.0f;
        colors[i].w = 1.0f;

        colors[24 + i].x = (fabs(points_on_circle[13].x) < fabs(points_on_circle[24 + i].x)) * 0.0f;
        colors[24 + i].y = (fabs(points_on_circle[13].x) < fabs(points_on_circle[24 + i].x)) * 1.0f;
        colors[24 + i].z = (fabs(points_on_circle[13].x) < fabs(points_on_circle[24 + i].x)) * 1.0f;
        colors[24 + i].w = 1.0f;
    }
}

__kernel void draw_circle(__global float2* pos, __global float2* radius, __global float2* points_on_circle)
{
    unsigned int gid = get_global_id(0);

    for (int i=0; i < POINTS_ON_CIRCLE; i++)
    {
        float theta = 2.0f * 3.1415926f * i / POINTS_ON_CIRCLE;//get the current angle

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

// center_mask is used to control which circles modify their radius
__kernel void increment_radius(__global float2* radius_xy, __global float2* center_mask)
{
    unsigned int gid = get_global_id(0);

    radius_xy[gid].x = radius_xy[gid].x + (fabs(radius_xy[gid].x)  < 1) * RADIUS_XY_INC * center_mask[gid].x;
    radius_xy[gid].y = radius_xy[gid].y + (fabs(radius_xy[gid].y)  < 1) * RADIUS_XY_INC * center_mask[gid].y;
}

__kernel void decrement_radius(__global float2* radius_xy, __global float2* center_mask)
{
    unsigned int gid = get_global_id(0);

    radius_xy[gid].x = radius_xy[gid].x - (fabs(radius_xy[gid].x)  < 1) * RADIUS_XY_INC * center_mask[gid].x;
    radius_xy[gid].y = radius_xy[gid].y - (fabs(radius_xy[gid].y)  < 1) * RADIUS_XY_INC * center_mask[gid].y;
}

//__kernel void points_on_circle(__global float2* radius_xy, __global int* no_points_on_circle)
//{
//    unsigned int gid = get_global_id(0);
//
//    no_points_on_circle[gid] = (radius_xy[gid].x < 0.25 || radius_xy[gid].y < 0.25) * 1/2 * POINTS_ON_CIRCLE +
//        ((radius_xy[gid].x > 0.5 && radius_xy[gid].x < 0.75) || (radius_xy[gid].y > 0.5 && radius_xy[gid].x < 0.75 )) * 2 * POINTS_ON_CIRCLE +
//        (radius_xy[gid].x > 0.75 || radius_xy[gid].y > 0.75) * 3 * POINTS_ON_CIRCLE;
//}
