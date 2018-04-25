#define POINTS_ON_CIRCLE    360
#define NO_CIRCLES          10

__kernel void circle_expansion_color_change(__global float2* radius_xy, __global float2* points_on_circle,
                                  __global float4* colors, __global float4* colors_mask,
                                  __global float2* pos)
{
    unsigned int gid = get_global_id(0);

    for (int j=0; j < NO_CIRCLES; j++)
    {
        float2 center_position = pos[j];
        float2 radius = radius_xy[j];

        for (int i = 0; i < POINTS_ON_CIRCLE; i++)
        {
            float2 dist = (points_on_circle[gid * POINTS_ON_CIRCLE + i] - center_position);
            int if_cond = (gid != j) && ((dist.x * dist.x + dist.y * dist.y) < radius.x * radius.y);

            colors[gid * POINTS_ON_CIRCLE + i].x = if_cond * colors_mask[gid].x;
            colors[gid * POINTS_ON_CIRCLE + i].y = if_cond * colors_mask[gid].y;
            colors[gid * POINTS_ON_CIRCLE + i].z = if_cond * colors_mask[gid].z;
            colors[gid * POINTS_ON_CIRCLE + i].w = colors_mask[gid].w;
        }
    }
}

__kernel void analize_circle_color(__global float2* radius_xy, __global float2* points_on_circle,
                                  __global float4* colors, __global float4* colors_mask,
                                  __global float2* pos)
                                  //, __global float2* pos_mask)
{
    unsigned int gid = get_global_id(0);

    for (int j=0; j < NO_CIRCLES; j++)
    {
        //int iteration_condition = (pos_mask[j].x == 1);

        float2 center_position = pos[j];
        float2 radius = radius_xy[j];

        //for (int i = 0; i < POINTS_ON_CIRCLE * iteration_condition; i++)
        for (int i = 0; i < POINTS_ON_CIRCLE; i++)
        {
            float2 dist = (points_on_circle[gid * POINTS_ON_CIRCLE + i] - center_position);
            int if_cond = (gid != j) && ((dist.x * dist.x + dist.y * dist.y) < radius.x * radius.y);

            colors[gid * POINTS_ON_CIRCLE + i].x += if_cond * colors_mask[gid].x;
            colors[gid * POINTS_ON_CIRCLE + i].y += if_cond * colors_mask[gid].y;
            colors[gid * POINTS_ON_CIRCLE + i].z += if_cond * colors_mask[gid].z;
            colors[gid * POINTS_ON_CIRCLE + i].w = colors_mask[gid].w;
        }
    }
}

__kernel void draw_all_circles(__global float2* pos, __global float2* radius, __global float2* points_on_circle)
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

    float sign_x = 1 - pos[gid].x - 0.05;
    float sign_y = 1 - pos[gid].y - 0.05;

    pos[gid].x = pos[gid].x + 0.01 * sign_x;
    pos[gid].y = pos[gid].y + 0.01 * sign_y;
}

// center_mask is used to control which circles modify their radius
__kernel void modify_radius(__global float2* radius_xy, __global float2* speed_mask)
{
    unsigned int gid = get_global_id(0);

    radius_xy[gid].x = radius_xy[gid].x + (fabs(radius_xy[gid].x) < 1) * speed_mask[gid].x;
    radius_xy[gid].y = radius_xy[gid].y + (fabs(radius_xy[gid].y) < 1) * speed_mask[gid].y;

    speed_mask[gid].x *= (fabs(radius_xy[gid].x) > 0.9) * (-1) + (fabs(radius_xy[gid].x) <= 0.9);
    speed_mask[gid].y *= (fabs(radius_xy[gid].y) > 0.9) * (-1) + (fabs(radius_xy[gid].y) <= 0.9);
}

__kernel void change_speed_mask_sign(__global float2* speed_mask, int index)
{
    unsigned int gid = get_global_id(0);
    speed_mask[gid] *= -1;
}
