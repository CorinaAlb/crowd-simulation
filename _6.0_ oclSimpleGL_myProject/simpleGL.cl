// ! entities are POINTS

#define NO_POINTS                   12
#define DISTANCE                    0.1
#define MAX_POINTS_ON_LIMIT          6

// !!! redundant computations
__kernel void simple_example(__global float2* pos)
{
    unsigned int gid = get_global_id(0);

    float2 point_position = pos[gid];
    int move_point = 0;

    for (int i = 0; i < NO_POINTS; i++)
    {
        float2 dist = pos[i] - point_position;
        int if_cond = (gid != i) && ((dist.x * dist.x + dist.y * dist.y) > DISTANCE * DISTANCE);

        move_point = move_point + if_cond;
    }

    float sign_x = 1 - pos[gid].x - 0.05;
    float sign_y = 1 - pos[gid].y - 0.05;

    pos[gid].x = pos[gid].x + 0.005 * sign_x * (move_point == NO_POINTS - 1);
    pos[gid].y = pos[gid].y + 0.005 * sign_y * (move_point != NO_POINTS - 1);
}

__kernel void collision_detection(__global float2* pos, __global float2* collision_map)
{
    unsigned int gid = get_global_id(0);

    float2 point_position = pos[gid];

    int points_on_limit = 0;

    for (int i = 0; i < NO_POINTS; i++)
    {
        float2 dist = pos[i] - point_position;
        int if_cond = (gid != i) && ((dist.x * dist.x + dist.y * dist.y) < DISTANCE * DISTANCE);

        points_on_limit += if_cond;

        collision_map[gid * MAX_POINTS_ON_LIMIT + points_on_limit].x += pos[i].x * if_cond;
        collision_map[gid * MAX_POINTS_ON_LIMIT + points_on_limit].y += pos[i].y * if_cond;
    }
}

__kernel void clean_collision_detection(__global float2* collision_map)
{
    unsigned int gid = get_global_id(0);

    for (int i=0; i < MAX_POINTS_ON_LIMIT; i++)
    {
        collision_map[gid * MAX_POINTS_ON_LIMIT + i] = (float2) ( 0.0f, 0.0f );
    }
}
