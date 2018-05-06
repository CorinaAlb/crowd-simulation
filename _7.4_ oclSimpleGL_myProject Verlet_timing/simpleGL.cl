// ! entities are POINTS

#define NO_POINTS                   100
#define LIMIT_PROXIMITY             0.02
#define MAX_POINTS_ON_LIMIT         6
#define BACK_OFF                    0.005
#define BOUNCING_SPEED_MODIFIER     0.95
#define GRAVITATIONAL_FORCE         0.0005
#define FRICTION_FORCE              0.9999

// 2. the obstacle changes position. the entity that is moving does not change it's route
__kernel void move_to_target_path_faithful(__global float2* pos, __global int2* collision_map,
                            __global float* path_faithful, __global float2* velocity, __global float* gravitational_influence)
{
//    unsigned int gid = get_global_id(0);
//
//    int x_outside = fabs(pos[gid].x) >= 0.8f;
//    int y_outside = fabs(pos[gid].y) >= 0.8f;
//    int x_inside = !x_outside;
//    int y_inside = !y_outside;
//
//    pos[gid].x += velocity[gid].x * (x_inside - x_outside);// * (sign_x != 0)* (obstacle_exists == 0 || path_faithful[gid] == 1) + back_off.x * (obstacle_exists != 0 && path_faithful[gid] == 0);
//    pos[gid].y += velocity[gid].y * (y_inside - y_outside);// * sign_y * (obstacle_exists == 0 || path_faithful[gid] == 1) + back_off.y * (obstacle_exists != 0 && path_faithful[gid] == 0);
//    pos[gid].y -= GRAVITATIONAL_FORCE * (velocity[gid].y != 0.0f);

    unsigned int gid = get_global_id(0);

    int obstacle_exists = 0;

    float2 back_off = (float2) (0.0f, 0.0f);

    int x_outside = fabs(pos[gid].x) >= 0.8f;
    int y_outside = fabs(pos[gid].y) >= 0.8f;
    int x_inside = !x_outside;
    int y_inside = !y_outside;

    for (int i = 0; i < MAX_POINTS_ON_LIMIT; i++)
    {
        obstacle_exists += (collision_map[gid * MAX_POINTS_ON_LIMIT + i].y != 0);

        int index = collision_map[gid * MAX_POINTS_ON_LIMIT + i].x;

        int obstacle_x_sign = ( pos[gid].x - pos[index].x ) > 0.0f;
        int obstacle_y_sign = ( pos[gid].y - pos[index].y ) > 0.0f;

        // obstacle_sign = 0 => - sign
        // obstacle_sign > 0 => + sign

        back_off.x += ((obstacle_x_sign == 1) - (obstacle_x_sign != 1)) * BACK_OFF;
        back_off.y += ((obstacle_y_sign == 1) - (obstacle_y_sign != 1)) * BACK_OFF;
    }

    pos[gid].x += (x_inside - x_outside) * ( velocity[gid].x * (path_faithful[gid] == 1) + (obstacle_exists > 0 && path_faithful[gid] == 0) * back_off.x );
    pos[gid].y += (y_inside - y_outside) * ( velocity[gid].y * (path_faithful[gid] == 1) + (obstacle_exists > 0 && path_faithful[gid] == 0) * back_off.y );
    pos[gid].y -= GRAVITATIONAL_FORCE * (fabs(pos[gid].y) < 0.855f) * (gravitational_influence[gid] == 1.0f);
}

__kernel void compute_velocity(__global float2* position, __global float2* old_position, __global float2* velocity)
{
    unsigned int gid = get_global_id(0);

    velocity[gid] = (position[gid] - old_position[gid]);// * (float2)(FRICTION_FORCE, FRICTION_FORCE);
    old_position[gid] = position[gid];

    int x_outside = fabs(position[gid].x) >= 0.8f;
    int y_outside = fabs(position[gid].y) >= 0.8f;

    int outside = x_outside || y_outside;

    velocity[gid].x *= (BOUNCING_SPEED_MODIFIER * (outside == 1) + !outside);
    velocity[gid].y *= (BOUNCING_SPEED_MODIFIER * (outside == 1) + !outside);
}

// avoid redundant computations
__kernel void collision_detection(__global float2* pos, __global int2* collision_map)
{
    unsigned int gid = get_global_id(0);
    int index = gid * MAX_POINTS_ON_LIMIT;

    float2 dist;
    float2 point_position = pos[gid];

    for (int i = 0; i < NO_POINTS; i++)
    {
        dist.x = pos[i].x - point_position.x;
        dist.y = pos[i].y - point_position.y;
        int if_cond = (gid != i) && ((dist.x * dist.x + dist.y * dist.y) < LIMIT_PROXIMITY * LIMIT_PROXIMITY) && (collision_map[index].y == 0);

        collision_map[index].x += gid * if_cond;
        collision_map[index].y += 1 * if_cond;

        index += 1 * if_cond;
    }
}


__kernel void clean_collision_detection(__global int2* collision_map)
{
    unsigned int gid = get_global_id(0);

    for (int i=0; i < MAX_POINTS_ON_LIMIT; i++)
    {
        collision_map[gid * MAX_POINTS_ON_LIMIT + i].x = 0;
        collision_map[gid * MAX_POINTS_ON_LIMIT + i].y = 0;
    }
}
