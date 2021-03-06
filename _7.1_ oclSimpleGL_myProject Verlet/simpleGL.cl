// ! entities are POINTS

#define NO_POINTS                   1
#define LIMIT_PROXIMITY             0.1
#define MAX_POINTS_ON_LIMIT         6
#define BACK_OFF                    0.0005
//#define VELOCITY                    0.5
#define BOUNCING_SPEED_MODIFIER     0.95
#define GRAVITATIONAL_FORCE         0.0005
#define FRICTION_FORCE              0.9999

// !!! redundant computations
//__kernel void simple_example(__global float2* pos)
// use the collision map to avoid redundant computations
//__kernel void simple_example(__global float2* pos, __global float2* collision_map, __global float2* moving_direction)
//{
//    unsigned int gid = get_global_id(0);
//
//    float2 point_position = pos[gid];
//
////    for (int i = 0; i < MAX_POINTS_ON_LIMIT; i++)
////    {
////
////    }
//
////    int move_point = 0;
//
////    for (int i = 0; i < NO_POINTS; i++)
////    {
////        float2 dist = pos[i] - point_position;
////        int if_cond = (gid != i) && ((dist.x * dist.x + dist.y * dist.y) > LIMIT_PROXIMITY * LIMIT_PROXIMITY);
////
////        move_point = move_point + if_cond;
////    }
//
//    float sign_x = 1 - pos[gid].x - 0.05;
//    float sign_y = 1 - pos[gid].y - 0.05;
//
//    int move_condition = fabs(pos[gid].x) < 0.9;
//
//    pos[gid].x = pos[gid].x + 0.005 * moving_direction[gid].x * move_condition;
//    pos[gid].y = pos[gid].y + 0.005 * moving_direction[gid].y * move_condition;
//}

// 1.1 both the obstacle & the moving entity change their position in order to avoid collision
// !! NOT WORKING LOGICALLY (physically correct) !!
__kernel void move_to_target(__global float2* pos, __global float2* collision_map, __global float2* target)
{
    unsigned int gid = get_global_id(0);

    float sign_x = target[gid].x - pos[gid].x;
    float sign_y = target[gid].y - pos[gid].y;

    int obstacle_exists = 0;
    float2 back_off = (float2) (0.0f, 0.0f);

    for (int i = 0; i < MAX_POINTS_ON_LIMIT; i++)
    {
        obstacle_exists += (collision_map[gid * MAX_POINTS_ON_LIMIT + i].x != 0.0f) || (collision_map[gid * MAX_POINTS_ON_LIMIT + i].y != 0.0f);

        int obstacle_x_sign = (pos[gid].x - collision_map[gid * MAX_POINTS_ON_LIMIT + i].x) >= 0;
        int obstacle_y_sign = (pos[gid].y - collision_map[gid * MAX_POINTS_ON_LIMIT + i].y) >= 0;

        back_off.x = back_off.x + (obstacle_x_sign == 1) * BACK_OFF - (obstacle_x_sign != 1) * BACK_OFF;
        back_off.y = back_off.y + (obstacle_y_sign == 1) * BACK_OFF - (obstacle_y_sign != 1) * BACK_OFF;
    }

//    int move_point = 0;

//    for (int i = 0; i < NO_POINTS; i++)
//    {
//        float2 dist = pos[i] - point_position;
//        int if_cond = (gid != i) && ((dist.x * dist.x + dist.y * dist.y) > LIMIT_PROXIMITY * LIMIT_PROXIMITY);

//        move_point = move_point + if_cond;
//    }

//    pos[gid].x = pos[gid].x + VELOCITY * sign_x * (obstacle_exists == 0) + back_off.x * (obstacle_exists != 0);
//    pos[gid].y = pos[gid].y + VELOCITY * sign_y * (obstacle_exists == 0) + back_off.y * (obstacle_exists != 0);
}

// 2. the obstacle changes position. the entity that is moving does not change it's route
__kernel void move_to_target_path_faithful(__global float2* pos, __global float2* collision_map,
                                        __global float2* target, __global int* path_faithful, __global float2* velocity)
{
    unsigned int gid = get_global_id(0);

//    float sign_x = target[gid].x - pos[gid].x;
//    float sign_y = target[gid].y - pos[gid].y;
//
//    int obstacle_exists = 0;
//    float2 back_off = (float2) (0.0f, 0.0f);
//
//    for (int i = 0; i < MAX_POINTS_ON_LIMIT; i++)
//    {
//        obstacle_exists += (collision_map[gid * MAX_POINTS_ON_LIMIT + i].x != 0.0f) || (collision_map[gid * MAX_POINTS_ON_LIMIT + i].y != 0.0f);
//
//        int obstacle_x_sign = (pos[gid].x - collision_map[gid * MAX_POINTS_ON_LIMIT + i].x) >= 0;
//        int obstacle_y_sign = (pos[gid].y - collision_map[gid * MAX_POINTS_ON_LIMIT + i].y) >= 0;
//
//        back_off.x = back_off.x + (obstacle_x_sign == 1) * BACK_OFF - (obstacle_x_sign != 1) * BACK_OFF;
//        back_off.y = back_off.y + (obstacle_y_sign == 1) * BACK_OFF - (obstacle_y_sign != 1) * BACK_OFF;
//    }

//    int move_point = 0;

//    for (int i = 0; i < NO_POINTS; i++)
//    {
//        float2 dist = pos[i] - point_position;
//        int if_cond = (gid != i) && ((dist.x * dist.x + dist.y * dist.y) > LIMIT_PROXIMITY * LIMIT_PROXIMITY);

//        move_point = move_point + if_cond;
//    }
    int x_outside = fabs(pos[gid].x) >= 0.8f;
    int y_outside = fabs(pos[gid].y) >= 0.8f;
    int x_inside = !x_outside;
    int y_inside = !y_outside;

    pos[gid].x += velocity[gid].x * (x_inside - x_outside);// * (sign_x != 0)* (obstacle_exists == 0 || path_faithful[gid] == 1) + back_off.x * (obstacle_exists != 0 && path_faithful[gid] == 0);
    pos[gid].y += velocity[gid].y * (y_inside - y_outside);// * sign_y * (obstacle_exists == 0 || path_faithful[gid] == 1) + back_off.y * (obstacle_exists != 0 && path_faithful[gid] == 0);
    pos[gid].y -= GRAVITATIONAL_FORCE * (velocity[gid].y != 0.0f);
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

    velocity[gid].y -= velocity[gid].y * (position[gid].y < -0.855f);
}

// avoid redundant computations
__kernel void collision_detection(__global float2* pos, __global float2* collision_map)
{
    unsigned int gid = get_global_id(0);

    float2 point_position = pos[gid];

    int points_on_limit = 0;

    for (int i = 0; i < NO_POINTS; i++)
    {
        float2 dist = pos[i] - point_position;
        int if_cond = (gid != i) && ((dist.x * dist.x + dist.y * dist.y) <= LIMIT_PROXIMITY * LIMIT_PROXIMITY);

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
