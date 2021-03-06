// ! entities are POINTS

#define NO_POINTS                   11
#define LIMIT_PROXIMITY             0.04
#define MAX_POINTS_ON_LIMIT         6
#define BACK_OFF                    0.005
#define BOUNCING_SPEED_MODIFIER     0.95
#define GRAVITATIONAL_FORCE         0.0005
#define FRICTION_FORCE              0.9999
#define ATTRACTION_FORCE            0.005

// 2. the obstacle changes position. the entity that is moving does not change it's route
__kernel void move_to_target_path_faithful(__global float2* pos, __global float* path_faithful,
                            __global float2* velocity, __global float* gravitational_influence)
{
    unsigned int gid = get_global_id(0);

    float2 current_point = (float2) pos[gid];

    int obstacle_exists = 0;

    float2 back_off = (float2) (0.0f, 0.0f);

    int x_outside = fabs(pos[gid].x) >= 0.8f;
    int y_outside = fabs(pos[gid].y) >= 0.8f;
    int x_inside = !x_outside;
    int y_inside = !y_outside;

    for (int i = 0; i < NO_POINTS; i++)
    {
        float2 point = (float2) pos[i];
        int influenced = (i != gid) && (fabs(current_point.x - point.x) <= LIMIT_PROXIMITY) && (fabs(current_point.y - point.y) <= LIMIT_PROXIMITY);

        obstacle_exists += influenced;

        int obstacle_x_sign = ( pos[gid].x - pos[i].x ) > 0.0f;
        int obstacle_y_sign = ( pos[gid].y - pos[i].y ) > 0.0f;

        // obstacle_sign = 0 => - sign
        // obstacle_sign > 0 => + sign

        back_off.x += influenced * ((obstacle_x_sign == 1) - (obstacle_x_sign != 1)) * BACK_OFF;
        back_off.y += influenced * ((obstacle_y_sign == 1) - (obstacle_y_sign != 1)) * BACK_OFF;
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

__kernel void attraction(__global float2* pos, __global int2* attraction_influence)
{
    unsigned int gid = get_global_id(0);

    int influenced_point_index = attraction_influence[gid].x;
    int atracted_by_index = attraction_influence[gid].y;

    float sign_x = (pos[atracted_by_index].x - pos[influenced_point_index].x) > 0.0f; // == 1 -> positive value
    float sign_y = (pos[atracted_by_index].y - pos[influenced_point_index].y) > 0.0f; // == 1 -> positive value

    pos[influenced_point_index].x += ( (sign_x == 1) - (sign_x != 1) ) * ATTRACTION_FORCE;
    pos[influenced_point_index].y += ( (sign_y == 1) - (sign_y != 1) ) * ATTRACTION_FORCE;
}
