// ! entities are POINTS

#define NO_POINTS                   7
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

__kernel void map_obstacles_to_pieces(__global float4 pieces_coordinates_x, __global float4 pieces_coordinates_y,
                    __global float2 obstacle_positions, __global float2 starting_points_obstacles, __global float2 ending_points_obstacles, int no_obstacles)
{
    unsigned int gid = get_global_id(0);

    float up_left_x = fabs(pieces_coordinates_x[gid * 4 + 0]);
    float up_right_x = fabs(pieces_coordinates_x[gid * 4 + 1]);
    float down_left_x = fabs(pieces_coordinates_x[gid * 4 + 2]);
    float down_right_x = fabs(pieces_coordinates_x[gid * 4 + 3]);
    float up_left_y = fabs(pieces_coordinates_y[gid * 4 + 0]);
    float up_right_y = fabs(pieces_coordinates_y[gid * 4 + 1]);
    float down_left_y = fabs(pieces_coordinates_y[gid * 4 + 2]);
    float down_right_y = fabs(pieces_coordinates_y[gid * 4 + 3]);

    float max_x = pieces_coordinates_x[gid * 4 + 1];
    float min_x = pieces_coordinates_x[gid * 4 + 0];
    float max_y = pieces_coordinates_y[gid * 4 + 0];
    float min_y = pieces_coordinates_y[gid * 4 + 2];

    // !!!
    // all obstacles are horizontal or vertical  lines
    int obstacles_index = gid * 5;

    for (int i=0; i<no_obstacles; i++)
    {
        float fabs_start_obstacle_position_x = fabs(obstacle_positions[i * 2].x);
        float fabs_start_obstacle_position_y = fabs(obstacle_positions[i * 2].y);
        float fabs_end_obstacle_position_x = fabs(obstacle_positions[i * 2 + 1].x);
        float fabs_end_obstacle_position_y = fabs(obstacle_positions[i * 2 + 1].y);

        float start_obstacle_position_x = obstacle_positions[i * 2].x;
        float start_obstacle_position_y = obstacle_positions[i * 2].y;
        float end_obstacle_position_x = obstacle_positions[i * 2 + 1].x;
        float end_obstacle_position_y = obstacle_positions[i * 2 + 1].y;

        // MANHATTAN
        int start_point_inside_x = (start_obstacle_position_x > up_left_x) && (start_obstacle_position_x < up_right_x);
        int start_point_inside_y = (start_obstacle_position_y < up_left_y) && (start_obstacle_position_y > down_left_y);
        int end_point_inside_x = (end_obstacle_position_x > up_left_x) && (end_obstacle_position_x < up_right_x);
        int end_point_inside_y = (end_obstacle_position_y < up_left_y) && (end_obstacle_position_y > down_left_y);

        int start_point_inside = (start_point_inside_x == 1) && (start_point_inside_y == 1);
        int end_point_inside = (end_point_inside_x == 1) && (end_point_inside_y == 1);

        // 1. end point inside & start point inside
        int case_one = (start_point_inside == 1) && (end_point_inside == 1);

        // 2. end point outside & start point inside
        int case_two = (start_point_inside == 1) && (end_point_inside == 0);

        // 3. end point inside & start point outside
        int case_three = (start_point_inside == 0) && (end_point_inside == 1);

        // 4. end point outside & start point outside
        int case_four = (start_point_inside == 0) && (end_point_inside == 0);

        obstacles_index += 1;

        starting_points_obstacles[obstacles_index].x = (case_one == 1) * start_obstacle_position_x
            + (case_two == 1) * start_obstacle_position_x
            + (case_three == 1) * (max_x * (start_obstacle_position_x > max_x) + min_x * (start_obstacle_position_x < min_x))
            + (case_four == 1) * (min_x * (start_point_inside_x == 0 && end_point_inside_x == 0 && end_point_inside_y == 1 && start_point_inside_y == 1)
            + start_obstacle_position_x * (start_point_inside_x == 1 && end_point_inside_x == 1 && end_point_inside_y == 0 && start_point_inside_y == 0));

        starting_points_obstacles[obstacles_index].y = (case_one == 1) * start_obstacle_position_y
            + (case_two == 1) * start_obstacle_position_y
            + (case_three == 1) * (max_y * (start_obstacle_position_y > max_y) + min_y * (start_obstacle_position_y < min_y))
            + (case_four == 1) * (max_y * (start_point_inside_x == 1 && end_point_inside_x == 1 && end_point_inside_y == 0 && start_point_inside_y == 0)
            + start_obstacle_position_y * (start_point_inside_x == 0 && end_point_inside_x == 0 && end_point_inside_y == 1 && start_point_inside_y == 1));

        ending_points_obstacles[obstacles_index].x = (case_one == 1) * end_obstacle_position_x
            + (case_two == 1) * (max_x * (end_obstacle_position_x >= max_x) + min_x * (end_obstacle_position_x =< min_x))
            + (case_three == 1) * end_obstacle_position_x
            + (case_four == 1) * (max_x * (start_point_inside_x == 0 && end_point_inside_x == 0 && end_point_inside_y == 1 && start_point_inside_y == 1)
            + end_obstacle_position_x * (start_point_inside_x == 1 && end_point_inside_x == 1 && end_point_inside_y == 0 && start_point_inside_y == 0));

        ending_points_obstacles[obstacles_index].y = (case_one == 1) * end_obstacle_position_y
            + (case_two == 1) * (max_y * (end_obstacle_position_y >= max_y) + min_y * (end_obstacle_position_y =< min_y))
            + (case_three == 1) * end_obstacle_position_y
            + (case_four == 1) * (min_y * (start_point_inside_x == 1 && end_point_inside_x == 1 && end_point_inside_y == 0 && start_point_inside_y == 0)
            + end_obstacle_position_y * (start_point_inside_x == 0 && end_point_inside_x == 0 && end_point_inside_y == 1 && start_point_inside_y == 1));
    }
}
