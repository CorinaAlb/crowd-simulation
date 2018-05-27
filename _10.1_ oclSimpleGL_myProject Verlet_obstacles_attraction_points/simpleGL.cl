// ! entities are POINTS

#define BACK_OFF                    0.005
//#define BOUNCING_SPEED_MODIFIER     0.95
#define GRAVITATIONAL_FORCE         0.005
#define ATTRACTION_FORCE            0.005

__kernel void labirinth(__global float2* pos, __global float2* target,
                        __global int* matrix_x, __global int* matrix_y,
                        __global int* neighbours_x, __global int* neighbours_y,
                        __global float* lookahead_x, __global float* lookahead_y,
                        __global int* activated)
{
    unsigned int gid = get_global_id(0);

    float2 current_point = (float2) pos[gid];

    int point_x = (int) (current_point.x * 100 + 96 + .5);
    int point_y = (int) (current_point.y * 100 + 96 + .5);

    int point_x_in_matrix = 193 * point_y + point_x;
    int point_y_in_matrix = 193 * point_x + point_y;

//    float start_point_x = lookahead_x[2 * point_x_in_matrix];
//    float end_point_x   = lookahead_x[2 * point_x_in_matrix + 1];

    float start_point_y = lookahead_y[2 * point_y_in_matrix];
    float end_point_y   = lookahead_y[2 * point_y_in_matrix + 1];

    int activated_start_point = activated[2 * point_y_in_matrix];
    int activated_end_point   = activated[2 * point_y_in_matrix + 1];

    /* here, there are only y coordinates */
    float obstacle_attraction_start_y = fabs(start_point_y - current_point.y) * activated_start_point;
    float obstacle_attraction_end_y   = fabs(end_point_y - current_point.y) * activated_end_point;

    int sign_obstacle_attraction_up   = ((obstacle_attraction_start_y < obstacle_attraction_end_y) * activated_start_point * activated_end_point)
                                    || (activated_start_point != 0 && activated_end_point == 0);
    int sign_obstacle_attraction_down = ((obstacle_attraction_start_y >= obstacle_attraction_end_y) * activated_end_point * activated_start_point)
                                    || (activated_end_point != 0 && activated_start_point == 0);

    int down    = point_y_in_matrix - 1;
    int up      = point_y_in_matrix + 1;
    int left    = point_x_in_matrix - 1;
    int right   = point_x_in_matrix + 1;

    /**
     *   FOLLOW THE TARGET & AVOID OBSTACLES
     */
    float sign_x = (target[gid].x - pos[gid].x) > 0;
    float sign_y = (target[gid].y - pos[gid].y) > 0;

    int obstacle_up         = (matrix_x[point_x_in_matrix] == 1) && (matrix_y[up] == 1);
    int obstacle_up_left    = (matrix_x[left] == 1) && (matrix_y[up] == 1);
    int obstacle_up_right   = (matrix_x[right] == 1) && (matrix_y[up] == 1);
    int obstacle_left       = (matrix_x[left] == 1) && (matrix_y[point_y_in_matrix] == 1);
    int obstacle_right      = (matrix_x[right] == 1) && (matrix_y[point_y_in_matrix] == 1);
    int obstacle_down       = (matrix_x[point_x_in_matrix] == 1) && (matrix_y[down] == 1);
    int obstacle_down_left  = (matrix_x[left] == 1) && (matrix_y[down] == 1);
    int obstacle_down_right = (matrix_x[right] == 1) && (matrix_y[down] == 1);

    int obstacle_exists = obstacle_up || obstacle_up_left || obstacle_up_right || obstacle_left
                       || obstacle_right || obstacle_down || obstacle_down_left || obstacle_down_right;

    int obstacle_for_x = obstacle_left + obstacle_right;
    int obstacle_for_y = obstacle_up + obstacle_down;

    pos[gid].x += 0.001 * sign_x * (obstacle_for_x == 0);// - 0.001 * (obstacle_for_x != 0 && obstacle_for_y == 0));
    pos[gid].y += 0.001 * sign_y * (obstacle_for_y == 0) + 0.001 * (sign_obstacle_attraction_down - sign_obstacle_attraction_up) * (obstacle_for_y != 0);

/*-----------------------------------------------------------------------------------------------------------------*/

    /**
     *   AVOID NEIGHBOUR COLLISION
     */
    float2 back_off = (float2) (0.0f, 0.0f);

    int neighbour_up         = (neighbours_x[point_x_in_matrix] == 1) && (neighbours_y[up] == 1);
    int neighbour_up_left    = (neighbours_x[left] == 1) && (neighbours_y[up] == 1);
    int neighbour_up_right   = (neighbours_x[right] == 1) && (neighbours_y[up] == 1);
    int neighbour_left       = (neighbours_x[left] == 1) && (neighbours_y[point_y_in_matrix] == 1);
    int neighbour_right      = (neighbours_x[right] == 1) && (neighbours_y[point_y_in_matrix] == 1);
    int neighbour_down       = (neighbours_x[point_x_in_matrix] == 1) && (neighbours_y[down] == 1);
    int neighbour_down_left  = (neighbours_x[left] == 1) && (neighbours_y[down] == 1);
    int neighbour_down_right = (neighbours_x[right] == 1) && (neighbours_y[down] == 1);

    int neighbour_exists = neighbour_up || neighbour_up_left || neighbour_up_right || neighbour_left
                        || neighbour_right || neighbour_down || neighbour_down_left || neighbour_down_right;

    int neighbour_for_x = neighbour_left + neighbour_right;
    int neighbour_for_y = neighbour_up + neighbour_down;

    back_off.x += (neighbour_left == 1 || neighbour_down_left == 1 || neighbour_up_left == 1) * BACK_OFF * (obstacle_right == 0 || obstacle_up_right == 0 || obstacle_down_right == 0)
            - (neighbour_right == 1 || neighbour_up_right == 1 || neighbour_down_right == 1) * BACK_OFF * (obstacle_left == 0 || obstacle_up_left == 0 || obstacle_down_left == 0);
    back_off.y += (neighbour_down == 1 || neighbour_down_left == 1 || neighbour_down_right == 1) * BACK_OFF * (obstacle_up == 0 || obstacle_up_right == 0 || obstacle_up_left == 0)
            - (neighbour_up == 1 || neighbour_up_right == 1 || neighbour_up_left == 1) * BACK_OFF * (obstacle_down == 0 || obstacle_down_right == 0 || obstacle_down_left == 0);

    neighbours_x[point_x_in_matrix] = 0;
    neighbours_y[point_y_in_matrix] = 0;

    pos[gid].x += back_off.x;
    pos[gid].y += back_off.y;

    /**
     *  UPDATE NEIGHBOURS COLLISION MATRIX
     */
    int new_point_x = (int) (pos[gid].x * 100 + 96 + .5);
    int new_point_y = (int) (pos[gid].y * 100 + 96 + .5);

    int new_point_x_in_matrix = 193 * new_point_y + new_point_x;
    int new_point_y_in_matrix = 193 * new_point_x + new_point_y;

    neighbours_x[new_point_x_in_matrix] = 1;
    neighbours_y[new_point_y_in_matrix] = 1;
}

//__kernel void compute_velocity(__global float2* position, __global float2* old_position, __global float2* velocity)
//{
//    unsigned int gid = get_global_id(0);
//
//    velocity[gid] = (position[gid] - old_position[gid]);
//    old_position[gid] = position[gid];
//
//    int x_outside = fabs(position[gid].x) >= 0.8005f;
//    int y_outside = fabs(position[gid].y) >= 0.8005f;
//
//    int outside = x_outside || y_outside;
//
//    velocity[gid].x -= BOUNCING_SPEED_MODIFIER * (outside == 1);
//    velocity[gid].y -= BOUNCING_SPEED_MODIFIER * (outside == 1);
//}
//
//__kernel void attraction(__global float2* pos, __global int2* attraction_influence)
//{
//    unsigned int gid = get_global_id(0);
//
//    int influenced_point_index = attraction_influence[gid].x;
//    int atracted_by_index = attraction_influence[gid].y;
//
//    float sign_x = (pos[atracted_by_index].x - pos[influenced_point_index].x) > 0.0f; // == 1 -> positive value
//    float sign_y = (pos[atracted_by_index].y - pos[influenced_point_index].y) > 0.0f; // == 1 -> positive value
//
//    pos[influenced_point_index].x += ( (sign_x == 1) - (sign_x != 1) ) * ATTRACTION_FORCE;
//    pos[influenced_point_index].y += ( (sign_y == 1) - (sign_y != 1) ) * ATTRACTION_FORCE;
//}
