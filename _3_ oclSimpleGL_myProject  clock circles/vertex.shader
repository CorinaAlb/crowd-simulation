#version 430 core

in vec4 position;
in vec4 color;
out vec4 color_from_vshader;

void main()
{
    gl_PointSize = 10.0;
    gl_Position = position;

    color_from_vshader = color;
}
