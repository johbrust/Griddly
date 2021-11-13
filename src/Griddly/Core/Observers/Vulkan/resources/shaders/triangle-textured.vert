#version 460

layout(location=0)in vec3 inPosition;
layout(location=1)in vec2 inFragTextureCoords;

layout(location=0)out vec4 outColor;
layout(location=1)out vec3 outFragTextureCoords;

out gl_PerVertex{
    vec4 gl_Position;
};

struct GlobalVariable{
    int value;
};

struct ObjectData{
    mat4 modelMatrix;
    vec2 textureMultiply;
    int textureIndex;
    int playerId;
    int zIdx;
};

layout(std140,binding=1)uniform EnvironmentData{
    mat4 projectionMatrix;
    mat4 viewMatrix;
    vec2 gridDims;
    uint playerId;
}environmentData;

layout(std430,binding=2)readonly buffer ObjectDataBuffer{
    ObjectData variables[];
}objectDataBuffer;

layout(std430,binding=3)readonly buffer GlobalVariableBuffer{
    GlobalVariable variables[];
}globalVariableBuffer;

layout(push_constant)uniform PushConsts{
    uint idx;
}pushConsts;

void main()
{
    ObjectData object=objectDataBuffer.variables[pushConsts.idx];
    GlobalVariable globalVariable=globalVariableBuffer.variables[0];
    
    outFragTextureCoords=vec3(
        inFragTextureCoords.x*object.textureMultiply.x,
        inFragTextureCoords.y*object.textureMultiply.y,
        object.textureIndex
    );
    
    mat4 mvp=environmentData.projectionMatrix*environmentData.viewMatrix*object.modelMatrix;
    
    gl_Position=mvp*vec4(
        inPosition.x,
        inPosition.y,
        inPosition.z,
        1.
    );
}