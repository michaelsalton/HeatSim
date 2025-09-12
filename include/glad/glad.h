#ifndef __glad_h_
#define __glad_h_

#include <GL/gl.h>
#include <GL/glext.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* (*GLADloadproc)(const char *name);
int gladLoadGLLoader(GLADloadproc);

#ifdef __cplusplus
}
#endif

#endif