#include <glad/glad.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void* get_proc(const char *namez);

int gladLoadGLLoader(GLADloadproc load) {
    if (load == NULL) {
        return 0;
    }
    
    // For now, just return success since we're using system GL headers
    return 1;
}

static void* get_proc(const char *namez) {
    return NULL;
}