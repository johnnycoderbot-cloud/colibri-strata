#include "mojo_split_loader.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32) && (defined(__linux__) || defined(__APPLE__))
#include <dlfcn.h>
#define COLI_NATIVE_MOJO_SUPPORTED 1
#else
#define COLI_NATIVE_MOJO_SUPPORTED 0
#endif

struct ColiMojoSplit {
    void *handle;
    ColiSplitPlanner planner;
};

static int mojo_error(char *error, size_t cap, const char *message) {
    if (error && cap) snprintf(error, cap, "%s", message);
    return -1;
}

int coli_mojo_split_open(const char *path, ColiMojoSplit **out,
                         char *error, size_t error_cap) {
    if (!out || *out)
        return mojo_error(error, error_cap, "Mojo output handle must initially be NULL");
#if !COLI_NATIVE_MOJO_SUPPORTED
    (void)path;
    return mojo_error(error, error_cap,
        "Native Mojo dispatch requires Linux/macOS; build and run inside WSL on Windows");
#else
    typedef uint32_t (*VersionFn)(void);
    typedef int32_t (*InitFn)(void);
    VersionFn version = NULL;
    InitFn init = NULL;
    ColiSplitPlanner planner = NULL;
    void *symbol;
    if (!path) path = getenv("COLI_MOJO_DISPATCH");
    if (!path || !*path)
        return mojo_error(error, error_cap, "Mojo dispatch path is not configured");
    ColiMojoSplit *result = calloc(1, sizeof *result);
    if (!result)
        return mojo_error(error, error_cap, "Cannot allocate Mojo dispatch handle");
    result->handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!result->handle) {
        const char *detail = dlerror();
        mojo_error(error, error_cap, detail ? detail : "Cannot load Mojo dispatch library");
        free(result);
        return -1;
    }
    _Static_assert(sizeof version == sizeof symbol, "POSIX function pointer size");
    _Static_assert(sizeof init == sizeof symbol, "POSIX function pointer size");
    _Static_assert(sizeof planner == sizeof symbol, "POSIX function pointer size");
    dlerror();
    symbol = dlsym(result->handle, "coli_mojo_split_abi");
    if (!symbol || dlerror()) goto missing;
    memcpy(&version, &symbol, sizeof version);
    symbol = dlsym(result->handle, "coli_mojo_split_init");
    if (!symbol || dlerror()) goto missing;
    memcpy(&init, &symbol, sizeof init);
    symbol = dlsym(result->handle, "coli_mojo_split_plan");
    if (!symbol || dlerror()) goto missing;
    memcpy(&planner, &symbol, sizeof planner);
    if (version() != COLI_SPLIT_ABI) {
        mojo_error(error, error_cap, "Mojo dispatch ABI version mismatch");
        goto failed;
    }
    if (init() != 0) {
        mojo_error(error, error_cap, "Mojo dispatch runtime initialization failed");
        goto failed;
    }
    result->planner = planner;
    *out = result;
    if (error && error_cap) error[0] = '\0';
    return 0;
missing:
    mojo_error(error, error_cap, "Mojo dispatch library lacks required ABI/init/plan symbols");
failed:
    dlclose(result->handle);
    free(result);
    return -1;
#endif
}

ColiSplitPlanner coli_mojo_split_planner(const ColiMojoSplit *context) {
    return context ? context->planner : NULL;
}

void coli_mojo_split_close(ColiMojoSplit *context) {
    if (!context) return;
#if COLI_NATIVE_MOJO_SUPPORTED
    if (context->handle) dlclose(context->handle);
#endif
    free(context);
}
