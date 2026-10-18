/* Compatibility for a PID namespace where /proc/<getpid>/exe is absent,
   while /proc/self/exe is available. No Lean or theorem-checking code changes. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

ssize_t readlink(const char *path, char *buf, size_t size) {
    static ssize_t (*original)(const char *, char *, size_t) = 0;
    if (!original) original = dlsym(RTLD_NEXT, "readlink");
    char expected[80];
    snprintf(expected, sizeof(expected), "/proc/%d/exe", getpid());
    return original(strcmp(path, expected) == 0 ? "/proc/self/exe" : path, buf, size);
}
