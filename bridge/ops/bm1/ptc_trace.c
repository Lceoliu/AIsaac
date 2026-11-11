// bm1 diagnosis: LD_PRELOAD shim that logs every pthread_create with the creating thread's native backtrace
// (glibc backtrace_symbols_fd), to name the library that starts a thread. Measurement only.
//   gcc -shared -fPIC -O1 -o ptc_trace.so ptc_trace.c -ldl ; LD_PRELOAD=./ptc_trace.so PTC_LOG=/path python ...
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef int (*create_fn)(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);

int pthread_create(pthread_t *t, const pthread_attr_t *a, void *(*f)(void *), void *arg) {
  static create_fn real = 0;
  if (!real) real = (create_fn)dlsym(RTLD_NEXT, "pthread_create");
  const char *path = getenv("PTC_LOG");
  int fd = path ? open(path, O_WRONLY | O_CREAT | O_APPEND, 0644) : 2;
  if (fd >= 0) {
    char head[96];
    int n = snprintf(head, sizeof head, "=== pthread_create by tid %d\n", (int)gettid());
    if (write(fd, head, n) < 0) {}
    void *bt[24];
    int k = backtrace(bt, 24);
    backtrace_symbols_fd(bt, k, fd);
    if (path) close(fd);
  }
  return real(t, a, f, arg);
}
