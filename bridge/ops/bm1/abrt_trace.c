// bm1 diagnosis: LD_PRELOAD shim that, in the train_tok process only, prints the aborting thread's native backtrace
// on SIGABRT (std::terminate -> abort) and every thread's name, then lets the abort proceed. Measurement only.
//   gcc -shared -fPIC -O1 -o abrt_trace.so abrt_trace.c ; LD_PRELOAD=./abrt_trace.so ABRT_LOG=/path python ...
#define _GNU_SOURCE
#include <dirent.h>
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int log_fd = 2;

static void on_abort(int sig) {
  char buf[256];
  int n = snprintf(buf, sizeof buf, "=== SIGABRT in tid %d (pid %d); backtrace:\n", (int)gettid(), (int)getpid());
  if (write(log_fd, buf, n) < 0) {}
  void *bt[48];
  int k = backtrace(bt, 48);
  backtrace_symbols_fd(bt, k, log_fd);
  DIR *d = opendir("/proc/self/task");   // (the threads still alive at that moment, with their names)
  struct dirent *e;
  while (d && (e = readdir(d))) {
    if (e->d_name[0] == '.') continue;
    char p[96], name[64] = "";
    snprintf(p, sizeof p, "/proc/self/task/%s/comm", e->d_name);
    int f = open(p, O_RDONLY);
    if (f >= 0) { int m = read(f, name, sizeof name - 1); if (m > 0) name[m] = 0; close(f); }
    n = snprintf(buf, sizeof buf, "  thread %s %s", e->d_name, name);
    if (write(log_fd, buf, n) < 0) {}
  }
  if (d) closedir(d);
  signal(SIGABRT, SIG_DFL);
  raise(SIGABRT);
}

__attribute__((constructor)) static void init(void) {
  char cmd[4096] = "";
  int f = open("/proc/self/cmdline", O_RDONLY);
  if (f < 0) return;
  int m = read(f, cmd, sizeof cmd - 1);
  close(f);
  if (m <= 0) return;
  for (int i = 0; i < m; i++) if (!cmd[i]) cmd[i] = ' ';
  cmd[m] = 0;
  if (!strstr(cmd, "train_tok.py")) return;   // (the trainer only: not its workers' or the games' processes)
  const char *path = getenv("ABRT_LOG");
  if (path) { int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644); if (fd >= 0) log_fd = fd; }
  signal(SIGABRT, on_abort);
}
