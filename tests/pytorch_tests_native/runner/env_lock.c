/* env_lock.c — LD_PRELOAD shim serializing environment access.
 *
 * Why: glibc < 2.41 getenv() walks the global `environ` array unlocked.
 * unsetenv()/setenv() mutate that array. A native thread calling getenv()
 * (e.g. libnrt kelf_load_from_neff during nrt_load on torch-neuronx's
 * RuntimeWorker) racing a thread calling unsetenv() (e.g. CPython restoring
 * os.environ in test teardown) segfaults the process.
 *
 * This shim interposes the public getenv/setenv/unsetenv/putenv symbols and
 * wraps them in a process-wide read-write lock: getenv takes the read lock
 * (concurrent readers OK), mutators take the write lock. All PLT-resolved
 * callers (libnrt, CPython, any dlopen'd library) go through the shim.
 * Callers *inside* libc itself bypass the PLT and are not covered — the
 * observed crash pair (libnrt reader / CPython writer) is fully covered.
 *
 * Fork safety: pthread_atfork reinitializes the lock in the child so a fork
 * during a held lock (pytest-forked, subprocess) can't deadlock the child.
 *
 * Built automatically by run_upstream_tests.py at startup (see
 * build_env_lock_shim). Manual build:
 *   cc -O2 -g -fPIC -shared -o env_lock.so env_lock.c -lpthread -ldl
 * Use:    LD_PRELOAD=/path/to/env_lock.so python -m pytest ...
 *
 * Maintenance Note:
 *  - Two independent things make this shim unnecessary, at which point it
 *    should be deleted:
 *      (a) glibc >= 2.41, where getenv() is synchronized against
 *          setenv/unsetenv (the CI image is currently glibc 2.35), OR
 *      (b) the runtime fix from issue #1290 lands.
 *  - Removal order:
 *      1. Get the fixed runtime (or a glibc >= 2.41 base) into the
 *         torch-neuronx CI image.
 *      2. Confirm the logging / test_foreach repros pass at -n 64 WITHOUT
 *         LD_PRELOAD (no SIGSEGV / worker loss).
 *      3. Then remove env_lock.c, its build/preload wiring in
 *         run_upstream_tests.py, and the shim-only tests in one
 *         cleanup PR.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>

/* NRT can issue getenv() from many threads at once. glibc's default
 * reader-preferred rwlock can starve an environment writer indefinitely, so
 * use its writer-preferred GNU initializer while preserving reader concurrency. */
#define ENV_RWLOCK_INITIALIZER PTHREAD_RWLOCK_WRITER_NONRECURSIVE_INITIALIZER_NP
static pthread_rwlock_t env_rwlock = ENV_RWLOCK_INITIALIZER;

static char *(*real_getenv)(const char *);
static int (*real_setenv)(const char *, const char *, int);
static int (*real_unsetenv)(const char *);
static int (*real_putenv)(char *);

static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;

static void resolve_real(void) {
  real_getenv = (char *(*)(const char *))dlsym(RTLD_NEXT, "getenv");
  real_setenv = (int (*)(const char *, const char *, int))dlsym(RTLD_NEXT, "setenv");
  real_unsetenv = (int (*)(const char *))dlsym(RTLD_NEXT, "unsetenv");
  real_putenv = (int (*)(char *))dlsym(RTLD_NEXT, "putenv");
}

/* After fork the child is single-threaded; a lock held by a thread that no
 * longer exists would deadlock it. Reset unconditionally. */
static void atfork_child(void) {
  pthread_rwlock_t fresh = ENV_RWLOCK_INITIALIZER;
  env_rwlock = fresh;
}

__attribute__((constructor)) static void init(void) {
  pthread_once(&resolve_once, resolve_real);
  pthread_atfork(NULL, NULL, atfork_child);
}

char *getenv(const char *name) {
  pthread_once(&resolve_once, resolve_real);
  if (!real_getenv)
    return NULL; /* resolution failed; nothing sane to do */
  pthread_rwlock_rdlock(&env_rwlock);
  char *value = real_getenv(name);
  pthread_rwlock_unlock(&env_rwlock);
  /* NOTE: the returned pointer is only guaranteed stable until the next
   * mutation — same contract as plain glibc getenv. The lock removes the
   * mid-walk crash, which is the failure mode being fixed. */
  return value;
}

int setenv(const char *name, const char *value, int overwrite) {
  pthread_once(&resolve_once, resolve_real);
  if (!real_setenv)
    return -1;
  pthread_rwlock_wrlock(&env_rwlock);
  int rc = real_setenv(name, value, overwrite);
  pthread_rwlock_unlock(&env_rwlock);
  return rc;
}

int unsetenv(const char *name) {
  pthread_once(&resolve_once, resolve_real);
  if (!real_unsetenv)
    return -1;
  pthread_rwlock_wrlock(&env_rwlock);
  int rc = real_unsetenv(name);
  pthread_rwlock_unlock(&env_rwlock);
  return rc;
}

int putenv(char *string) {
  pthread_once(&resolve_once, resolve_real);
  if (!real_putenv)
    return -1;
  pthread_rwlock_wrlock(&env_rwlock);
  int rc = real_putenv(string);
  pthread_rwlock_unlock(&env_rwlock);
  return rc;
}
