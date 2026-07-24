/*
 * Why: glibc < 2.41 getenv() walks the global `environ` array unlocked.
 * unsetenv()/setenv() mutate that array. A native thread calling getenv()
 * (e.g. libnrt kelf_load_from_neff during nrt_load on torch-neuronx's
 * RuntimeWorker) racing a thread calling unsetenv() (e.g. CPython restoring
 * os.environ in test teardown) can segfault the process.
 *
 * ProcessExecutor has a second, semantic hazard. unittest.mock.patch.dict
 * restores os.environ by clearing every key and then inserting the original
 * keys one at a time. A compiler spawn can therefore observe a valid but
 * incomplete environment with no PATH and return ENOENT.
 *
 * This upstream-test-only shim interposes getenv/setenv/unsetenv/putenv and
 * wraps them in a process-wide read-write lock. It also captures an immutable
 * startup baseline before Python test code runs. The ProcessExecutor helper
 * takes a consistent snapshot of the current environment and fills a
 * temporarily absent PATH from that baseline. Intentional late additions,
 * updates, and removals of every other key still reach compiler children, while
 * patch.dict cannot remove executable lookup mid-restore. Executables are resolved
 * against the child environment and intended child working directory before
 * real posix_spawn is called with an absolute path and the private snapshot.
 *
 * Fork safety: pthread_atfork reinitializes the lock in the child so a fork
 * during a held lock (pytest-forked, subprocess) cannot deadlock the child.
 * The immutable startup baseline is inherited safely by the child.
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
 *         LD_PRELOAD (no SIGSEGV / worker loss / spawn failure).
 *      3. Then remove env_lock.c, its build/preload wiring in
 *         run_upstream_tests.py, and the shim-only tests in one cleanup PR.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <spawn.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

extern char **environ;

/* NRT can issue getenv() from many threads at once. glibc's default
 * reader-preferred rwlock can starve an environment writer indefinitely, so
 * use its writer-preferred GNU initializer while preserving reader concurrency. */
#define ENV_RWLOCK_INITIALIZER PTHREAD_RWLOCK_WRITER_NONRECURSIVE_INITIALIZER_NP
static pthread_rwlock_t env_rwlock = ENV_RWLOCK_INITIALIZER;
static uint64_t environment_generation;

static char *(*real_getenv)(const char *);
static int (*real_setenv)(const char *, const char *, int);
static int (*real_unsetenv)(const char *);
static int (*real_putenv)(char *);
static int (*real_posix_spawn)(pid_t *, const char *, const posix_spawn_file_actions_t *,
                               const posix_spawnattr_t *, char *const[], char *const[]);

static char **startup_environ;
static int startup_init_error;
static pthread_once_t symbols_once = PTHREAD_ONCE_INIT;
static pthread_once_t startup_once = PTHREAD_ONCE_INIT;

static void resolve_symbols(void) {
  /* Keep this callback allocation-free. A later LD_PRELOAD library may
   * interpose allocation/string routines and call getenv from those wrappers. */
  real_getenv = (char *(*)(const char *))dlsym(RTLD_NEXT, "getenv");
  real_setenv = (int (*)(const char *, const char *, int))dlsym(RTLD_NEXT, "setenv");
  real_unsetenv = (int (*)(const char *))dlsym(RTLD_NEXT, "unsetenv");
  real_putenv = (int (*)(char *))dlsym(RTLD_NEXT, "putenv");
  real_posix_spawn =
      (int (*)(pid_t *, const char *, const posix_spawn_file_actions_t *,
               const posix_spawnattr_t *, char *const[], char *const[]))
          dlsym(RTLD_NEXT, "posix_spawn");
}

static void free_environment_copy(char **copy) {
  if (!copy)
    return;
  for (size_t i = 0; copy[i]; ++i)
    free(copy[i]);
  free(copy);
}

static void capture_startup_environment(void) {
  size_t count = 0;
  while (environ[count])
    ++count;

  char **copy = (char **)calloc(count + 1, sizeof(char *));
  if (!copy) {
    startup_init_error = ENOMEM;
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    copy[i] = strdup(environ[i]);
    if (!copy[i]) {
      free_environment_copy(copy);
      startup_init_error = ENOMEM;
      return;
    }
  }
  startup_environ = copy;
}

/* These helpers run while env_rwlock is held. Volatile byte access prevents
 * the compiler from replacing the loops with interposable libc string calls. */
static size_t locked_string_length(const char *value) {
  const volatile unsigned char *cursor = (const volatile unsigned char *)value;
  size_t length = 0;
  while (cursor[length] != '\0')
    ++length;
  return length;
}

static void locked_copy_bytes(char *destination, const char *source, size_t length) {
  volatile unsigned char *out = (volatile unsigned char *)destination;
  const volatile unsigned char *in = (const volatile unsigned char *)source;
  for (size_t i = 0; i < length; ++i)
    out[i] = in[i];
}

static size_t locked_environment_key_length(const char *entry) {
  const volatile unsigned char *cursor = (const volatile unsigned char *)entry;
  size_t length = 0;
  while (cursor[length] != '\0' && cursor[length] != '=')
    ++length;
  return length;
}

static int locked_keys_equal(const char *left, const char *right, size_t length) {
  const volatile unsigned char *a = (const volatile unsigned char *)left;
  const volatile unsigned char *b = (const volatile unsigned char *)right;
  for (size_t i = 0; i < length; ++i) {
    if (a[i] != b[i])
      return 0;
  }
  return 1;
}

static int is_path_entry(const char *entry) {
  return locked_environment_key_length(entry) == 4 && locked_keys_equal(entry, "PATH", 4);
}

static int locked_environment_has_key(char *const envp[], const char *entry) {
  size_t key_length = locked_environment_key_length(entry);
  for (size_t i = 0; envp[i]; ++i) {
    if (locked_environment_key_length(envp[i]) == key_length &&
        locked_keys_equal(envp[i], entry, key_length))
      return 1;
  }
  return 0;
}

/* Build a private spawn environment without allocating while env_rwlock is
 * held. Current values win; only a temporarily absent PATH is filled from the
 * startup baseline. Retry if a mutator ran between sizing and copying. The returned strings share one allocation at envp[0]'s backing
 * storage, tracked separately through storage_out. */
static int snapshot_spawn_environment(char ***envp_out, char **storage_out) {
  for (;;) {
    size_t count = 0;
    size_t bytes = 0;
    uint64_t generation;

    pthread_rwlock_rdlock(&env_rwlock);
    generation = environment_generation;
    for (size_t i = 0; environ[i]; ++i) {
      ++count;
      bytes += locked_string_length(environ[i]) + 1;
    }
    if (startup_environ) {
      for (size_t i = 0; startup_environ[i]; ++i) {
        if (is_path_entry(startup_environ[i]) &&
            !locked_environment_has_key(environ, startup_environ[i])) {
          ++count;
          bytes += locked_string_length(startup_environ[i]) + 1;
        }
      }
    }
    pthread_rwlock_unlock(&env_rwlock);

    char **envp = (char **)calloc(count + 1, sizeof(char *));
    char *storage = (char *)malloc(bytes ? bytes : 1);
    if (!envp || !storage) {
      free(envp);
      free(storage);
      return ENOMEM;
    }

    pthread_rwlock_rdlock(&env_rwlock);
    if (generation != environment_generation) {
      pthread_rwlock_unlock(&env_rwlock);
      free(envp);
      free(storage);
      continue;
    }

    size_t index = 0;
    char *destination = storage;
    for (size_t i = 0; environ[i]; ++i) {
      size_t length = locked_string_length(environ[i]) + 1;
      envp[index++] = destination;
      locked_copy_bytes(destination, environ[i], length);
      destination += length;
    }
    if (startup_environ) {
      for (size_t i = 0; startup_environ[i]; ++i) {
        if (is_path_entry(startup_environ[i]) &&
            !locked_environment_has_key(environ, startup_environ[i])) {
          size_t length = locked_string_length(startup_environ[i]) + 1;
          envp[index++] = destination;
          locked_copy_bytes(destination, startup_environ[i], length);
          destination += length;
        }
      }
    }
    pthread_rwlock_unlock(&env_rwlock);

    envp[index] = NULL;
    *envp_out = envp;
    *storage_out = storage;
    return 0;
  }
}

static const char *find_environment_value(char *const envp[], const char *name) {
  size_t name_length = strlen(name);
  for (size_t i = 0; envp[i]; ++i) {
    if (strncmp(envp[i], name, name_length) == 0 && envp[i][name_length] == '=')
      return envp[i] + name_length + 1;
  }
  return NULL;
}

static char *make_absolute_candidate(const char *directory, size_t directory_length,
                                     const char *file, const char *working_dir, int *error) {
  const char *base = directory;
  size_t base_length = directory_length;
  char *combined = NULL;

  if (directory_length == 0) {
    base = ".";
    base_length = 1;
  }
  if (base[0] != '/' && working_dir && *working_dir) {
    size_t working_length = strlen(working_dir);
    size_t combined_length = working_length + 1 + base_length + 1;
    combined = (char *)malloc(combined_length);
    if (!combined) {
      *error = ENOMEM;
      return NULL;
    }
    snprintf(combined, combined_length, "%s/%.*s", working_dir, (int)base_length, base);
    base = combined;
    base_length = strlen(combined);
  }

  size_t candidate_length = base_length + 1 + strlen(file) + 1;
  char *candidate = (char *)malloc(candidate_length);
  if (!candidate) {
    free(combined);
    *error = ENOMEM;
    return NULL;
  }
  snprintf(candidate, candidate_length, "%.*s/%s", (int)base_length, base, file);
  free(combined);
  return candidate;
}

/* Resolve like posix_spawnp after its chdir file action: relative slash paths
 * and relative PATH entries are interpreted from the intended child CWD. */
static char *resolve_spawn_file(const char *file, const char *path, const char *working_dir,
                                int *error) {
  if (!file || !*file) {
    *error = ENOENT;
    return NULL;
  }

  if (strchr(file, '/')) {
    char *candidate;
    if (file[0] == '/' || !working_dir || !*working_dir) {
      candidate = strdup(file);
      if (!candidate) {
        *error = ENOMEM;
        return NULL;
      }
    } else {
      candidate = make_absolute_candidate(working_dir, strlen(working_dir), file, NULL, error);
      if (!candidate)
        return NULL;
    }
    char *resolved = realpath(candidate, NULL);
    free(candidate);
    if (resolved)
      return resolved;
    *error = errno;
    return NULL;
  }

  if (!path)
    path = "/bin:/usr/bin";
  const char *cursor = path;
  int saw_eacces = 0;
  while (1) {
    const char *separator = strchr(cursor, ':');
    size_t directory_length = separator ? (size_t)(separator - cursor) : strlen(cursor);
    char *candidate =
        make_absolute_candidate(cursor, directory_length, file, working_dir, error);
    if (!candidate)
      return NULL;

    struct stat candidate_stat;
    if (stat(candidate, &candidate_stat) != 0) {
      int candidate_error = errno;
      free(candidate);
      if (candidate_error == EACCES) {
        saw_eacces = 1;
      } else if (candidate_error != ENOENT && candidate_error != ENOTDIR) {
        *error = candidate_error;
        return NULL;
      }
    } else if (!S_ISREG(candidate_stat.st_mode)) {
      saw_eacces = 1;
      free(candidate);
    } else if (access(candidate, X_OK) != 0) {
      int candidate_error = errno;
      free(candidate);
      if (candidate_error == EACCES) {
        saw_eacces = 1;
      } else if (candidate_error != ENOENT && candidate_error != ENOTDIR) {
        *error = candidate_error;
        return NULL;
      }
    } else {
      char *resolved = realpath(candidate, NULL);
      int candidate_error = errno;
      free(candidate);
      if (resolved)
        return resolved;
      if (candidate_error == EACCES) {
        saw_eacces = 1;
      } else if (candidate_error != ENOENT && candidate_error != ENOTDIR) {
        *error = candidate_error;
        return NULL;
      }
    }

    if (!separator)
      break;
    cursor = separator + 1;
  }

  *error = saw_eacces ? EACCES : ENOENT;
  return NULL;
}

/* After fork the child is single-threaded; a lock held by a thread that no
 * longer exists would deadlock it. Reset unconditionally. */
static void atfork_child(void) {
  pthread_rwlock_t fresh = ENV_RWLOCK_INITIALIZER;
  env_rwlock = fresh;
}

__attribute__((constructor)) static void init(void) {
  pthread_once(&symbols_once, resolve_symbols);
  pthread_once(&startup_once, capture_startup_environment);
  pthread_atfork(NULL, NULL, atfork_child);
}

char *getenv(const char *name) {
  pthread_once(&symbols_once, resolve_symbols);
  if (!real_getenv)
    return NULL;
  pthread_rwlock_rdlock(&env_rwlock);
  char *value = real_getenv(name);
  pthread_rwlock_unlock(&env_rwlock);
  /* Same lifetime contract as glibc getenv: valid only until a mutation. */
  return value;
}

int setenv(const char *name, const char *value, int overwrite) {
  pthread_once(&symbols_once, resolve_symbols);
  if (!real_setenv) {
    errno = ENOSYS;
    return -1;
  }
  pthread_rwlock_wrlock(&env_rwlock);
  int rc = real_setenv(name, value, overwrite);
  if (rc == 0)
    ++environment_generation;
  pthread_rwlock_unlock(&env_rwlock);
  return rc;
}

int unsetenv(const char *name) {
  pthread_once(&symbols_once, resolve_symbols);
  if (!real_unsetenv) {
    errno = ENOSYS;
    return -1;
  }
  pthread_rwlock_wrlock(&env_rwlock);
  int rc = real_unsetenv(name);
  if (rc == 0)
    ++environment_generation;
  pthread_rwlock_unlock(&env_rwlock);
  return rc;
}

int putenv(char *string) {
  pthread_once(&symbols_once, resolve_symbols);
  if (!real_putenv) {
    errno = ENOSYS;
    return -1;
  }
  pthread_rwlock_wrlock(&env_rwlock);
  int rc = real_putenv(string);
  if (rc == 0)
    ++environment_generation;
  pthread_rwlock_unlock(&env_rwlock);
  return rc;
}

int torch_neuronx_posix_spawn_with_stable_environ(
    pid_t *pid, const char *file, const posix_spawn_file_actions_t *file_actions,
    const posix_spawnattr_t *attrp, char *const argv[], const char *working_dir) {
  pthread_once(&symbols_once, resolve_symbols);
  pthread_once(&startup_once, capture_startup_environment);
  if (startup_init_error)
    return startup_init_error;
  if (!real_posix_spawn || !startup_environ)
    return ENOSYS;

  char **spawn_environ = NULL;
  char *spawn_storage = NULL;
  int rc = snapshot_spawn_environment(&spawn_environ, &spawn_storage);
  if (rc != 0)
    return rc;

  const char *path = find_environment_value(spawn_environ, "PATH");
  int resolve_error = 0;
  char *resolved_file = resolve_spawn_file(file, path, working_dir, &resolve_error);
  if (!resolved_file) {
    free(spawn_storage);
    free(spawn_environ);
    return resolve_error;
  }

  rc = real_posix_spawn(pid, resolved_file, file_actions, attrp, argv, spawn_environ);
  free(resolved_file);
  free(spawn_storage);
  free(spawn_environ);
  return rc;
}
