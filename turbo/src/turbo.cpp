// isaac_turbo.dll —— J460 原版加速层（L1）的原生部分 + 原生崩溃探针。
//
// 加速层只做三件事，不碰任何玩法状态：
//   1. glfwGetTime（RVA 0x006266A0）返回虚拟时钟：每次 Manager::Update 完成推进 1/60 s（略多），
//      外层循环的 Sleep/自旋限帧立即满足；读时钟不推进时间。
//   2. 整帧渲染函数（RVA 0x005555C0）按控制块决定是否跳过；可按 N 帧或按请求放行。
//   3. 计数与计时：Manager::Update / Game::Update 调用数与耗时，写入命名共享内存供 Python 读取。
//
// 崩溃探针（2026-09-20，rl/docs/NATIVE_CRASH_ANALYSIS.md）：
//   4. shader 入栈 FUN_00a140c0（RVA 0x006140C0）：返回 0 时记录注册表节点/对象 flags/栈状态/返回地址并可写带堆的 minidump。
//      这是 A 类崩溃（字体绘制 shader 栈下溢）的直接前提，旧转储缺堆页，只能在这里取证。
//   5. 字体绘制 FUN_00a1bd80（RVA 0x0061BD80）：kFlagFontGuard 打开时先查 KAGE_ColorTextureShader 是否可用，
//      不可用则跳过这一次文本绘制（缓解，不改玩法状态；训练默认无渲染时不会触发）。
//   6. KAGE 文件打开 FUN_00a17ea0（RVA 0x00617EA0）：失败时记录路径、errno/_doserrno/GetLastError；
//      kFlagFileRetry 仅重试绝对 .a；相对归档路径解析失败不在此策略范围。
//   7. 可选resolver诊断（ISAAC_TURBO_RESOLVER_PROBE=1），默认不安装，不改查找结果。
//   8. worker启动时可排除NvCamera32/nvspcap录屏叠加层；不改驱动/系统设置，不卸载已加载模块。
//
// 单步同步（step(N) 恰好 N 个逻辑帧）由 Lua 桥接 mod 在 MC_POST_UPDATE 里阻塞完成，本 DLL 不做。
// 证据与设计：rl/docs/L1_FEASIBILITY_PLAN.md、analysis/docs/J460_FRAME_LOOP.md。
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <dbghelp.h>
#include <winternl.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>

#include "MinHook.h"
#include "generated/j460_targets.hpp"
#include "shader_registry.hpp"
#include "turbo_shared.hpp"
#include "virtual_clock.hpp"

namespace {

using namespace isaac_turbo;

constexpr std::uint32_t kDllVersion = 3;

HMODULE gSelf = nullptr;
volatile LONG gInitState = 0;  // 0 未开始，1 进行中，2 完成
volatile LONG gFinalStatus = kStatusLoading;
ControlBlock* gBlock = nullptr;
HANDLE gMapping = nullptr;
FILE* gLog = nullptr;
char gLogDir[MAX_PATH] = {};
char gLogPath[256] = {};
char gStatus[128] = "LOADING";

VirtualClock gClock;
bool gClockUsable = false;
const std::uint8_t* gImageBase = nullptr;
std::uint32_t gImageSize = 0;
const std::uint64_t* gGlfwOffset = nullptr;
const std::uint64_t* gGlfwFrequency = nullptr;
const std::uint8_t* const* gManagerPtr = nullptr;
LARGE_INTEGER gQpf{};
LARGE_INTEGER gQpc0{};

bool gDrawParityOk = false;
std::uint32_t gLastCounterSeen = 0;
std::uint32_t gCounterChecks = 0;
std::uint32_t gCounterMismatches = 0;
std::uint64_t gRenderOpportunities = 0;
std::uint64_t gGameUpdateMinUs = ~0ull;

using RenderFn = void(__fastcall*)(void*, void*);
using ManagerUpdateFn = void(__stdcall*)(char);
using GameUpdateFn = void(__fastcall*)(void*, void*);
using PushShaderFn = int(__stdcall*)(const char*);
using FileOpenFn = unsigned char(__stdcall*)(const char*, void**);
using ResolvePathFn = char*(__fastcall*)(void*, void*, const char*);
using SearchPathFn = void*(__fastcall*)(void*, void*, const char*, unsigned char*);
using GetErrnoFn = int(__cdecl*)(int*);
using AccessFn = int(__cdecl*)(const char*, int);
using SetErrnoFn = int(__cdecl*)(int);
using SetDosErrnoFn = int(__cdecl*)(unsigned long);
RenderFn gOrigRender = nullptr;
ManagerUpdateFn gOrigManagerUpdate = nullptr;
GameUpdateFn gOrigGameUpdate = nullptr;
PushShaderFn gOrigPushShader = nullptr;
FileOpenFn gOrigFileOpen = nullptr;
ResolvePathFn gOrigResolvePath = nullptr;
SearchPathFn gOrigSearchPath = nullptr;
LONG gArchiveResolveCalls = 0;
LONG gArchiveResolveRecoveries = 0;
bool gArchivePathFix = false;
bool gResolverProbe = false;
thread_local void* gLastSearchDirectory = nullptr;
thread_local unsigned char gLastSearchInArchive = 0;
thread_local char gLastSearchCopy[384] = {};
thread_local std::uint32_t gLastSearchKey = 0;
void* gOrigGlfwGetTime = nullptr;  // 只为满足 MinHook 接口，本 DLL 不回调它（返回值在 XMM0，C 拿不到）
GetErrnoFn gGetErrno = nullptr;
GetErrnoFn gGetDosErrno = nullptr;
AccessFn gOrigAccess = nullptr;
SetErrnoFn gSetErrno = nullptr;
SetDosErrnoFn gSetDosErrno = nullptr;
volatile LONG gAccessTotal = 0;
volatile LONG gAccessFailed = 0;
using LdrLoadDllFn = LONG(NTAPI*)(PWSTR, ULONG*, UNICODE_STRING*, HMODULE*);
LdrLoadDllFn gOrigLdrLoadDll = nullptr;

// ---------------------------------------------------------------- 日志与状态

void logf(const char* fmt, ...) {
  if (!gLog) return;
  SYSTEMTIME st;
  GetLocalTime(&st);
  std::fprintf(gLog, "%02u:%02u:%02u.%03u ", st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(gLog, fmt, ap);
  va_end(ap);
  std::fputc('\n', gLog);
  std::fflush(gLog);
}

void setStatus(Status s, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(gStatus, sizeof(gStatus), fmt, ap);
  va_end(ap);
  gStatus[sizeof(gStatus) - 1] = '\0';
  if (gBlock) {
    gBlock->status = s;
    std::memcpy(gBlock->status_text, gStatus, sizeof(gStatus));
  }
  logf("status %u: %s", static_cast<unsigned>(s), gStatus);
}

void ensureDirectory(const char* path) {
  // 逐级建目录（只在用户目录下，绝不在游戏目录里）。
  char buf[MAX_PATH];
  std::snprintf(buf, sizeof(buf), "%s", path);
  for (char* p = buf + 3; *p; ++p) {
    if (*p == '\\' || *p == '/') {
      *p = '\0';
      CreateDirectoryA(buf, nullptr);
      *p = '\\';
    }
  }
  CreateDirectoryA(buf, nullptr);
}

void openLog(DWORD pid) {
  char dir[MAX_PATH] = {};
  DWORD n = GetEnvironmentVariableA("ISAAC_TURBO_LOG_DIR", dir, sizeof(dir));
  if (n == 0 || n >= sizeof(dir)) {
    char local[MAX_PATH] = {};
    n = GetEnvironmentVariableA("LOCALAPPDATA", local, sizeof(local));
    if (n == 0 || n >= sizeof(local)) return;
    std::snprintf(dir, sizeof(dir), "%s\\IsaacRL\\turbo", local);
  }
  ensureDirectory(dir);
  std::snprintf(gLogDir, sizeof(gLogDir), "%s", dir);
  std::snprintf(gLogPath, sizeof(gLogPath), "%s\\turbo-%lu.log", dir, static_cast<unsigned long>(pid));
  gLog = std::fopen(gLogPath, "a");
}

std::uint64_t wallMicros() {
  LARGE_INTEGER q;
  QueryPerformanceCounter(&q);
  return static_cast<std::uint64_t>(q.QuadPart - gQpc0.QuadPart) * 1000000ull /
         static_cast<std::uint64_t>(gQpf.QuadPart);
}

// ---------------------------------------------------------------- 控制块

bool envFlag(const char* name) {
  char v[16] = {};
  const DWORD n = GetEnvironmentVariableA(name, v, sizeof(v));
  return n > 0 && n < sizeof(v) && v[0] == '1';
}

bool openControlBlock(DWORD pid) {
  char name[64];
  formatMappingName(name, sizeof(name), pid);
  HANDLE h = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name);
  const bool preexisting = h != nullptr;
  if (!h) {
    h = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, kControlBlockBytes, name);
  }
  if (!h) {
    logf("control block: CreateFileMapping failed error=%lu", GetLastError());
    return false;
  }
  void* view = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, kControlBlockBytes);
  if (!view) {
    logf("control block: MapViewOfFile failed error=%lu", GetLastError());
    CloseHandle(h);
    return false;
  }
  gMapping = h;
  auto* b = static_cast<ControlBlock*>(view);
  const bool configured = preexisting && b->magic == kMagic && b->version == kLayoutVersion;
  std::uint32_t flags = 0, renderEvery = 0, renderRequest = 0;
  if (configured) {
    flags = b->flags;
    renderEvery = b->render_every;
    renderRequest = b->render_request;
  } else {
    if (envFlag("ISAAC_TURBO")) flags |= kFlagVirtualClock | kFlagSkipRender;
    if (envFlag("ISAAC_TURBO_CLOCK")) flags |= kFlagVirtualClock;
    if (envFlag("ISAAC_TURBO_SKIP_RENDER")) flags |= kFlagSkipRender;
    if (envFlag("ISAAC_TURBO_FONT_GUARD")) flags |= kFlagFontGuard;
    if (envFlag("ISAAC_TURBO_FILE_RETRY")) flags |= kFlagFileRetry;
    if (envFlag("ISAAC_TURBO_PROBE_DUMP")) flags |= kFlagProbeDump;
    char v[16] = {};
    if (GetEnvironmentVariableA("ISAAC_TURBO_RENDER_EVERY", v, sizeof(v)) > 0) {
      renderEvery = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
    }
  }
  std::memset(b, 0, sizeof(*b));
  b->magic = kMagic;
  b->version = kLayoutVersion;
  b->pid = pid;
  b->flags = flags;
  b->render_every = renderEvery;
  b->render_request = renderRequest;
  b->render_done = renderRequest;
  b->status = kStatusLoading;
  b->game_update_min_us = 0;
  b->font_shader_flags = 0xFFFFFFFFu;
  std::snprintf(b->log_path, sizeof(b->log_path), "%s", gLogPath);
  gBlock = b;
  logf("control block: name=%s preexisting=%d configured=%d flags=0x%X render_every=%u", name,
       preexisting ? 1 : 0, configured ? 1 : 0, flags, renderEvery);
  return true;
}

// ---------------------------------------------------------------- 身份门

bool checkIdentity() {
  gImageBase = reinterpret_cast<const std::uint8_t*>(GetModuleHandleW(nullptr));
  if (!gImageBase) {
    setStatus(kStatusIdentityFailed, "ERROR:identity no_main_module");
    return false;
  }
  const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(gImageBase);
  if (dos->e_magic != IMAGE_DOS_SIGNATURE) {
    setStatus(kStatusIdentityFailed, "ERROR:identity bad_dos_header");
    return false;
  }
  const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS32*>(gImageBase + dos->e_lfanew);
  if (nt->Signature != IMAGE_NT_SIGNATURE) {
    setStatus(kStatusIdentityFailed, "ERROR:identity bad_nt_header");
    return false;
  }
  if (nt->FileHeader.TimeDateStamp != j460::kPeTimestamp ||
      nt->OptionalHeader.SizeOfImage != j460::kSizeOfImage) {
    setStatus(kStatusIdentityFailed, "ERROR:identity pe timestamp=0x%08lX size=0x%08lX want=0x%08X/0x%08X",
              static_cast<unsigned long>(nt->FileHeader.TimeDateStamp),
              static_cast<unsigned long>(nt->OptionalHeader.SizeOfImage), j460::kPeTimestamp,
              j460::kSizeOfImage);
    return false;
  }
  gImageSize = nt->OptionalHeader.SizeOfImage;
  logf("identity: module base=0x%08lX (file ImageBase 0x00400000)",
       static_cast<unsigned long>(reinterpret_cast<std::uintptr_t>(gImageBase)));
  for (const auto& t : j460::kFunctionTargets) {
    bool matches = static_cast<std::uint64_t>(t.rva) + t.prologue_bytes <= nt->OptionalHeader.SizeOfImage;
    if (matches) {
      // 掩码为 0 的字节含绝对地址，ASLR 重定位后与文件不同，跳过。
      for (std::size_t i = 0; i < t.prologue_bytes; ++i) {
        if (((gImageBase[t.rva + i] ^ t.prologue[i]) & t.mask[i]) != 0) {
          matches = false;
          break;
        }
      }
    }
    if (!matches) {
      char live[3 * 24 + 1] = {};
      const std::size_t n = t.prologue_bytes < 24 ? t.prologue_bytes : 24;
      for (std::size_t i = 0; i < n; ++i) std::snprintf(live + 3 * i, 4, "%02X ", gImageBase[t.rva + i]);
      logf("identity: %s @0x%08X live bytes: %s", t.name, t.rva, live);
      setStatus(kStatusIdentityFailed, "ERROR:identity prologue mismatch %s @0x%08X", t.name, t.rva);
      return false;
    }
    logf("identity: %s @0x%08X ok (%u bytes)", t.name, t.rva, static_cast<unsigned>(t.prologue_bytes));
  }
  gGlfwOffset = reinterpret_cast<const std::uint64_t*>(gImageBase + j460::kGlfwTimerOffsetRva);
  gGlfwFrequency = reinterpret_cast<const std::uint64_t*>(gImageBase + j460::kGlfwTimerFrequencyRva);
  gManagerPtr = reinterpret_cast<const std::uint8_t* const*>(gImageBase + j460::kManagerPtrRva);
  gClockUsable = *gGlfwFrequency == static_cast<std::uint64_t>(gQpf.QuadPart);
  logf("identity: glfw timer frequency=%llu qpf=%llu offset=%llu usable=%d",
       static_cast<unsigned long long>(*gGlfwFrequency), static_cast<unsigned long long>(gQpf.QuadPart),
       static_cast<unsigned long long>(*gGlfwOffset), gClockUsable ? 1 : 0);
  return true;
}

// ---------------------------------------------------------------- 时钟

double realGlfwTime() {
  LARGE_INTEGER q;
  QueryPerformanceCounter(&q);
  // 与 glfwGetTime 完全相同的公式：(QPC - _glfw.timer.offset) / frequency
  return static_cast<double>(q.QuadPart - static_cast<long long>(*gGlfwOffset)) /
         static_cast<double>(*gGlfwFrequency);
}

void syncClockMode(double real) {
  const bool want = gBlock && (gBlock->flags & kFlagVirtualClock) != 0 && gClockUsable;
  if (want && !gClock.enabled) {
    gClock.enable(real);
    logf("virtual clock ON base=%.6f real=%.6f", gClock.base, real);
  } else if (!want && gClock.enabled) {
    gClock.disable(real);
    logf("virtual clock OFF offset=%.6f real=%.6f ticks=%llu", gClock.offset, real,
         static_cast<unsigned long long>(gClock.ticks));
  }
}

bool gameWouldDraw() {
  // 渲染函数只在 Manager 迭代计数器为奇数（Manager::Update 刚跑过 Game::Update）或插值标志非零时绘制。
  if (!gDrawParityOk || !gManagerPtr || !*gManagerPtr) return true;
  const std::uint8_t* m = *gManagerPtr;
  const std::uint32_t counter = *reinterpret_cast<const std::uint32_t*>(m + j460::kManagerIterationCounterOffset);
  const std::uint8_t interp = *(m + j460::kManagerInterpolationFlagOffset);
  return (counter & 1u) != 0 || interp != 0;
}

// ---------------------------------------------------------------- 探针公共部分

// 把 [start, start+bytes) 里落在游戏映像内的 dword 以 RVA 形式打进一行（不做展开，只是候选返回地址）。
void logStackScan(const void* start, std::size_t bytes, const char* tag) {
  const auto* tib = reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
  const auto lo = reinterpret_cast<std::uintptr_t>(tib->StackLimit);
  const auto hi = reinterpret_cast<std::uintptr_t>(tib->StackBase);
  auto p = reinterpret_cast<std::uintptr_t>(start);
  if (p < lo || p >= hi) return;
  if (p + bytes > hi) bytes = hi - p;
  char line[512];
  int n = std::snprintf(line, sizeof(line), "%s stack rva:", tag);
  int hits = 0;
  const auto base = reinterpret_cast<std::uintptr_t>(gImageBase);
  for (std::size_t i = 0; i + 4 <= bytes && hits < 24 && n < static_cast<int>(sizeof(line)) - 16; i += 4) {
    const std::uint32_t v = *reinterpret_cast<const std::uint32_t*>(p + i);
    if (v >= base && v < base + gImageSize) {
      n += std::snprintf(line + n, sizeof(line) - n, " +%x:%x", static_cast<unsigned>(i), static_cast<unsigned>(v - base));
      ++hits;
    }
  }
  logf("%s", line);
}

using MiniDumpWriteDumpFn = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, MINIDUMP_TYPE, PMINIDUMP_EXCEPTION_INFORMATION,
                                          PMINIDUMP_USER_STREAM_INFORMATION, PMINIDUMP_CALLBACK_INFORMATION);

// 在异常发生前写一份包含私有读写页（堆）的 minidump。dbghelp 已被游戏自身加载。
bool writeProbeDump(const char* tag) {
  if (!gBlock || gBlock->probe_dumps_written >= kMaxProbeDumps || gLogDir[0] == '\0') return false;
  HMODULE dbg = GetModuleHandleA("dbghelp.dll");
  if (!dbg) dbg = LoadLibraryA("dbghelp.dll");
  if (!dbg) {
    logf("probe dump: dbghelp unavailable error=%lu", GetLastError());
    return false;
  }
  const auto fn = reinterpret_cast<MiniDumpWriteDumpFn>(GetProcAddress(dbg, "MiniDumpWriteDump"));
  if (!fn) {
    logf("probe dump: MiniDumpWriteDump missing");
    return false;
  }
  char path[MAX_PATH];
  std::snprintf(path, sizeof(path), "%s\\probe-%lu-%s-%u.dmp", gLogDir, static_cast<unsigned long>(GetCurrentProcessId()),
                tag, static_cast<unsigned>(gBlock->probe_dumps_written + 1));
  HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    logf("probe dump: cannot create %s error=%lu", path, GetLastError());
    return false;
  }
  const auto type = static_cast<MINIDUMP_TYPE>(MiniDumpWithPrivateReadWriteMemory | MiniDumpWithDataSegs |
                                               MiniDumpWithHandleData | MiniDumpWithThreadInfo |
                                               MiniDumpWithIndirectlyReferencedMemory | MiniDumpWithUnloadedModules |
                                               MiniDumpWithFullMemoryInfo);
  const BOOL ok = fn(GetCurrentProcess(), GetCurrentProcessId(), h, type, nullptr, nullptr, nullptr);
  const DWORD err = ok ? 0 : GetLastError();
  CloseHandle(h);
  if (ok) gBlock->probe_dumps_written++;
  logf("probe dump %s: %s error=%lu", ok ? "written" : "FAILED", path, err);
  return ok != 0;
}

void recordFailText(const char* text) {
  if (!gBlock) return;
  std::strncpy(gBlock->last_fail_text, text, sizeof(gBlock->last_fail_text) - 1);
  gBlock->last_fail_text[sizeof(gBlock->last_fail_text) - 1] = '\0';
}

bool shouldLogOccurrence(std::uint64_t n) { return n <= 20 || n % 100 == 0; }

// Filesys::ResolvePath 原始相对归档路径与返回值；不把可选文件缺失当作游戏故障。
void* __fastcall hookSearchPath(void* self, void* edx, const char* path, unsigned char* inArchive) {
  void* result = gOrigSearchPath(self, edx, path, inArchive);
  gLastSearchDirectory = result;
  gLastSearchInArchive = *inArchive;
  return result;
}

char* __fastcall hookResolvePath(void* self, void* edx, const char* path) {
  const std::size_t n = path ? std::strlen(path) : 0;
  const bool absolutePath = n > 2 && path[1] == ':' && (path[2] == '/' || path[2] == '\\');
  const char* lookup = path;
  char absolute[MAX_PATH * 2] = {};
  int containerSlot = -1;
  if (gArchivePathFix && path && !absolutePath) {
    // Only mounted container identity pointers, including the pending mount slot.
    // Resolve them as OS files before entering the relative asset/archive search.
    const auto* slots = reinterpret_cast<const std::uint32_t*>(gImageBase + j460::kArchiveSlotTableRva);
    for (unsigned i = 0; i < 32; ++i) {
      if (reinterpret_cast<const char*>(slots[i * 2]) != path) continue;
      const DWORD length = GetFullPathNameA(path, sizeof(absolute), absolute, nullptr);
      // Nested containers (notably resources/secret.a) exist only inside an archive.
      // They must retain native archive lookup instead of becoming nonexistent OS paths.
      const DWORD attributes = length > 0 && length < sizeof(absolute)
          ? GetFileAttributesA(absolute) : INVALID_FILE_ATTRIBUTES;
      if (attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        lookup = absolute;
        containerSlot = static_cast<int>(i);
      }
      break;
    }
  }
  // The original absolute-path branch performs access checking and native allocation.
  // No retry, fabricated stream, or ownership transfer to the hook's stack buffer.
  char* result = gOrigResolvePath(self, edx, lookup);
  if (containerSlot >= 0) {
    const LONG count = InterlockedIncrement(&gArchiveResolveRecoveries);
    if (!result || count <= 3) {
      const DWORD error = GetLastError();
      int e = 0, de = 0;
      if (gGetErrno) gGetErrno(&e);
      if (gGetDosErrno) gGetDosErrno(&de);
      logf("archive_resolve_absolute count=%ld slot=%d success=%d path=%s absolute=%s",
           count, containerSlot, result ? 1 : 0, path, absolute);
      if (gSetErrno) gSetErrno(e);
      if (gSetDosErrno) gSetDosErrno(static_cast<unsigned long>(de));
      SetLastError(error);
    }
  }
  const DWORD lastError = GetLastError();
  int err = 0, doserr = 0;
  if (gGetErrno) gGetErrno(&err);
  if (gGetDosErrno) gGetDosErrno(&doserr);
  if (gResolverProbe && n >= 2 && path[n - 2] == '.' && (path[n - 1] == 'a' || path[n - 1] == 'A')) {
    const LONG count = InterlockedIncrement(&gArchiveResolveCalls);
    if (!result || count <= 8) {
      const auto* fs = static_cast<const std::uint32_t*>(self);
      const auto* begin = reinterpret_cast<const std::uint32_t*>(fs[2]);
      const auto* end = reinterpret_cast<const std::uint32_t*>(fs[3]);
      std::uint32_t hash = 5381;
      for (const unsigned char* p = reinterpret_cast<const unsigned char*>(path); *p; ++p) {
        unsigned char c = *p;
        if (c >= 'A' && c <= 'Z') c += 32;
        if (c == '\\') c = '/';
        hash = hash * 33 + c;
      }
      logf("resolve_archive #%ld self=%p hash=%08X dirs=%ld result=%p path=%s resolved=%s",
           count, self, hash, static_cast<long>(end - begin), result, path, result ? result : "(null)");
      if (!result) {
        logf("resolve_search returned=%p inArchive=%u native_key=%08X native_copy=%s",
             gLastSearchDirectory, gLastSearchInArchive, gLastSearchKey, gLastSearchCopy);
        for (const auto* p = begin; p != end; ++p) {
          const auto* dir = reinterpret_cast<const std::uint32_t*>(*p);
          const std::uint32_t head = dir[1];
          std::uint32_t node = *reinterpret_cast<const std::uint32_t*>(head + 4), value = 0;
          unsigned steps = 0;
          while (node != head && steps++ < 64) {
            const auto* words = reinterpret_cast<const std::uint32_t*>(node);
            logf("resolve_tree step=%u node=%08X nil=%u key=%08X", steps, node,
                 *(reinterpret_cast<const unsigned char*>(node) + 0xD), words[4]);
            if (words[4] == hash) { value = words[5]; break; }
            node = words[hash < words[4] ? 0 : 2];
          }
          logf("resolve_dir index=%ld dir=%p entries=%u hash_node=%08X value=%08X prefix=%s value_path=%s",
               static_cast<long>(p - begin), dir, dir[2], node, value, reinterpret_cast<const char*>(dir[3]),
               value ? reinterpret_cast<const char*>(value) : "(null)");
        }
        logStackScan(reinterpret_cast<const std::uint32_t*>(&path) - 1, 0x200, "resolve_archive");
      }
    }
  }
  if (gSetErrno) gSetErrno(err);
  if (gSetDosErrno) gSetDosErrno(static_cast<unsigned long>(doserr));
  SetLastError(lastError);
  return result;
}

// ---------------------------------------------------------------- Hook：shader 入栈（A 类取证）

void onPushFailure(const char* name, const void* stackStart) {
  gBlock->push_failures++;
  const std::uint32_t hash = shaderNameHash(name);
  const RegistryLookup l = lookupShaderInRegistry(gImageBase + j460::kShaderRegistryMapRva, hash);
  const ShaderStackState s = readShaderStack(gImageBase + j460::kShaderStackRva);
  const std::uint32_t current = *reinterpret_cast<const std::uint32_t*>(gImageBase + j460::kShaderCurrentRva);
  gBlock->registry_size_seen = l.size;
  gBlock->font_shader_flags = l.found ? l.flags : 0xFFFFFFFFu;
  char text[256];
  std::snprintf(text, sizeof(text),
                "push_fail #%llu name=%s hash=%08X registry_size=%u found=%d node=%08X object=%08X flags=%08X steps=%u "
                "stack{size=%u offset=%u mapsize=%u} current=%08X thread=%lu",
                static_cast<unsigned long long>(gBlock->push_failures), name ? name : "(null)", hash, l.size,
                l.found ? 1 : 0, l.node, l.object, l.flags, l.steps, s.size, s.offset, s.mapsize, current,
                GetCurrentThreadId());
  recordFailText(text);
  if (shouldLogOccurrence(gBlock->push_failures)) {
    logf("%s", text);
    logStackScan(stackStart, 0x400, "push_fail");
  }
  if ((gBlock->flags & kFlagProbeDump) != 0) writeProbeDump("push-fail");
}

int __stdcall hookPushShader(const char* name) {
  const int r = gOrigPushShader(name);
  if (gBlock) {
    gBlock->push_calls++;
    // &name 是本帧的第一个栈参数，其下方一个 dword 是调用者的返回地址（进入字体函数等）。
    if (r == 0) onPushFailure(name, reinterpret_cast<const std::uint32_t*>(&name) - 1);
  }
  return r;
}

// ---------------------------------------------------------------- Hook：字体绘制守卫（A 类缓解）

void onFontGuardSkip(const RegistryLookup& l) {
  gBlock->font_skipped++;
  const ShaderStackState s = readShaderStack(gImageBase + j460::kShaderStackRva);
  char text[256];
  std::snprintf(text, sizeof(text),
                "font_guard skip #%llu registry_size=%u found=%d object=%08X flags=%08X stack{size=%u offset=%u} thread=%lu",
                static_cast<unsigned long long>(gBlock->font_skipped), l.size, l.found ? 1 : 0, l.object, l.flags,
                s.size, s.offset, GetCurrentThreadId());
  recordFailText(text);
  if (shouldLogOccurrence(gBlock->font_skipped)) {
    int marker = 0;
    logf("%s", text);
    logStackScan(&marker, 0x400, "font_guard");
  }
  if ((gBlock->flags & kFlagProbeDump) != 0) writeProbeDump("font-guard");
}

// ---------------------------------------------------------------- Hook：KAGE 文件打开（B 类取证 + 缓解）

// 现有重试策略只覆盖 DOS 绝对路径 + .a，不等于识别了所有归档容器。
// 实机槽表存 resources/packed/*.a 相对路径；其解析失败不在此策略的覆盖范围。
bool looksLikeArchive(const char* path) {
  if (!path) return false;
  const std::size_t n = std::strlen(path);
  if (n < 4 || path[1] != ':' || (path[2] != '/' && path[2] != '\\')) return false;
  return path[n - 2] == '.' && (path[n - 1] == 'a' || path[n - 1] == 'A');
}

unsigned char __stdcall hookFileOpen(const char* path, void** out) {
  unsigned char r = gOrigFileOpen(path, out);
  if (!gBlock || out == nullptr) return r;  // out==NULL 只是存在性检查，失败是正常结果
  gBlock->file_open_calls++;
  if (r != 0) return r;
  const DWORD lastError = GetLastError();
  int err = 0, doserr = 0;
  if (gGetErrno) gGetErrno(&err);
  if (gGetDosErrno) gGetDosErrno(&doserr);
  gBlock->file_open_failures++;
  const std::uint32_t slots = *reinterpret_cast<const std::uint32_t*>(gImageBase + j460::kArchiveSlotCountRva);
  char text[256];
  std::snprintf(text, sizeof(text), "file_open_fail #%llu errno=%d doserrno=%d lastError=%lu archive_slots=%u thread=%lu path=%s",
                static_cast<unsigned long long>(gBlock->file_open_failures), err, doserr, lastError, slots,
                GetCurrentThreadId(), path ? path : "(null)");
  recordFailText(text);
  if (shouldLogOccurrence(gBlock->file_open_failures)) {
    logf("%s", text);
    logStackScan(reinterpret_cast<const std::uint32_t*>(&path) - 1, 0x400, "file_open_fail");
  }
  if ((gBlock->flags & kFlagProbeDump) != 0 && looksLikeArchive(path)) writeProbeDump("file-open-fail");
  if ((gBlock->flags & kFlagFileRetry) != 0 && looksLikeArchive(path)) {
    for (int attempt = 1; attempt <= 5 && r == 0; ++attempt) {
      Sleep(static_cast<DWORD>(20 * attempt));
      gBlock->file_open_retries++;
      r = gOrigFileOpen(path, out);
      if (r != 0) {
        gBlock->file_open_retry_ok++;
        logf("file_open retry ok attempt=%d path=%s", attempt, path);
      } else if (gGetErrno) {
        gGetErrno(&err);
        logf("file_open retry fail attempt=%d errno=%d lastError=%lu path=%s", attempt, err, GetLastError(), path);
      }
    }
  }
  return r;
}

// ---------------------------------------------------------------- Hook：ucrtbase!_access（B 类候选上游；记录过滤前证据）
//
// 路径解析 FUN_00a17180（RVA 0x617180）对绝对路径只做一件事：复制字符串后 _access(path, 0)，返回 -1 就整个返回空；
// NULL 路径和遗留 errno 不能证明这条分支失败；必须捕获本次调用的实际参数/返回值。

void diagnoseAccessFailure(const char* path, int mode, int err, int doserr, DWORD lastError) {
  wchar_t wide[MAX_PATH * 2] = {};
  const int n = MultiByteToWideChar(CP_ACP, 0, path, -1, wide, MAX_PATH * 2 - 1);
  DWORD attrs = INVALID_FILE_ATTRIBUTES, attrsError = 0, openError = 0;
  int opened = -1;
  if (n > 0) {
    attrs = GetFileAttributesW(wide);
    attrsError = attrs == INVALID_FILE_ATTRIBUTES ? GetLastError() : 0;
    HANDLE h = CreateFileW(wide, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    opened = h != INVALID_HANDLE_VALUE ? 1 : 0;
    openError = opened ? 0 : GetLastError();
    if (opened) CloseHandle(h);
  }
  DWORD handles = 0;
  GetProcessHandleCount(GetCurrentProcess(), &handles);
  char text[256];
  std::snprintf(text, sizeof(text),
                "access_fail #%llu mode=%d errno=%d doserrno=%d lastError=%lu recheck{attrs=%08lX err=%lu open=%d err=%lu} "
                "handles=%lu thread=%lu path=%s",
                static_cast<unsigned long long>(gBlock->access_failures), mode, err, doserr, lastError, attrs, attrsError,
                opened, openError, handles, GetCurrentThreadId(), path ? path : "(null)");
  recordFailText(text);
  logf("%s", text);
}

int __cdecl hookCrtAccess(const char* path, int mode) {
  int r = gOrigAccess(path, mode);
  DWORD lastError = GetLastError();
  int err = 0, doserr = 0;
  if (gGetErrno) gGetErrno(&err);
  if (gGetDosErrno) gGetDosErrno(&doserr);
  const LONG total = InterlockedIncrement(&gAccessTotal);
  const LONG failed = r != 0 ? InterlockedIncrement(&gAccessFailed) : 0;
  const bool archive = looksLikeArchive(path);
  // 有界采样；原 access_calls 仍只统计绝对 .a，避免悄悄改变共享内存字段语义。
  if (total <= 8 || (failed != 0 && shouldLogOccurrence(failed))) {
    logf("access_call total=%ld failed=%ld result=%d mode=%d archive=%d errno=%d doserrno=%d lastError=%lu path=%s",
         total, failed, r, mode, archive ? 1 : 0, err, doserr, lastError, path ? path : "(null)");
    if (gSetErrno) gSetErrno(err);
    if (gSetDosErrno) gSetDosErrno(static_cast<unsigned long>(doserr));
    SetLastError(lastError);
  }
  if (!gBlock || !archive) return r;
  gBlock->access_calls++;
  if (r == 0) return r;
  gBlock->access_failures++;
  diagnoseAccessFailure(path, mode, err, doserr, lastError);
  logStackScan(reinterpret_cast<const std::uint32_t*>(&path) - 1, 0x400, "access_fail");
  if ((gBlock->flags & kFlagProbeDump) != 0) writeProbeDump("access-fail");
  if ((gBlock->flags & kFlagFileRetry) != 0) {
    for (int attempt = 1; attempt <= 5 && r != 0; ++attempt) {
      Sleep(static_cast<DWORD>(10 * attempt));
      gBlock->access_retries++;
      r = gOrigAccess(path, mode);
      lastError = GetLastError();
      if (gGetErrno) gGetErrno(&err);
      if (gGetDosErrno) gGetDosErrno(&doserr);
      if (r == 0) {
        gBlock->access_retry_ok++;
        logf("access retry ok attempt=%d path=%s", attempt, path);
      } else {
        int e2 = 0;
        if (gGetErrno) gGetErrno(&e2);
        logf("access retry fail attempt=%d errno=%d path=%s", attempt, e2, path);
      }
    }
  }
  // 日志/复查/转储不向游戏泄漏诊断错误；重试模式保留最后一次实际 _access 的状态。
  if (gSetErrno) gSetErrno(err);
  if (gSetDosErrno) gSetDosErrno(static_cast<unsigned long>(doserr));
  SetLastError(lastError);
  return r;
}

}  // namespace

// ---------------------------------------------------------------- Hook：glfwGetTime（返回 double 在 XMM0）

extern "C" {
double g_turbo_now_value = 0.0;
void* g_orig_font_draw = nullptr;
void* g_orig_search_miss = nullptr;

void __cdecl turbo_search_miss_record(const char* copy, std::uint32_t key) {
  std::snprintf(gLastSearchCopy, sizeof(gLastSearchCopy), "%s", copy ? copy : "(null)");
  gLastSearchKey = key;
}

__attribute__((naked)) void turbo_hook_search_miss(void) {
  __asm__ volatile(
      "pushfl\n\t"
      "pushal\n\t"
      "pushl 8(%ebp)\n\t"
      "pushl %ebx\n\t"
      "call _turbo_search_miss_record\n\t"
      "addl $8, %esp\n\t"
      "popal\n\t"
      "popfl\n\t"
      "jmp *_g_orig_search_miss\n\t");
}

void __cdecl turbo_clock_read(void) {
  const double real = realGlfwTime();
  syncClockMode(real);
  g_turbo_now_value = gClock.read(real);
  if (gBlock) {
    gBlock->clock_reads++;
    gBlock->clock_value = g_turbo_now_value;
  }
}

// 原函数无参数、结果在 XMM0（LTCG 自定义约定，6 处调用点都紧接 movsd/cvtsd2ss xmm0）。
// C 无法表达"返回 double 到 XMM0"，用 naked 桩：先调 C 计算，再把结果装入 XMM0。
__attribute__((naked)) void turbo_hook_glfwGetTime(void) {
  __asm__ volatile(
      "call _turbo_clock_read\n\t"
      "movsd _g_turbo_now_value, %xmm0\n\t"
      "ret\n\t");
}

// 字体守卫检查：返回非 0 = 放行原函数；0 = 跳过本次绘制（守卫关闭时恒放行）。
int __cdecl turbo_font_guard_check(void) {
  if (!gBlock) return 1;
  gBlock->font_calls++;
  if ((gBlock->flags & kFlagFontGuard) == 0) return 1;
  const RegistryLookup l = lookupShaderInRegistry(gImageBase + j460::kShaderRegistryMapRva, kColorTextureShaderHash);
  gBlock->registry_size_seen = l.size;
  gBlock->font_shader_flags = l.found ? l.flags : 0xFFFFFFFFu;
  if (l.found && l.object != 0 && (l.flags & 1u) != 0) return 1;
  onFontGuardSkip(l);
  return 0;
}

// 字体绘制 FUN_00a1bd80 是 thiscall + 9 个栈参数（ret 0x24），另外经 XMM2/XMM3 传两个浮点。
// naked 桩保存 ECX/EDX/XMM0-3，调 C 检查；放行则跳到 MinHook 蹦床，否则按原约定 ret 0x24。
__attribute__((naked)) void turbo_hook_fontDraw(void) {
  __asm__ volatile(
      "pushl %ecx\n\t"
      "pushl %edx\n\t"
      "subl $64, %esp\n\t"
      "movups %xmm0, (%esp)\n\t"
      "movups %xmm1, 16(%esp)\n\t"
      "movups %xmm2, 32(%esp)\n\t"
      "movups %xmm3, 48(%esp)\n\t"
      "call _turbo_font_guard_check\n\t"
      "movups (%esp), %xmm0\n\t"
      "movups 16(%esp), %xmm1\n\t"
      "movups 32(%esp), %xmm2\n\t"
      "movups 48(%esp), %xmm3\n\t"
      "addl $64, %esp\n\t"
      "popl %edx\n\t"
      "popl %ecx\n\t"
      "testl %eax, %eax\n\t"
      "jz 1f\n\t"
      "jmp *_g_orig_font_draw\n\t"
      "1:\n\t"
      "ret $0x24\n\t");
}
}

namespace {

// ---------------------------------------------------------------- Hook：Manager::Update

void __stdcall hookManagerUpdate(char parameter) {
  gOrigManagerUpdate(parameter);
  gClock.tick();
  if (gManagerPtr && *gManagerPtr) {
    const std::uint32_t c =
        *reinterpret_cast<const std::uint32_t*>(*gManagerPtr + j460::kManagerIterationCounterOffset);
    if (gCounterChecks > 0 && c != gLastCounterSeen + 1) ++gCounterMismatches;
    gLastCounterSeen = c;
    ++gCounterChecks;
    // 在线等待/退出路径会不加计数就返回；离线单机应恒为每次 +1。允许 1/64 的偶发失配。
    gDrawParityOk = gCounterChecks >= 16 && gCounterMismatches * 64 <= gCounterChecks;
  }
  if (gBlock) {
    gBlock->manager_update_calls++;
    gBlock->virtual_ticks = gClock.ticks;
    gBlock->wall_us = wallMicros();
    gBlock->draw_parity_ok = gDrawParityOk ? 1u : 0u;
    gBlock->counter_checks = gCounterChecks;
    gBlock->counter_mismatches = gCounterMismatches;
  }
}

// ---------------------------------------------------------------- Hook：Game::Update

void __fastcall hookGameUpdate(void* self, void* edx) {
  LARGE_INTEGER a, b;
  QueryPerformanceCounter(&a);
  gOrigGameUpdate(self, edx);
  QueryPerformanceCounter(&b);
  const std::uint64_t us = static_cast<std::uint64_t>(b.QuadPart - a.QuadPart) * 1000000ull /
                           static_cast<std::uint64_t>(gQpf.QuadPart);
  if (!gBlock) return;
  gBlock->game_update_calls++;
  gBlock->game_update_total_us += us;
  gBlock->game_update_last_us = us;
  if (us > gBlock->game_update_max_us) gBlock->game_update_max_us = us;
  if (us < gGameUpdateMinUs) {
    gGameUpdateMinUs = us;
    gBlock->game_update_min_us = us;
  }
  if (us < kFastGameUpdateMicros) {
    gBlock->game_update_fast_calls++;
    gBlock->game_update_fast_total_us += us;
  }
}

// ---------------------------------------------------------------- Hook：整帧渲染

void __fastcall hookRender(void* ecx, void* edx) {
  const std::uint32_t flags = gBlock ? gBlock->flags : 0;
  bool render = true;
  if (gBlock && (flags & kFlagSkipRender) != 0) {
    render = false;
    if (gameWouldDraw()) {
      ++gRenderOpportunities;
      gBlock->render_opportunities = gRenderOpportunities;
      if (gBlock->render_request != gBlock->render_done) {
        render = true;
      } else if (gBlock->render_every != 0 && gRenderOpportunities % gBlock->render_every == 0) {
        render = true;
      }
    }
  }
  if (render) {
    gOrigRender(ecx, edx);
    if (gBlock) {
      gBlock->render_calls++;
      gBlock->render_done = gBlock->render_request;
    }
  } else {
    gBlock->render_skipped++;
  }
}

// ---------------------------------------------------------------- 安装

// 仅当前worker的可选NVIDIA录屏/照片叠加层；不拦截图形驱动、Steam或其他进程。
LONG NTAPI hookLdrLoadDll(PWSTR search, ULONG* flags, UNICODE_STRING* name, HMODULE* module) {
  unsigned count = name->Length / sizeof(wchar_t), start = count;
  while (start && name->Buffer[start - 1] != L'\\' && name->Buffer[start - 1] != L'/') --start;
  wchar_t base[32] = {};
  if (count - start < 32) {
    for (unsigned i = start; i < count; ++i) {
      wchar_t c = name->Buffer[i];
      base[i - start] = c >= L'A' && c <= L'Z' ? c + 32 : c;
    }
    if (std::wcscmp(base, L"nvcamera32.dll") == 0 || std::wcscmp(base, L"nvspcap.dll") == 0) {
      *module = nullptr;
      logf("capture_overlay excluded: %ls", base);
      return static_cast<LONG>(0xC0000135u);  // STATUS_DLL_NOT_FOUND
    }
  }
  return gOrigLdrLoadDll(search, flags, name, module);
}

bool installHooks() {
  MH_STATUS st = MH_Initialize();
  if (st != MH_OK && st != MH_ERROR_ALREADY_INITIALIZED) {
    setStatus(kStatusHookFailed, "ERROR:hook MH_Initialize=%d", static_cast<int>(st));
    return false;
  }
  struct Spec {
    const char* name;
    std::uint32_t rva;
    void* detour;
    void** original;
    std::uint32_t bit;
  };
  const Spec specs[] = {
      {"glfw_get_time", j460::kGlfwGetTimeRva, reinterpret_cast<void*>(&turbo_hook_glfwGetTime),
       &gOrigGlfwGetTime, kHookGlfwGetTime},
      {"manager_update", j460::kManagerUpdateRva, reinterpret_cast<void*>(&hookManagerUpdate),
       reinterpret_cast<void**>(&gOrigManagerUpdate), kHookManagerUpdate},
      {"game_update", j460::kGameUpdateRva, reinterpret_cast<void*>(&hookGameUpdate),
       reinterpret_cast<void**>(&gOrigGameUpdate), kHookGameUpdate},
      {"render_frame", j460::kRenderFrameRva, reinterpret_cast<void*>(&hookRender),
       reinterpret_cast<void**>(&gOrigRender), kHookRender},
      {"push_shader", j460::kPushShaderRva, reinterpret_cast<void*>(&hookPushShader),
       reinterpret_cast<void**>(&gOrigPushShader), kHookPushShader},
      {"font_draw", j460::kFontDrawRva, reinterpret_cast<void*>(&turbo_hook_fontDraw), &g_orig_font_draw,
       kHookFontDraw},
      {"file_open_plain", j460::kFileOpenPlainRva, reinterpret_cast<void*>(&hookFileOpen),
       reinterpret_cast<void**>(&gOrigFileOpen), kHookFileOpen},
      {"resolve_path", j460::kResolvePathRva, reinterpret_cast<void*>(&hookResolvePath),
       reinterpret_cast<void**>(&gOrigResolvePath), kHookResolvePath},
      {"search_path", j460::kSearchPathRva, reinterpret_cast<void*>(&hookSearchPath),
       reinterpret_cast<void**>(&gOrigSearchPath), kHookSearchPath},
      {"search_miss", j460::kSearchMissRva, reinterpret_cast<void*>(&turbo_hook_search_miss),
       &g_orig_search_miss, kHookSearchMiss},
  };
  gResolverProbe = envFlag("ISAAC_TURBO_RESOLVER_PROBE");
  gArchivePathFix = envFlag("ISAAC_TURBO_ARCHIVE_PATH_FIX");
  std::uint32_t mask = 0;
  for (const auto& s : specs) {
    if (s.bit == kHookResolvePath && !(gResolverProbe || gArchivePathFix)) continue;
    if ((s.bit & (kHookSearchPath | kHookSearchMiss)) && !gResolverProbe) continue;
    void* target = const_cast<std::uint8_t*>(gImageBase) + s.rva;
    st = MH_CreateHook(target, s.detour, s.original);
    if (st != MH_OK) {
      setStatus(kStatusHookFailed, "ERROR:hook create %s=%d", s.name, static_cast<int>(st));
      return false;
    }
    mask |= s.bit;
  }
  if (envFlag("ISAAC_TURBO_DISABLE_CAPTURE_OVERLAYS")) {
    void* load = reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("ntdll.dll"), "LdrLoadDll"));
    st = MH_CreateHook(load, reinterpret_cast<void*>(&hookLdrLoadDll), reinterpret_cast<void**>(&gOrigLdrLoadDll));
    if (st != MH_OK) {
      setStatus(kStatusHookFailed, "ERROR:hook LdrLoadDll=%d", static_cast<int>(st));
      return false;
    }
    mask |= kHookCaptureOverlay;
  }
  if (const HMODULE ucrt = GetModuleHandleA("ucrtbase.dll")) {
    if (void* access = reinterpret_cast<void*>(GetProcAddress(ucrt, "_access"))) {
      st = MH_CreateHook(access, reinterpret_cast<void*>(&hookCrtAccess), reinterpret_cast<void**>(&gOrigAccess));
      if (st != MH_OK) {
        setStatus(kStatusHookFailed, "ERROR:hook create ucrtbase!_access=%d", static_cast<int>(st));
        return false;
      }
      mask |= kHookCrtAccess;
    } else {
      logf("probe: ucrtbase!_access not exported, B-class access probe disabled");
    }
  }
  // 一次性启用，避免出现只装了一半 Hook 的状态。
  st = MH_EnableHook(MH_ALL_HOOKS);
  if (st != MH_OK) {
    setStatus(kStatusHookFailed, "ERROR:hook enable=%d", static_cast<int>(st));
    return false;
  }
  if (gBlock) gBlock->hooks_mask = mask;
  logf("hooks installed mask=0x%X", mask);
  return true;
}

void loadCrtErrno() {
  // 游戏用 ucrtbase 的 CRT；本 DLL 的 CRT 是另一份，errno 必须从游戏那份读。
  const HMODULE ucrt = GetModuleHandleA("ucrtbase.dll");
  if (!ucrt) {
    logf("probe: ucrtbase.dll not loaded, errno unavailable");
    return;
  }
  gGetErrno = reinterpret_cast<GetErrnoFn>(GetProcAddress(ucrt, "_get_errno"));
  gGetDosErrno = reinterpret_cast<GetErrnoFn>(GetProcAddress(ucrt, "_get_doserrno"));
  gSetErrno = reinterpret_cast<SetErrnoFn>(GetProcAddress(ucrt, "_set_errno"));
  gSetDosErrno = reinterpret_cast<SetDosErrnoFn>(GetProcAddress(ucrt, "_set_doserrno"));
  logf("probe: ucrt errno getters %s", gGetErrno && gGetDosErrno ? "ok" : "missing");
}

int initialize() {
  if (InterlockedCompareExchange(&gInitState, 1, 0) != 0) {
    while (gInitState != 2) Sleep(1);
    return static_cast<int>(gFinalStatus);
  }
  QueryPerformanceFrequency(&gQpf);
  QueryPerformanceCounter(&gQpc0);
  const DWORD pid = GetCurrentProcessId();
  openLog(pid);
  char exe[MAX_PATH] = {};
  GetModuleFileNameA(nullptr, exe, sizeof(exe));
  logf("isaac_turbo v%u attach pid=%lu exe=%s", kDllVersion, static_cast<unsigned long>(pid), exe);
  Status result = kStatusActive;
  if (!openControlBlock(pid)) {
    setStatus(kStatusShmFailed, "ERROR:shm control block unavailable");
    result = kStatusShmFailed;
  } else if (!checkIdentity()) {
    result = kStatusIdentityFailed;
  } else {
    loadCrtErrno();
    if (!installHooks()) {
      result = kStatusHookFailed;
    } else {
      setStatus(kStatusActive, "OK active flags=0x%X clock_usable=%d", gBlock->flags, gClockUsable ? 1 : 0);
    }
  }
  InterlockedExchange(&gFinalStatus, static_cast<LONG>(result));
  InterlockedExchange(&gInitState, 2);
  return static_cast<int>(result);
}

DWORD WINAPI initThread(LPVOID) {
  initialize();
  return 0;
}

}  // namespace

extern "C" __declspec(dllexport) std::uint32_t IsaacTurboVersion() { return kDllVersion; }
extern "C" __declspec(dllexport) const char* IsaacTurboStatus() { return gStatus; }
extern "C" __declspec(dllexport) int IsaacTurboInit() { return initialize(); }
extern "C" __declspec(dllexport) void* IsaacTurboControlBlock() { return gBlock; }

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID) {
  if (reason == DLL_PROCESS_ATTACH) {
    gSelf = instance;
    DisableThreadLibraryCalls(instance);
    HANDLE t = CreateThread(nullptr, 0, &initThread, nullptr, 0, nullptr);
    if (t) CloseHandle(t);
  }
  return TRUE;
}
