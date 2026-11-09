// rr_mt.dll —— 整局录制会话用的最小 DLL（Repentance v1.7.8a.J746 离线 GOG x86；DESIGN_BRIEF §7 E1/E2/E7）。
// 来源：experiments/run-recorder/mt-probe/src/rr_mt.cpp（MT_PROBE.md 里 E 组用的"最小版"），去掉了探针的
// rr_mt2/rr_mt3 内容（OpenAL 钩子、MT 取数 trace）。只做两件事：
//   1. 挂 userenv!GetUserProfileDirectoryA，把游戏的"用户目录"改成环境变量 RR_MT_PROFILE 指定的目录
//      （存档落到 <RR_MT_PROFILE>\Documents\My Games\Binding of Isaac Repentance (Galaxy)\）。
//   2. 导出 Lua C 函数（mod 用 package.loadlib 取得）：
//        RRMT_Seed(seed)  -> 调用引擎自己的 GlobalMT_Seed（RVA 0x2BA260，seed 在 ECX），返回 (cursor, hash)
//        RRMT_State()     -> 只读：返回 (cursor, FNV-1a32(624 个状态字 + cursor))，不推进 MT
//        RRMT_Info()      -> (mt_ready 0/1, profile 调用次数, reseed 次数, PE 时间戳, SizeOfImage)
//      MT 身份门没通过时 RRMT_Seed/RRMT_State 不返回任何值（Lua 侧得到 nil），mod 记 mt.available=false。
//   4.（phase8-mouse）环境变量 RR_MT_AFFINITY=<掩码> 时启动即把进程限制到这些 CPU（回放的确定性，见 initThread）。
//   3.（phase8-mouse，只给回放用）环境变量 RR_MT_MOUSE=1 时导出 RRMT_SetMouse/RRMT_GetMouse：直接写 GLFW 的虚拟光标位置与
//      鼠标按键（引擎的鼠标只来自这里），并子类化窗口过程丢掉真实鼠标消息。录制会话不设这个变量，行为不变。见文件后部。
//   5.（phase10-mods-console，只在录制会话 --allow-console 时）环境变量 RR_MT_CONSOLE=1 时挂引擎的"控制台提交输入"函数
//      Console::SubmitInput（RVA 0x258AB0，thiscall + 1 个哑栈参数，ret 4；玩家在调试控制台敲回车时调用，内部把输入行
//      推进历史并调用命令执行函数 RVA 0x25E160；Lua 的 Isaac.ExecuteCommand 走 RVA 0x4015C0 直接调 0x25E160，不经过这里），
//      把输入行（this+0x1C 的 std::string）排队，mod 用 RRMT_ConsoleTake() 取走写成 ev cmd（src = console）。身份门 +
//      3 处代码签名不符时只记日志、不装（不影响其它功能）。
// 默认没有 turbo / 虚拟时钟 / 跳渲染 / GPU gate / 文件重试 / OpenAL 等任何其他钩子。
// 唯一例外（phase4-turbo，只给回放验证用）：环境变量 RR_MT_TURBO=1 时另外安装 rr_turbo.cpp 的虚拟时钟
// （glfwGetTime + Manager::Update 两个钩子），RR_MT_TURBO_GPU_GATE=1 时再加 GPU 提交门控；录制会话不设这两个变量，
// 进程里没有这些钩子。RR_MT_TURBO=1 但不是 J746 或安装失败 → 控制块 status=HookFailed（启动器结束进程），不会悄悄按原速跑。
//
// 与探针版的唯一行为差异（E7）：探针版身份门不过就不装存档重定向、挂起启动器随即结束进程；这里存档重定向
// 与 EXE 版本无关，总是安装（控制块 status=Active，启动器照常恢复主线程），身份门只决定 MT 导出是否可用，
// 这样 EXE 不是 J746 时游戏照常运行、录制照常进行，只是没有 MT reseed。
// 测试开关：环境变量 RR_MT_TEST_IDENTITY_FAIL=1 时假装身份门不通过（只用于验证 E7 的降级路径）。
//
// 挂起式启动（rr_launch.exe --launch）要求：DLL 建与 isaac_turbo 相同布局的命名控制块 "IsaacTurbo.<pid>"，
// 只填 magic/version/pid/status/status_text/hooks_mask/log_path；启动器等 status 离开 loading 才恢复主线程，
// 所以存档路径钩子在游戏第一次查询用户目录之前就绪。
// 身份门：PE TimeDateStamp 0x624393F6、SizeOfImage 8540160（J746），并核对 0x2BA260 处
// `mov [MT_STATE], ecx` 的立即数等于 base+0x7CA070、0x2BA2A0 处 `mov eax, [CURSOR]` 等于 base+0x79EE34。
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>

#include "MinHook.h"
#include "rr_turbo.hpp"
#include "turbo_shared.hpp"

namespace {

using namespace isaac_turbo;

constexpr std::uint32_t kRvaSeed = 0x2BA260;
constexpr std::uint32_t kRvaNext = 0x2BA2A0;
constexpr std::uint32_t kRvaState = 0x7CA070;
constexpr std::uint32_t kRvaCursor = 0x79EE34;
constexpr std::uint32_t kHookProfile = 1u << 12;  // 与 turbo 的 kHookWorkerProfile 同位
constexpr std::uint32_t kMtReady = 1u << 30;      // hooks_mask 里借一位表示 MT 导出可用

ControlBlock* gBlock = nullptr;
HANDLE gMapping = nullptr;
std::uint8_t* gBase = nullptr;
volatile LONG gMtReady = 0;
std::uint32_t gTimeStamp = 0;
std::uint32_t gImageSize = 0;
char gProfile[MAX_PATH] = {};
volatile LONG gProfileCalls = 0;
volatile LONG gSeedCalls = 0;
FILE* gLog = nullptr;

using GetProfileFn = BOOL(WINAPI*)(HANDLE, LPSTR, LPDWORD);
GetProfileFn gOrigGetProfile = nullptr;

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

void setStatus(std::uint32_t status, const char* text) {
  logf("status=%u %s", status, text);
  if (!gBlock) return;
  std::snprintf(gBlock->status_text, sizeof(gBlock->status_text), "%s", text);
  MemoryBarrier();
  gBlock->status = status;
}

BOOL WINAPI hookGetProfile(HANDLE, LPSTR buffer, LPDWORD size) {
  const LONG k = InterlockedIncrement(&gProfileCalls);
  const DWORD needed = static_cast<DWORD>(std::strlen(gProfile) + 1);
  if (!size) { SetLastError(ERROR_INVALID_PARAMETER); return FALSE; }
  if (!buffer || *size < needed) {
    *size = needed;
    SetLastError(ERROR_INSUFFICIENT_BUFFER);
    return FALSE;
  }
  std::memcpy(buffer, gProfile, needed);
  *size = needed;
  if (k <= 8) logf("GetUserProfileDirectoryA -> %s (call %ld)", gProfile, k);
  return TRUE;
}

std::uint32_t mtHash(std::uint32_t* cursorOut) {
  const auto* st = reinterpret_cast<const std::uint32_t*>(gBase + kRvaState);
  const std::uint32_t cursor = *reinterpret_cast<const std::uint32_t*>(gBase + kRvaCursor);
  std::uint32_t h = 2166136261u;
  auto mix = [&h](std::uint32_t w) {
    for (int i = 0; i < 4; ++i) { h ^= (w >> (8 * i)) & 0xFF; h *= 16777619u; }
  };
  for (int i = 0; i < 624; ++i) mix(st[i]);
  mix(cursor);
  if (cursorOut) *cursorOut = cursor;
  return h;
}

struct LuaApi {
  long long(__cdecl* checkinteger)(void*, int) = nullptr;
  void(__cdecl* pushinteger)(void*, long long) = nullptr;
  double(__cdecl* checknumber)(void*, int) = nullptr;
  void(__cdecl* pushnumber)(void*, double) = nullptr;
  const char*(__cdecl* checklstring)(void*, int, std::size_t*) = nullptr;
  const char*(__cdecl* pushlstring)(void*, const char*, std::size_t) = nullptr;
  bool ok = false;
};
LuaApi gLua;

bool loadLua() {
  if (gLua.ok) return true;
  HMODULE api = GetModuleHandleA("Lua5.3.3r.dll");
  if (!api) return false;
  gLua.checkinteger = reinterpret_cast<long long(__cdecl*)(void*, int)>(GetProcAddress(api, "luaL_checkinteger"));
  gLua.pushinteger = reinterpret_cast<void(__cdecl*)(void*, long long)>(GetProcAddress(api, "lua_pushinteger"));
  gLua.checknumber = reinterpret_cast<double(__cdecl*)(void*, int)>(GetProcAddress(api, "luaL_checknumber"));
  gLua.pushnumber = reinterpret_cast<void(__cdecl*)(void*, double)>(GetProcAddress(api, "lua_pushnumber"));
  gLua.checklstring = reinterpret_cast<const char*(__cdecl*)(void*, int, std::size_t*)>(GetProcAddress(api, "luaL_checklstring"));
  gLua.pushlstring = reinterpret_cast<const char*(__cdecl*)(void*, const char*, std::size_t)>(GetProcAddress(api, "lua_pushlstring"));
  gLua.ok = gLua.checkinteger && gLua.pushinteger && gLua.checknumber && gLua.pushnumber && gLua.checklstring && gLua.pushlstring;
  return gLua.ok;
}

bool openControlBlock(DWORD pid) {
  char name[64];
  formatMappingName(name, sizeof(name), pid);
  gMapping = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, kControlBlockBytes, name);
  if (!gMapping) return false;
  gBlock = static_cast<ControlBlock*>(MapViewOfFile(gMapping, FILE_MAP_ALL_ACCESS, 0, 0, kControlBlockBytes));
  if (!gBlock) return false;
  std::memset(gBlock, 0, kControlBlockBytes);
  gBlock->version = kLayoutVersion;
  gBlock->pid = pid;
  gBlock->status = kStatusLoading;
  MemoryBarrier();
  gBlock->magic = kMagic;
  return true;
}

// 返回 true = 是 J746 且 MT 两处指令的立即数与预期地址一致（MT 导出可用）。不改任何状态位。
bool checkMtIdentity(char* why, std::size_t whyLen) {
  gBase = reinterpret_cast<std::uint8_t*>(GetModuleHandleA(nullptr));
  const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(gBase);
  const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS32*>(gBase + dos->e_lfanew);
  gTimeStamp = nt->FileHeader.TimeDateStamp;
  gImageSize = nt->OptionalHeader.SizeOfImage;
  char forced[8] = {};
  if (GetEnvironmentVariableA("RR_MT_TEST_IDENTITY_FAIL", forced, sizeof(forced)) && forced[0] == '1') {
    std::snprintf(why, whyLen, "identity forced to fail (RR_MT_TEST_IDENTITY_FAIL=1)");
    return false;
  }
  if (gTimeStamp != 0x624393F6 || gImageSize != 8540160) {
    std::snprintf(why, whyLen, "identity ts=0x%08X size=%u (want 0x624393F6 / 8540160)", gTimeStamp, gImageSize);
    return false;
  }
  // 89 0D <imm32> : mov [imm32], ecx  (state[0] = seed)
  const std::uint8_t* seed = gBase + kRvaSeed;
  std::uint32_t immState = 0, immCursor = 0;
  std::memcpy(&immState, seed + 2, 4);
  // A1 <imm32> : mov eax, [imm32]  (cursor load in GlobalMT_Next)
  const std::uint8_t* next = gBase + kRvaNext;
  std::memcpy(&immCursor, next + 1, 4);
  const auto wantState = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(gBase) + kRvaState);
  const auto wantCursor = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(gBase) + kRvaCursor);
  logf("identity: base=%p seed[0..1]=%02X %02X imm=0x%08X want=0x%08X; next[0]=%02X imm=0x%08X want=0x%08X", gBase,
       seed[0], seed[1], immState, wantState, next[0], immCursor, wantCursor);
  if (seed[0] != 0x89 || seed[1] != 0x0D || immState != wantState || next[0] != 0xA1 || immCursor != wantCursor) {
    std::snprintf(why, whyLen, "identity MT prologue mismatch");
    return false;
  }
  return true;
}

void openLog(DWORD pid) {
  char dir[MAX_PATH] = {};
  const DWORD n = GetEnvironmentVariableA("RR_MT_LOG_DIR", dir, MAX_PATH);
  if (!n || n >= MAX_PATH - 40) return;
  CreateDirectoryA(dir, nullptr);
  char path[MAX_PATH];
  std::snprintf(path, sizeof(path), "%s\\rr_mt.%lu.log", dir, static_cast<unsigned long>(pid));
  gLog = std::fopen(path, "a");
  if (gBlock) std::snprintf(gBlock->log_path, sizeof(gBlock->log_path), "%s", path);
}

// ---- phase10-mods-console：控制台命令（RR_MT_CONSOLE=1 才安装）
// 静态分析（experiments/run-recorder/phase10-mods-console/NOTES.md §2）：控制台对象的输入行是 this+0x1C 的 MSVC std::string
// （SSO 缓冲/指针在 +0x1C，长度 +0x2C，容量 +0x30）。Console::SubmitInput（RVA 0x258AB0）在输入非空白时把它推进历史、清空
// 输入行并以 thiscall 调命令执行函数（RVA 0x25E160，调用点 RVA 0x258BB5）。两个调用点（RVA 0x25CFF8、0x25D280）都先
// push ecx 作哑参数，函数以 ret 4 返回，所以钩子按 __fastcall(this, edx, int) 声明（被调方清 4 字节，与 ret 4 一致）。
constexpr std::uint32_t kRvaConsoleSubmit = 0x258AB0;
constexpr std::uint32_t kRvaConsoleRunCmd = 0x25E160;
constexpr std::uint32_t kConsoleInput = 0x1C;
using ConsoleSubmitFn = void(__fastcall*)(void*, void*, int);
ConsoleSubmitFn gOrigConsoleSubmit = nullptr;
CRITICAL_SECTION gConsoleLock;
std::deque<std::string> gConsoleQueue;
volatile LONG gConsoleOn = 0;
volatile LONG gConsoleSubmits = 0;
volatile LONG gConsoleDropped = 0;

bool consoleCodeIs(std::uint32_t rva, const std::uint8_t* want, std::size_t n) {
  return std::memcmp(gBase + rva, want, n) == 0;
}

std::string consoleReadInput(const std::uint8_t* self) {
  const std::uint8_t* s = self + kConsoleInput;
  const std::uint32_t size = *reinterpret_cast<const std::uint32_t*>(s + 0x10);
  const std::uint32_t cap = *reinterpret_cast<const std::uint32_t*>(s + 0x14);
  const char* data = cap > 15 ? *reinterpret_cast<const char* const*>(s) : reinterpret_cast<const char*>(s);
  if (!data || size > 4096) return std::string();
  return std::string(data, size);
}

void __fastcall hookConsoleSubmit(void* self, void* edx, int arg) {
  std::string line;
  bool blank = true;
  if (self) line = consoleReadInput(static_cast<const std::uint8_t*>(self));
  for (char c : line) if (c != ' ') { blank = false; break; }
  if (!blank) {
    const LONG k = InterlockedIncrement(&gConsoleSubmits);
    EnterCriticalSection(&gConsoleLock);
    if (gConsoleQueue.size() < 256) gConsoleQueue.push_back(line);
    else InterlockedIncrement(&gConsoleDropped);
    LeaveCriticalSection(&gConsoleLock);
    logf("console: submit #%ld \"%s\"", k, line.c_str());
  }
  gOrigConsoleSubmit(self, edx, arg);
}

bool consoleInstall(char* why, std::size_t whyLen) {
  if (gTimeStamp != 0x624393F6 || gImageSize != 8540160) {
    std::snprintf(why, whyLen, "EXE is not J746");
    return false;
  }
  // 函数头（栈对齐帧）、读输入行的 3 条指令、调用命令执行函数的 call
  static const std::uint8_t head[16] = {0x53, 0x8B, 0xDC, 0x83, 0xEC, 0x08, 0x83, 0xE4, 0xF8, 0x83, 0xC4, 0x04, 0x55, 0x8B, 0x6B, 0x04};
  static const std::uint8_t input[8] = {0x8B, 0xF1, 0x8B, 0x56, 0x30, 0x8D, 0x7E, 0x1C};  // mov esi,ecx; mov edx,[esi+30]; lea edi,[esi+1C]
  std::uint8_t call[5] = {0xE8};
  const std::int32_t rel = static_cast<std::int32_t>(kRvaConsoleRunCmd) - static_cast<std::int32_t>(0x258BB5 + 5);
  std::memcpy(call + 1, &rel, 4);
  if (!consoleCodeIs(kRvaConsoleSubmit, head, sizeof(head)) || !consoleCodeIs(0x258AEE, input, sizeof(input)) ||
      !consoleCodeIs(0x258BB5, call, sizeof(call))) {
    std::snprintf(why, whyLen, "console code signature mismatch");
    return false;
  }
  InitializeCriticalSection(&gConsoleLock);
  void* target = gBase + kRvaConsoleSubmit;
  MH_STATUS st = MH_CreateHook(target, reinterpret_cast<void*>(&hookConsoleSubmit), reinterpret_cast<void**>(&gOrigConsoleSubmit));
  if (st != MH_OK) { std::snprintf(why, whyLen, "MH_CreateHook %d", static_cast<int>(st)); return false; }
  st = MH_EnableHook(target);
  if (st != MH_OK) { std::snprintf(why, whyLen, "MH_EnableHook %d", static_cast<int>(st)); return false; }
  InterlockedExchange(&gConsoleOn, 1);
  std::snprintf(why, whyLen, "Console::SubmitInput RVA 0x%X", kRvaConsoleSubmit);
  return true;
}

DWORD WINAPI initThread(LPVOID) {
  const DWORD pid = GetCurrentProcessId();
  const bool shm = openControlBlock(pid);
  openLog(pid);
  char exe[MAX_PATH] = {};
  GetModuleFileNameA(nullptr, exe, sizeof(exe));
  logf("rr_mt (j746_rec) attach pid=%lu exe=%s shm=%d", static_cast<unsigned long>(pid), exe, shm ? 1 : 0);
  if (!shm) return 0;
  char why[160] = {};
  const bool mtOk = checkMtIdentity(why, sizeof(why));
  if (!mtOk) logf("MT exports disabled: %s", why);
  const DWORD n = GetEnvironmentVariableA("RR_MT_PROFILE", gProfile, MAX_PATH);
  if (!n || n >= MAX_PATH || GetFileAttributesA(gProfile) == INVALID_FILE_ATTRIBUTES) {
    setStatus(kStatusHookFailed, "ERROR:RR_MT_PROFILE missing or not a directory");
    return 0;
  }
  // phase8-mouse：RR_MT_AFFINITY=<掩码>（十进制或 0x 十六进制）时把整个进程限制在这些 CPU 上。注入器挂起启动、等控制块
  // Active 才恢复主线程，所以游戏自己的线程都在设置之后创建。实测（FINDINGS.md §H4）多核并行时引擎的 1000.x 环境特效偶发
  // 不确定（线程竞争），限制到 1 个核后同条件 16 次回放 0 次；回放默认用，录制会话不设。
  {
    char aff[32] = {};
    if (GetEnvironmentVariableA("RR_MT_AFFINITY", aff, sizeof(aff))) {
      const unsigned long long m = std::strtoull(aff, nullptr, 0);
      if (m) {
        const BOOL ok = SetProcessAffinityMask(GetCurrentProcess(), static_cast<DWORD_PTR>(m));
        DWORD_PTR pm = 0, sm = 0;
        GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm);
        logf("affinity: RR_MT_AFFINITY=%s -> SetProcessAffinityMask %s (process mask now 0x%lX, system 0x%lX)", aff,
             ok ? "ok" : "FAILED", static_cast<unsigned long>(pm), static_cast<unsigned long>(sm));
      }
    }
  }
  MH_STATUS st = MH_Initialize();
  if (st != MH_OK) { setStatus(kStatusHookFailed, "ERROR:MH_Initialize"); return 0; }
  HMODULE userenv = LoadLibraryA("userenv.dll");
  void* target = userenv ? reinterpret_cast<void*>(GetProcAddress(userenv, "GetUserProfileDirectoryA")) : nullptr;
  if (!target) { setStatus(kStatusHookFailed, "ERROR:GetUserProfileDirectoryA not found"); return 0; }
  st = MH_CreateHook(target, reinterpret_cast<void*>(&hookGetProfile), reinterpret_cast<void**>(&gOrigGetProfile));
  if (st != MH_OK) { setStatus(kStatusHookFailed, "ERROR:profile hook create"); return 0; }
  st = MH_EnableHook(target);
  if (st != MH_OK) { setStatus(kStatusHookFailed, "ERROR:profile hook enable"); return 0; }
  std::uint32_t mask = kHookProfile;
  // phase4-turbo：可选加速（默认关）
  // phase6-watch：RR_MT_WATCH=1（观战速率旋钮）/ RR_MT_CAPTURE=1（实录导出）同样需要虚拟时钟钩子，也走 turbo 安装
  auto envOn = [](const char* k) { char v[8] = {}; return GetEnvironmentVariableA(k, v, sizeof(v)) && v[0] == '1'; };
  rr_turbo::Options topt;
  topt.watch = envOn("RR_MT_WATCH");
  topt.capture = envOn("RR_MT_CAPTURE");
  const bool turbo = envOn("RR_MT_TURBO") || topt.watch || topt.capture;
  bool gate = false;
  if (turbo) {
    if (gTimeStamp != 0x624393F6 || gImageSize != 8540160) {
      setStatus(kStatusHookFailed, "ERROR:RR_MT_TURBO=1 but EXE is not J746");
      return 0;
    }
    gate = envOn("RR_MT_TURBO_GPU_GATE");
    topt.gpuGate = gate;
    char twhy[128] = {};
    if (!rr_turbo::install(gBase, gBlock, topt, &logf, twhy, sizeof(twhy))) {
      setStatus(kStatusHookFailed, twhy);
      return 0;
    }
    mask |= gBlock->hooks_mask;
  }
  // phase10：控制台命令钩子（录制会话 --allow-console 才设 RR_MT_CONSOLE=1）；装不上只记日志
  bool consoleOn = false;
  if (envOn("RR_MT_CONSOLE")) {
    char cwhy[128] = {};
    consoleOn = consoleInstall(cwhy, sizeof(cwhy));
    logf("console: %s (%s)", consoleOn ? "hook installed" : "OFF", cwhy);
  }
  if (mtOk) {
    std::uint32_t cursor = 0;
    const std::uint32_t h = mtHash(&cursor);
    logf("profile=%s; MT at load: cursor=%u hash=0x%08X", gProfile, cursor, h);
    InterlockedExchange(&gMtReady, 1);
    mask |= kMtReady;
  } else {
    logf("profile=%s; MT not available", gProfile);
  }
  gBlock->hooks_mask = mask;
  char text[128];
  std::snprintf(text, sizeof(text), "OK j746_rec profile-redirect; MT exports %s%s%s%s%s", mtOk ? "ready" : "OFF (identity)",
                turbo ? (gate ? "; TURBO virtual-clock + GPU gate" : "; TURBO virtual-clock") : "",
                topt.watch ? "; WATCH" : "", topt.capture ? "; CAPTURE" : "", consoleOn ? "; CONSOLE" : "");
  setStatus(kStatusActive, text);
  return 0;
}

}  // namespace

// Lua: cursor, hash = RRMT_Seed(seed)        （身份门没过：不返回值）
extern "C" __declspec(dllexport) int RRMT_Seed(void* L) {
  if (!loadLua() || !gMtReady) return 0;
  const auto seed = static_cast<std::uint32_t>(gLua.checkinteger(L, 1));
  reinterpret_cast<void(__fastcall*)(std::uint32_t)>(gBase + kRvaSeed)(seed);
  const LONG k = InterlockedIncrement(&gSeedCalls);
  std::uint32_t cursor = 0;
  const std::uint32_t h = mtHash(&cursor);
  logf("RRMT_Seed(%u) #%ld -> cursor=%u hash=0x%08X", seed, k, cursor, h);
  gLua.pushinteger(L, cursor);
  gLua.pushinteger(L, h);
  return 2;
}

// 稳定读：把 624 个状态字 + cursor 拷两份，两份相同才算数（最多 8 次）。MT_PROBE §5.3 与第三阶段 eB3L 见过
// "单帧 MT 哈希读数异常"（cursor 相同、hash 不同、下一帧恢复、游戏状态无差异），怀疑读的同时有别的线程短暂改写；
// 两份不同时把不同的字（前 8 个）记进 DLL 日志，供诊断。返回读的次数（1 = 第一次就稳定）。
volatile LONG gUnstableReads = 0;
std::uint32_t gSnapA[625];
std::uint32_t gSnapB[625];

int stableSnapshot(std::uint32_t* out) {
  const auto* st = reinterpret_cast<const volatile std::uint32_t*>(gBase + kRvaState);
  const auto* cur = reinterpret_cast<const volatile std::uint32_t*>(gBase + kRvaCursor);
  std::uint32_t* a = gSnapA;
  std::uint32_t* b = gSnapB;
  for (int i = 0; i < 624; ++i) a[i] = st[i];
  a[624] = *cur;
  int reads = 1;
  for (; reads < 8; ++reads) {
    for (int i = 0; i < 624; ++i) b[i] = st[i];
    b[624] = *cur;
    if (std::memcmp(a, b, sizeof(gSnapA)) == 0) break;
    const LONG k = InterlockedIncrement(&gUnstableReads);
    if (k <= 50) {
      char buf[512];
      int n = std::snprintf(buf, sizeof(buf), "unstable MT read #%ld (attempt %d, thread %lu): ", k, reads,
                            static_cast<unsigned long>(GetCurrentThreadId()));
      int shown = 0;
      for (int i = 0; i < 625 && shown < 8 && n < static_cast<int>(sizeof(buf)) - 40; ++i) {
        if (a[i] != b[i]) {
          n += std::snprintf(buf + n, sizeof(buf) - n, "[%d] %08X->%08X ", i, a[i], b[i]);
          ++shown;
        }
      }
      logf("%s", buf);
    }
    std::uint32_t* t = a;
    a = b;
    b = t;
  }
  std::memcpy(out, a, sizeof(gSnapA));
  return reads;
}

std::uint32_t hashWords(const std::uint32_t* w) {
  std::uint32_t h = 2166136261u;
  for (int k = 0; k < 625; ++k)
    for (int i = 0; i < 4; ++i) { h ^= (w[k] >> (8 * i)) & 0xFF; h *= 16777619u; }
  return h;
}

// Lua: cursor, hash, reads = RRMT_State()   （只读，不推进 MT；hash 与 mtHash 相同口径；身份门没过：不返回值）
extern "C" __declspec(dllexport) int RRMT_State(void* L) {
  if (!loadLua() || !gMtReady) return 0;
  static std::uint32_t words[625];
  const int reads = stableSnapshot(words);
  gLua.pushinteger(L, words[624]);
  gLua.pushinteger(L, hashWords(words));
  gLua.pushinteger(L, reads);
  return 3;
}

// ---- phase11-human3：自定义种子局标记。`seed` 命令开出的局 Seeds::_isCustomRun = 1（自然局为 0）；引擎在这种局里不刷
// 捐款机（6.8）等，回放自然局的录制时从商店起分歧。Lua 没有能清掉它的方法（ClearStartSeed 只把种子清 0），所以这里直接写：
// Seeds 布局（REPENTOGON libzhl Seeds.zhl；AB+ Seeds::IsCustomRun 同为 `return *(bool*)this`）：+0 bool _isCustomRun，
// +4 uint32 _gameStartSeed。Seeds* 从 Lua 的 Seeds 用户数据里取（LuaBridge 式块：[vtable][Seeds*] 或 [Seeds*]），
// 身份门：候选指针所指内存可读、+4 处等于 Lua 传入的 GetStartSeed()、+0 处是 0/1，才写。
// Lua: ok, old, how = RRMT_SetCustomRun(seeds_userdata, start_seed, value)   （ok = 1 写了；0 = 没通过门，how 说明）
namespace {
bool readable(const void* p, std::size_t n) {
  MEMORY_BASIC_INFORMATION mbi;
  if (!p || VirtualQuery(p, &mbi, sizeof(mbi)) != sizeof(mbi)) return false;
  if (mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD))) return false;
  const DWORD rw = PAGE_READWRITE | PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_WRITECOPY;
  if (!(mbi.Protect & rw)) return false;
  const auto end = reinterpret_cast<std::uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
  return reinterpret_cast<std::uintptr_t>(p) + n <= end;
}
}  // namespace

extern "C" __declspec(dllexport) int RRMT_SetCustomRun(void* L) {
  if (!loadLua()) return 0;
  static void*(__cdecl* touserdata)(void*, int) = nullptr;
  if (!touserdata) {
    HMODULE api = GetModuleHandleA("Lua5.3.3r.dll");
    if (api) touserdata = reinterpret_cast<void*(__cdecl*)(void*, int)>(GetProcAddress(api, "lua_touserdata"));
  }
  auto fail = [&](const char* how) {
    logf("RRMT_SetCustomRun: %s", how);
    gLua.pushinteger(L, 0);
    gLua.pushinteger(L, -1);
    gLua.pushlstring(L, how, std::strlen(how));
    return 3;
  };
  if (!touserdata) return fail("no lua_touserdata");
  const auto seed = static_cast<std::uint32_t>(gLua.checkinteger(L, 2));
  const auto value = static_cast<std::uint8_t>(gLua.checkinteger(L, 3) != 0 ? 1 : 0);
  auto* ud = static_cast<std::uint8_t*>(touserdata(L, 1));
  if (!ud || !readable(ud, 8)) return fail("userdata not readable");
  for (int off = 0; off <= 4; off += 4) {
    auto* p = *reinterpret_cast<std::uint8_t**>(ud + off);
    if (!readable(p, 8)) continue;
    const std::uint32_t s = *reinterpret_cast<std::uint32_t*>(p + 4);
    const std::uint8_t old = *p;
    if (s != seed || old > 1) continue;
    *p = value;
    logf("RRMT_SetCustomRun: Seeds*=%p (userdata+%d) start_seed=%u custom %u -> %u", static_cast<void*>(p), off, s, old, value);
    gLua.pushinteger(L, 1);
    gLua.pushinteger(L, old);
    char how[32];
    std::snprintf(how, sizeof(how), "userdata+%d", off);
    gLua.pushlstring(L, how, std::strlen(how));
    return 3;
  }
  return fail("identity gate failed (start seed not found at Seeds+4)");
}

// Lua: n = RRMT_UnstableReads()   （进程启动以来两份拷贝不一致的次数）
extern "C" __declspec(dllexport) int RRMT_UnstableReads(void* L) {
  if (!loadLua()) return 0;
  gLua.pushinteger(L, gUnstableReads);
  return 1;
}

// Lua: ready, profile_calls, seed_calls, pe_timestamp, size_of_image = RRMT_Info()
extern "C" __declspec(dllexport) int RRMT_Info(void* L) {
  if (!loadLua()) return 0;
  gLua.pushinteger(L, gMtReady ? 1 : 0);
  gLua.pushinteger(L, gProfileCalls);
  gLua.pushinteger(L, gSeedCalls);
  gLua.pushinteger(L, gTimeStamp);
  gLua.pushinteger(L, gImageSize);
  return 5;
}

// ---- phase8-mouse：回放时把录制的鼠标还原给引擎（只有环境变量 RR_MT_MOUSE=1 且 J746 身份门通过才可用）
// 静态分析（experiments/run-recorder/phase8-mouse/FINDINGS.md §H1c）：J746 不用 GetCursorPos 取光标位置（两处 GetCursorPos
// 只在显示/隐藏光标的辅助函数里），位置只来自窗口消息 WM_MOUSEMOVE → 静态链接的 GLFW 把它写进主窗口结构的
// virtualCursorPosX/Y（double，+0x48/+0x50；全局指针 [base+0x7A15D8]）。读它的只有 3 处：Lua Input.GetMousePosition
// （RVA 0x4014C0）、Manager::Update 末尾的"光标闲置计数"（RVA 0x499D6E：本次位置与上次 float 不同 → Manager+0x1EA978 = 300，
// 否则 -1；>0 时 Manager::Render 画游戏内光标）、菜单的鼠标点击处理（RVA 0x583E83）。按键在 +0x5C 起的 char[8]。
// 回放进程窗口不在前台、也没人动鼠标，所以这里直接写这几个字段（不经过消息），并在第一次调用时子类化窗口过程，丢掉
// 真实的鼠标消息（WM_MOUSEMOVE/按键/滚轮/WM_MOUSELEAVE），免得有人把鼠标划过回放窗口改变结果。
constexpr std::uint32_t kRvaGlfwWindowPtr = 0x7A15D8;
constexpr std::uint32_t kGlfwCursorX = 0x48;
constexpr std::uint32_t kGlfwCursorY = 0x50;
constexpr std::uint32_t kGlfwButtons = 0x5C;
constexpr std::uint32_t kGlfwHwnd = 0x218;
constexpr std::uint32_t kRvaManagerPtr = 0x7C768C;
constexpr std::uint32_t kMgrCursorLastX = 0x1EA970;  // float：上一次 Manager::Update 看到的光标位置
constexpr std::uint32_t kMgrCursorLastY = 0x1EA974;
constexpr std::uint32_t kMgrCursorIdle = 0x1EA978;   // int：光标闲置倒计时（移动后 = 300，每次 Manager::Update -1）

volatile LONG gMouseReady = 0;   // 0 = 未检查，1 = 可用，-1 = 不可用
WNDPROC gMouseOrigProc = nullptr;
HWND gMouseHwnd = nullptr;
volatile LONG gMouseDropped = 0;
volatile LONG gMouseSets = 0;

bool codeIs(std::uint32_t rva, const std::uint8_t* want, std::size_t n) { return std::memcmp(gBase + rva, want, n) == 0; }

bool mouseCheck() {
  if (gMouseReady) return gMouseReady > 0;
  char v[8] = {};
  bool ok = GetEnvironmentVariableA("RR_MT_MOUSE", v, sizeof(v)) && v[0] == '1';
  const char* why = ok ? "" : "RR_MT_MOUSE!=1";
  if (ok && (gTimeStamp != 0x624393F6 || gImageSize != 8540160)) { ok = false; why = "EXE is not J746"; }
  if (ok) {
    const auto win = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(gBase) + kRvaGlfwWindowPtr);
    std::uint8_t a[5] = {0xA1};
    std::memcpy(a + 1, &win, 4);
    static const std::uint8_t ldX[5] = {0xF2, 0x0F, 0x10, 0x48, 0x48};          // movsd xmm1, [eax+0x48]
    static const std::uint8_t idle[6] = {0xC7, 0x87, 0x78, 0xA9, 0x1E, 0x00};    // mov dword [edi+0x1EA978], imm32
    if (!codeIs(0x4014C6, a, 5) || !codeIs(0x4014F4, ldX, 5) || !codeIs(0x499D6E, a, 5) || !codeIs(0x499DD1, idle, 6)) {
      ok = false;
      why = "mouse code signature mismatch";
    }
  }
  logf("mouse: %s %s", ok ? "ready" : "OFF", why);
  InterlockedExchange(&gMouseReady, ok ? 1 : -1);
  return ok;
}

std::uint8_t* glfwWindow() { return *reinterpret_cast<std::uint8_t**>(gBase + kRvaGlfwWindowPtr); }
std::uint8_t* manager() { return *reinterpret_cast<std::uint8_t**>(gBase + kRvaManagerPtr); }

LRESULT CALLBACK mouseWndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
  if ((m >= WM_MOUSEFIRST && m <= WM_MOUSELAST) || m == WM_MOUSELEAVE) {
    const LONG k = InterlockedIncrement(&gMouseDropped);
    if (k <= 3 || (k % 1000) == 0) logf("mouse: dropped real mouse message 0x%04X (#%ld)", m, k);
    return 0;
  }
  return CallWindowProcW(gMouseOrigProc, h, m, w, l);
}

void mouseSubclass(std::uint8_t* win) {
  if (gMouseHwnd) return;
  HWND hw = *reinterpret_cast<HWND*>(win + kGlfwHwnd);
  if (!hw || !IsWindow(hw)) return;
  gMouseHwnd = hw;
  gMouseOrigProc = reinterpret_cast<WNDPROC>(SetWindowLongW(hw, GWL_WNDPROC, reinterpret_cast<LONG>(&mouseWndProc)));
  logf("mouse: window %p subclassed (orig wndproc %p); real mouse messages are dropped from now on",
       reinterpret_cast<void*>(hw), reinterpret_cast<void*>(gMouseOrigProc));
}

// Lua: ok = RRMT_SetMouse(x, y, buttons, mode)
//   x, y    = 录制 ms 流的"屏幕坐标"（= Input.GetMousePosition(false) = GLFW virtualCursorPos），写进 +0x48/+0x50（double）
//   buttons = 左 1 右 2 中 4（GLFW mouseButtons[0..2] = 1/0）
//   mode    = 0：只改光标（下一次 Manager::Update 看到位置变了，光标闲置计数照常重置为 300，与真人移动鼠标相同）
//             1：同时把 Manager 的"上次光标位置"改成同一值（这次改动不重置闲置计数）
//             2：同 1，并把闲置计数清 0（光标隐藏）——回放开局设初始位置用
// 不可用（RR_MT_MOUSE!=1、身份门不过、窗口还没建）时不返回值。
extern "C" __declspec(dllexport) int RRMT_SetMouse(void* L) {
  if (!loadLua() || !mouseCheck()) return 0;
  std::uint8_t* win = glfwWindow();
  if (!win) return 0;
  const double x = gLua.checknumber(L, 1);
  const double y = gLua.checknumber(L, 2);
  const auto buttons = static_cast<std::uint32_t>(gLua.checkinteger(L, 3));
  const int mode = static_cast<int>(gLua.checkinteger(L, 4));
  mouseSubclass(win);
  *reinterpret_cast<double*>(win + kGlfwCursorX) = x;
  *reinterpret_cast<double*>(win + kGlfwCursorY) = y;
  for (int b = 0; b < 3; ++b) {
    auto* s = reinterpret_cast<char*>(win + kGlfwButtons + b);
    const char want = (buttons >> b) & 1;
    if (want == 0 || *s != 3) *s = want;  // 3 = GLFW 的 sticky 按下（已按下时保持）
  }
  if (mode >= 1) {
    std::uint8_t* mgr = manager();
    if (mgr) {
      *reinterpret_cast<float*>(mgr + kMgrCursorLastX) = static_cast<float>(x);
      *reinterpret_cast<float*>(mgr + kMgrCursorLastY) = static_cast<float>(y);
      if (mode == 2) *reinterpret_cast<std::int32_t*>(mgr + kMgrCursorIdle) = 0;
    }
  }
  const LONG k = InterlockedIncrement(&gMouseSets);
  if (k <= 3) logf("mouse: set #%ld x=%.2f y=%.2f buttons=%u mode=%d", k, x, y, buttons, mode);
  gLua.pushinteger(L, 1);
  return 1;
}

// Lua: x, y, buttons, idle, last_x, last_y, dropped, sets = RRMT_GetMouse()   （诊断/单元测试；不可用时不返回值）
extern "C" __declspec(dllexport) int RRMT_GetMouse(void* L) {
  if (!loadLua() || !mouseCheck()) return 0;
  std::uint8_t* win = glfwWindow();
  if (!win) return 0;
  gLua.pushnumber(L, *reinterpret_cast<double*>(win + kGlfwCursorX));
  gLua.pushnumber(L, *reinterpret_cast<double*>(win + kGlfwCursorY));
  std::uint32_t b = 0;
  for (int i = 0; i < 3; ++i) b |= (win[kGlfwButtons + i] ? 1u : 0u) << i;
  gLua.pushinteger(L, b);
  std::uint8_t* mgr = manager();
  gLua.pushinteger(L, mgr ? *reinterpret_cast<std::int32_t*>(mgr + kMgrCursorIdle) : -999);
  gLua.pushnumber(L, mgr ? *reinterpret_cast<float*>(mgr + kMgrCursorLastX) : 0.0);
  gLua.pushnumber(L, mgr ? *reinterpret_cast<float*>(mgr + kMgrCursorLastY) : 0.0);
  gLua.pushinteger(L, gMouseDropped);
  gLua.pushinteger(L, gMouseSets);
  return 8;
}

// ---- phase10：控制台命令
// Lua: line, k, pending = RRMT_ConsoleTake()   取走最早一条玩家在控制台提交的输入行（k = 第几条，pending = 取走后还剩几条）；
// 没有排队的（或钩子没装）不返回值
extern "C" __declspec(dllexport) int RRMT_ConsoleTake(void* L) {
  if (!loadLua() || !gConsoleOn) return 0;
  EnterCriticalSection(&gConsoleLock);
  if (gConsoleQueue.empty()) { LeaveCriticalSection(&gConsoleLock); return 0; }
  const std::string line = gConsoleQueue.front();
  gConsoleQueue.pop_front();
  const long long left = static_cast<long long>(gConsoleQueue.size());
  LeaveCriticalSection(&gConsoleLock);
  gLua.pushlstring(L, line.data(), line.size());
  gLua.pushinteger(L, gConsoleSubmits - left);
  gLua.pushinteger(L, left);
  return 3;
}

// Lua: on, submits, dropped = RRMT_ConsoleInfo()
extern "C" __declspec(dllexport) int RRMT_ConsoleInfo(void* L) {
  if (!loadLua()) return 0;
  gLua.pushinteger(L, gConsoleOn ? 1 : 0);
  gLua.pushinteger(L, gConsoleSubmits);
  gLua.pushinteger(L, gConsoleDropped);
  return 3;
}

// ---- phase8-mouse 诊断：主线程浮点环境。Lua: mxcsr, x87cw = RRMT_Fpu(set_mxcsr, set_x87cw)（参数 -1 = 不改）
// 只给实验用（假设检验"录制进程的 MXCSR / x87 控制字与回放进程不同"），录制/回放流程不调用。
extern "C" __declspec(dllexport) int RRMT_Fpu(void* L) {
  if (!loadLua()) return 0;
  const long long setMx = gLua.checkinteger(L, 1);
  const long long setCw = gLua.checkinteger(L, 2);
  unsigned int mx = 0;
  unsigned short cw = 0;
  __asm__ __volatile__("stmxcsr %0" : "=m"(mx));
  __asm__ __volatile__("fnstcw %0" : "=m"(cw));
  if (setMx >= 0) {
    unsigned int v = static_cast<unsigned int>(setMx);
    __asm__ __volatile__("ldmxcsr %0" : : "m"(v));
  }
  if (setCw >= 0) {
    unsigned short v = static_cast<unsigned short>(setCw);
    __asm__ __volatile__("fldcw %0" : : "m"(v));
  }
  logf("fpu: mxcsr=0x%08X x87cw=0x%04X set_mxcsr=%lld set_cw=%lld", mx, cw, setMx, setCw);
  gLua.pushinteger(L, mx);
  gLua.pushinteger(L, cw);
  return 2;
}

// ---- phase6-watch：观战 / 实录导出（没装 watch/capture 钩子时 RRW_Info 第一个返回值为 0，其余函数无副作用）
// Lua: RRW_Set(rate, present_min_ms)   rate 1 = 原速，0 = 挂起（整次跳过 Manager::Update），<0 = 不限速
extern "C" __declspec(dllexport) int RRW_Set(void* L) {
  if (!loadLua()) return 0;
  rr_turbo::watchSet(gLua.checknumber(L, 1), gLua.checknumber(L, 2));
  return 0;
}

// Lua: available, skipped_this, iterations, skipped, presents, presents_skipped, wall_ms, rate, cap_frames, cap_dups,
//      cap_w, cap_h, cap_active, draw_this = RRW_Info()
extern "C" __declspec(dllexport) int RRW_Info(void* L) {
  if (!loadLua()) return 0;
  const rr_turbo::WatchInfo w = rr_turbo::watchInfo();
  gLua.pushinteger(L, w.available);
  gLua.pushinteger(L, w.skippedThis);
  gLua.pushinteger(L, static_cast<long long>(w.iterations));
  gLua.pushinteger(L, static_cast<long long>(w.skipped));
  gLua.pushinteger(L, static_cast<long long>(w.presents));
  gLua.pushinteger(L, static_cast<long long>(w.presentsSkipped));
  gLua.pushnumber(L, w.wallMs);
  gLua.pushnumber(L, w.rate);
  gLua.pushinteger(L, static_cast<long long>(w.capFrames));
  gLua.pushinteger(L, static_cast<long long>(w.capDups));
  gLua.pushinteger(L, w.capWidth);
  gLua.pushinteger(L, w.capHeight);
  gLua.pushinteger(L, w.capActive);
  gLua.pushinteger(L, w.drawThis);
  return 14;
}

// Lua: RRW_Mark(n)   每个 MC_POST_UPDATE（n = 本局刚完成的逻辑帧号）；导出模式下写出第 n-1 帧
extern "C" __declspec(dllexport) int RRW_Mark(void* L) {
  if (!loadLua()) return 0;
  rr_turbo::watchMark(gLua.checkinteger(L, 1));
  return 0;
}

// Lua: RRW_CapFinal(n)   最后一帧（帧上限）的 MC_POST_UPDATE 里调
extern "C" __declspec(dllexport) int RRW_CapFinal(void* L) {
  if (!loadLua()) return 0;
  rr_turbo::captureFinal(gLua.checkinteger(L, 1));
  return 0;
}

// Lua: frames = RRW_CapClose()
extern "C" __declspec(dllexport) int RRW_CapClose(void* L) {
  if (!loadLua()) return 0;
  gLua.pushinteger(L, rr_turbo::captureClose());
  return 1;
}

// Lua: edges = RRW_Keys()   自上次调用以来按下的键（字符串，每字节一个 VK 码）
extern "C" __declspec(dllexport) int RRW_Keys(void* L) {
  if (!loadLua()) return 0;
  const std::string k = rr_turbo::watchKeys();
  gLua.pushlstring(L, k.data(), k.size());
  return 1;
}

// Lua: held = RRW_Held(vk)   0/1
extern "C" __declspec(dllexport) int RRW_Held(void* L) {
  if (!loadLua()) return 0;
  gLua.pushinteger(L, rr_turbo::watchHeld(static_cast<int>(gLua.checkinteger(L, 1))));
  return 1;
}

// Lua: RRW_Shot(path)   下一次 SwapBuffers 前存 BMP（含 Lua GUI）
extern "C" __declspec(dllexport) int RRW_Shot(void* L) {
  if (!loadLua()) return 0;
  rr_turbo::shotNext(gLua.checklstring(L, 1, nullptr));
  return 0;
}

// Lua: RRW_ShotAt(n, path)   第 n 帧最后一次渲染（与导出同口径）存 BMP
extern "C" __declspec(dllexport) int RRW_ShotAt(void* L) {
  if (!loadLua()) return 0;
  rr_turbo::shotAt(gLua.checkinteger(L, 1), gLua.checklstring(L, 2, nullptr));
  return 0;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID) {
  if (reason == DLL_PROCESS_ATTACH) {
    DisableThreadLibraryCalls(instance);
    HANDLE t = CreateThread(nullptr, 0, &initThread, nullptr, 0, nullptr);
    if (t) CloseHandle(t);
  }
  return TRUE;
}
