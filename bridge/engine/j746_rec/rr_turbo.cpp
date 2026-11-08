// rr_turbo —— rr_mt.dll 的可选加速模式（Repentance v1.7.8a.J746 离线 GOG x86），只给回放验证用。
//
// 默认不安装：只有 rr_mt.dll 看到环境变量 RR_MT_TURBO=1 / RR_MT_WATCH=1 / RR_MT_CAPTURE=1 时才调用 rr_turbo::install()；
// 录制会话不设这些变量，进程里没有任何本文件的钩子（与第三阶段的最小 DLL 行为相同）。
//
// 直接移植 analysis/j746/port-20260930/modified/turbo.cpp 已在 J746 上实机验证过的三件东西（PORT_STATUS.md
// "原生调用边界"与"时钟与 GPU 提交门控"两节；C87 等训练用的同一套）：
//   1. glfwGetTime（RVA 0x5523B0，无参数、double 经 XMM0 返回）→ 虚拟时钟 VirtualClock（rl/turbo/src/virtual_clock.hpp
//      的原样副本）：每次 Manager::Update 完成推进 1/60 s(+1/65536)，读时钟不推进。外层 60 Hz 调度器（RVA 0x475E70）
//      的 Sleep/自旋限帧因此每次迭代都立即满足；主循环结构（每次迭代 1 次 Manager::Update + 1 次 Manager::Render，
//      逻辑每两次迭代 1 帧）不变，只是不再等真实时间。
//   2. Manager::Update（RVA 0x499880，cdecl(void)）后挂钩：在更新完成边界才切换时钟模式（turbo.cpp 记录过的坑：
//      在限帧读时钟时开启会使 tick=0、主循环无限自旋），然后推进一步。
//   3. 可选 GPU 提交门控（RR_MT_TURBO_GPU_GATE=1）：只抑制 DrawElements/DrawArrays（libepoxy 首次解析后写回的分发槽
//      RVA 0x79C6D4 / 0x79B45C）与 SwapBuffers（EXE 的 IAT 槽 RVA 0x5E1014）；CPU 侧渲染（Manager::Render 0x499E20、
//      Lua MC_POST_RENDER、动画、粒子、批次清理与池回收、纹理上传等其余 GL 调用）全部照常执行。不是 render-lite E+F，
//      也没有整段跳过渲染（AB+ 上整段跳过会改变结果）。
// 与 turbo.cpp 的差异：没有 profile/训练 capture/restore/计时等其余钩子；naked 时钟桩额外保存 ECX/EDX/XMM1–7
// （原函数的寄存器约定是 LTCG 自定义的，多保存只会更安全）；计数写进 DLL 日志（每 36000 次 Manager::Update 一行）。
// 确定性边界：本模式只改"什么时候跑下一次迭代"，不改迭代内容与顺序；是否逐帧不变由回放验收实测（phase4-turbo）。
//
// phase6-watch 增加两种模式（同样默认不装）：
//   观战 RR_MT_WATCH=1：钩子同上，另有速率旋钮（Lua 调 RRW_Set）。速率 r：每次迭代开头按墙钟定步（r 倍的 60 Hz）；
//     r=0（挂起）或 0<r<1 的"空迭代"整次跳过原 Manager::Update —— 不轮询输入、不做逻辑/插值、不推进 Manager 计数器，
//     但虚拟时钟照样推进一步（外层限帧才会放行），渲染、Lua MC_POST_RENDER、glfw 事件泵照常 60 Hz，窗口不卡、热键可读。
//     因此逻辑看到的迭代序列与不暂停时逐次相同（只是两次迭代之间多了几次渲染；Lua 用 RRW_Info 的 skippedThis
//     把这些渲染排除在 tick 分类之外）。呈现节流：两次真正 SwapBuffers 之间至少 presentMinMs；绘制每次都提交（见 watchPace）。
//   导出 RR_MT_CAPTURE=1：每次 SwapBuffers 前把默认帧缓冲的后台缓冲异步读进 PBO 环（BGRA），Lua 在 MC_POST_UPDATE 调
//     RRW_Mark(n) 时把"最近一次读回"= 第 n-1 帧最后一次渲染写进 ffmpeg 的 stdin（DLL 自己用 CreateProcess 启动 ffmpeg，
//     命令由 RR_CAPTURE_FFMPEG / RR_CAPTURE_OUT / RR_CAPTURE_VARGS / RR_CAPTURE_LOG 给出）。导出时每次迭代都提交绘制
//     （读回需要完整渲染结果），只节流真正的 SwapBuffers。
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

#include "MinHook.h"
#include "rr_turbo.hpp"
#include "virtual_clock.hpp"

namespace rr_turbo {
void emitLatest(long long f);
long long captureClose();
}  // namespace rr_turbo

namespace {

using namespace isaac_turbo;

constexpr std::uint32_t kRvaClock = 0x5523B0;
constexpr std::uint32_t kRvaUpdate = 0x499880;
constexpr std::uint32_t kRvaGlfwInit = 0x7C767B;       // glfw 已初始化标志（未初始化时原函数返回 0）
constexpr std::uint32_t kRvaTimerQpc = 0x7C8370;       // 1 = 用 QPC
constexpr std::uint32_t kRvaTimerSpt = 0x7C8378;       // double：每 tick 秒数
constexpr std::uint32_t kRvaTimerOffset = 0x7C8380;    // uint64：起点
constexpr std::uint32_t kRvaTimerFn = 0x7C8398;        // 非 QPC 时的计时函数指针
constexpr std::uint32_t kRvaManagerPtr = 0x7C768C;
constexpr std::uint32_t kManagerCounter = 0x4A29C;     // Manager 迭代计数
constexpr std::uint32_t kRvaSlotElements = 0x79C6D4;
constexpr std::uint32_t kRvaSlotArrays = 0x79B45C;
constexpr std::uint32_t kRvaResolverElements = 0x55BE50;  // 未解析时槽里是 libepoxy 的 resolver
constexpr std::uint32_t kRvaResolverArrays = 0x55BA90;
constexpr std::uint32_t kRvaSwapIat = 0x5E1014;

const std::uint8_t* gBase = nullptr;
ControlBlock* gB = nullptr;
rr_turbo::LogFn gLog = nullptr;
VirtualClock gClock;
LARGE_INTEGER gQpf{};
LARGE_INTEGER gQpc0{};

using UpdateFn = void(__cdecl*)();
UpdateFn gOrigUpdate = nullptr;
void* gOrigClock = nullptr;  // 只为满足 MinHook 接口，不回调（原函数的返回值在 XMM0）

using DrawElementsFn = void(WINAPI*)(unsigned, int, unsigned, const void*);
using DrawArraysFn = void(WINAPI*)(unsigned, int, int);
using SwapFn = BOOL(WINAPI*)(HDC);
DrawElementsFn gDrawElements = nullptr;
DrawArraysFn gDrawArrays = nullptr;
SwapFn gSwap = nullptr;
bool gGate = false;       // 装了绘制槽 + SwapBuffers 钩子
bool gGateAll = false;    // phase4：整局按 kFlagSkipRender 门控
std::uint64_t gDraws = 0, gSkippedDraws = 0, gPresents = 0, gSkippedPresents = 0;
std::uint32_t gLastCounter = 0;
std::uint64_t gCounterChecks = 0, gCounterMismatches = 0;

// ---- 观战 / 导出状态（全部只在主线程访问）
bool gWatch = false;       // 速率旋钮生效（watch 或 capture）
bool gCapture = false;
double gRate = 1.0;        // 1 = 原速；0 = 挂起；<0 = 不限速
double gPresentMinUs = 0;  // 两次真正呈现之间的最小墙钟间隔
double gAcc = 0.0;         // 0<rate<1 时的执行配额累加
double gDue = 0.0;         // 下一次迭代的墙钟（微秒）
bool gSkipThis = false;    // 本迭代跳过了原 Manager::Update
bool gDrawThis = true;     // 本迭代的渲染提交绘制
bool gSwapThis = true;     // 本迭代的渲染真正 SwapBuffers
double gLastPresentUs = -1e18;
std::uint64_t gIterations = 0, gSkippedIters = 0;
HANDLE gTimer = nullptr;

std::uint64_t wallMicros() {
  LARGE_INTEGER q;
  QueryPerformanceCounter(&q);
  return static_cast<std::uint64_t>(q.QuadPart - gQpc0.QuadPart) * 1000000ull / static_cast<std::uint64_t>(gQpf.QuadPart);
}

void sleepUntil(double dueUs) {
  for (;;) {
    const double now = static_cast<double>(wallMicros());
    const double left = dueUs - now;
    if (left <= 0) return;
    if (left > 1500 && gTimer) {
      LARGE_INTEGER li;
      li.QuadPart = -static_cast<LONGLONG>((left - 1000) * 10.0);  // 100 ns 单位，负 = 相对
      if (SetWaitableTimer(gTimer, &li, 0, nullptr, nullptr, FALSE)) {
        WaitForSingleObject(gTimer, INFINITE);
        continue;
      }
    }
    if (left > 2500) { Sleep(1); continue; }
    YieldProcessor();
  }
}

// 与 J746 glfwGetTime 相同的公式（turbo.cpp realGlfwTime 的 J746 分支）
double realTime() {
  if (!*(gBase + kRvaGlfwInit)) return 0.0;
  const bool qpc = *(gBase + kRvaTimerQpc) != 0;
  std::uint64_t ticks;
  if (qpc) {
    LARGE_INTEGER q;
    QueryPerformanceCounter(&q);
    ticks = static_cast<std::uint64_t>(q.QuadPart);
  } else {
    using TimerFn = DWORD(WINAPI*)();
    ticks = (*reinterpret_cast<TimerFn const*>(gBase + kRvaTimerFn))();
  }
  return static_cast<double>(ticks - *reinterpret_cast<const std::uint64_t*>(gBase + kRvaTimerOffset)) *
         *reinterpret_cast<const double*>(gBase + kRvaTimerSpt);
}

// ------------------------------------------------------------------ GL 读回（导出与截图）
using GLenum = unsigned;
using GLint = int;
using GLuint = unsigned;
using GLsizei = int;
constexpr GLenum GL_BGRA = 0x80E1, GL_UNSIGNED_BYTE = 0x1401, GL_PIXEL_PACK_BUFFER = 0x88EB, GL_STREAM_READ = 0x88E1,
                 GL_READ_ONLY = 0x88B8, GL_PIXEL_PACK_BUFFER_BINDING = 0x88ED, GL_READ_FRAMEBUFFER = 0x8CA8,
                 GL_READ_FRAMEBUFFER_BINDING = 0x8CAA, GL_READ_BUFFER = 0x0C02, GL_BACK = 0x0405,
                 GL_PACK_ALIGNMENT = 0x0D05, GL_PACK_ROW_LENGTH = 0x0D02, GL_PACK_SKIP_ROWS = 0x0D03,
                 GL_PACK_SKIP_PIXELS = 0x0D04;
struct Gl {
  void(WINAPI* ReadPixels)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*) = nullptr;
  void(WINAPI* GetIntegerv)(GLenum, GLint*) = nullptr;
  void(WINAPI* PixelStorei)(GLenum, GLint) = nullptr;
  void(WINAPI* ReadBuffer)(GLenum) = nullptr;
  void(WINAPI* GenBuffers)(GLsizei, GLuint*) = nullptr;
  void(WINAPI* BindBuffer)(GLenum, GLuint) = nullptr;
  void(WINAPI* BufferData)(GLenum, std::intptr_t, const void*, GLenum) = nullptr;
  void*(WINAPI* MapBuffer)(GLenum, GLenum) = nullptr;
  unsigned char(WINAPI* UnmapBuffer)(GLenum) = nullptr;
  void(WINAPI* BindFramebuffer)(GLenum, GLuint) = nullptr;
  bool tried = false, ok = false, pbo = false;
} gl;

bool loadGl() {
  if (gl.tried) return gl.ok;
  gl.tried = true;
  HMODULE m = GetModuleHandleA("opengl32.dll");
  if (!m) { gLog("capture: opengl32.dll not loaded"); return false; }
  using WglGpa = PROC(WINAPI*)(LPCSTR);
  auto wgl = reinterpret_cast<WglGpa>(GetProcAddress(m, "wglGetProcAddress"));
  auto any = [&](const char* name) -> void* {
    void* p = reinterpret_cast<void*>(GetProcAddress(m, name));
    if (!p && wgl) {
      p = reinterpret_cast<void*>(wgl(name));
      const auto v = reinterpret_cast<std::uintptr_t>(p);
      if (v <= 3 || v == static_cast<std::uintptr_t>(-1)) p = nullptr;
    }
    return p;
  };
  gl.ReadPixels = reinterpret_cast<decltype(gl.ReadPixels)>(any("glReadPixels"));
  gl.GetIntegerv = reinterpret_cast<decltype(gl.GetIntegerv)>(any("glGetIntegerv"));
  gl.PixelStorei = reinterpret_cast<decltype(gl.PixelStorei)>(any("glPixelStorei"));
  gl.ReadBuffer = reinterpret_cast<decltype(gl.ReadBuffer)>(any("glReadBuffer"));
  gl.GenBuffers = reinterpret_cast<decltype(gl.GenBuffers)>(any("glGenBuffers"));
  gl.BindBuffer = reinterpret_cast<decltype(gl.BindBuffer)>(any("glBindBuffer"));
  gl.BufferData = reinterpret_cast<decltype(gl.BufferData)>(any("glBufferData"));
  gl.MapBuffer = reinterpret_cast<decltype(gl.MapBuffer)>(any("glMapBuffer"));
  gl.UnmapBuffer = reinterpret_cast<decltype(gl.UnmapBuffer)>(any("glUnmapBuffer"));
  gl.BindFramebuffer = reinterpret_cast<decltype(gl.BindFramebuffer)>(any("glBindFramebuffer"));
  gl.ok = gl.ReadPixels && gl.GetIntegerv && gl.PixelStorei;
  gl.pbo = gl.ok && gl.GenBuffers && gl.BindBuffer && gl.BufferData && gl.MapBuffer && gl.UnmapBuffer;
  gLog("capture: GL ok=%d pbo=%d bindfb=%d", gl.ok ? 1 : 0, gl.pbo ? 1 : 0, gl.BindFramebuffer ? 1 : 0);
  return gl.ok;
}

struct PackState {
  GLint pack = 0, readFb = 0, readBuf = 0, align = 4, rowLen = 0, skipRows = 0, skipPix = 0;
};

void beginRead(PackState& s) {
  gl.GetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &s.pack);
  if (gl.BindFramebuffer) {
    gl.GetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &s.readFb);
    if (s.readFb != 0) gl.BindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  }
  gl.GetIntegerv(GL_READ_BUFFER, &s.readBuf);
  if (gl.ReadBuffer && s.readBuf != static_cast<GLint>(GL_BACK)) gl.ReadBuffer(GL_BACK);
  gl.GetIntegerv(GL_PACK_ALIGNMENT, &s.align);
  gl.GetIntegerv(GL_PACK_ROW_LENGTH, &s.rowLen);
  gl.GetIntegerv(GL_PACK_SKIP_ROWS, &s.skipRows);
  gl.GetIntegerv(GL_PACK_SKIP_PIXELS, &s.skipPix);
  gl.PixelStorei(GL_PACK_ALIGNMENT, 4);
  gl.PixelStorei(GL_PACK_ROW_LENGTH, 0);
  gl.PixelStorei(GL_PACK_SKIP_ROWS, 0);
  gl.PixelStorei(GL_PACK_SKIP_PIXELS, 0);
}

void endRead(const PackState& s) {
  gl.PixelStorei(GL_PACK_ALIGNMENT, s.align);
  gl.PixelStorei(GL_PACK_ROW_LENGTH, s.rowLen);
  gl.PixelStorei(GL_PACK_SKIP_ROWS, s.skipRows);
  gl.PixelStorei(GL_PACK_SKIP_PIXELS, s.skipPix);
  if (gl.ReadBuffer && s.readBuf != static_cast<GLint>(GL_BACK)) gl.ReadBuffer(static_cast<GLenum>(s.readBuf));
  if (gl.BindFramebuffer && s.readFb != 0) gl.BindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(s.readFb));
  if (gl.BindBuffer) gl.BindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<GLuint>(s.pack));
}

constexpr int kRing = 3;
GLuint gPbo[kRing] = {};
int gPboW = 0, gPboH = 0;
int gLatest = -1;                 // 最近一次读回所在的槽
std::uint64_t gLatestSerial = 0;  // 最近一次读回对应的 SwapBuffers 序号
std::uint64_t gLastEmitSerial = 0;
bool gReadArmed = false;          // 有导出或 shotAt 待办：每次 SwapBuffers 都读回

// 观战热键：子类化游戏窗口过程，记下按键（VK 码）。挂起时原 Manager::Update 整次跳过，引擎的键盘状态（Lua 的
// Input.IsButtonPressed）不再更新，但外层循环的 glfw 事件泵照常派发窗口消息，所以热键在这里收。消息继续交给 GLFW。
HWND gHwnd = nullptr;
WNDPROC gOrigWndProc = nullptr;
bool gKeyHeld[256] = {};
std::string gKeyEdges;  // 自上次 RRW_Keys 以来的按下沿（每字节一个 VK）

bool gCapActive = false, gCapClosed = false, gCapFailed = false;
HANDLE gFfIn = nullptr, gFfProc = nullptr;
std::uint64_t gCapFrames = 0, gCapDups = 0, gCapWriteUs = 0, gCapMapUs = 0, gCapReads = 0;
long long gCapFirst = -1, gCapLast = -1;
int gCapW = 0, gCapH = 0;
std::string gCapOut, gCapFfmpeg, gCapVargs, gCapLog;
std::map<long long, std::string> gShotAt;
std::string gShotNext;
long long gFinalFrame = 0;        // RRW_CapFinal：最后一帧（帧上限）已更新，等它的第 2 次呈现或下一次 Mark
std::uint64_t gFinalPresents = 0;

std::string envStr(const char* k) {
  char buf[2048] = {};
  const DWORD n = GetEnvironmentVariableA(k, buf, sizeof(buf));
  if (!n || n >= sizeof(buf)) return {};
  return std::string(buf, n);
}

bool writeBmp(const char* path, const std::uint8_t* bgra, int w, int h) {
  FILE* f = std::fopen(path, "wb");
  if (!f) return false;
  const std::uint32_t img = static_cast<std::uint32_t>(w) * static_cast<std::uint32_t>(h) * 4u;
  std::uint8_t hdr[54] = {};
  hdr[0] = 'B'; hdr[1] = 'M';
  const std::uint32_t fsz = 54 + img, off = 54, ih = 40, ww = static_cast<std::uint32_t>(w), hh = static_cast<std::uint32_t>(h);
  std::memcpy(hdr + 2, &fsz, 4);
  std::memcpy(hdr + 10, &off, 4);
  std::memcpy(hdr + 14, &ih, 4);
  std::memcpy(hdr + 18, &ww, 4);
  std::memcpy(hdr + 22, &hh, 4);  // 正高度 = 自下而上（与 glReadPixels 相同的行序）
  const std::uint16_t planes = 1, bpp = 32;
  std::memcpy(hdr + 26, &planes, 2);
  std::memcpy(hdr + 28, &bpp, 2);
  std::memcpy(hdr + 34, &img, 4);
  std::fwrite(hdr, 1, 54, f);
  std::fwrite(bgra, 1, img, f);
  std::fclose(f);
  return true;
}

bool clientSize(HDC dc, int* w, int* h) {
  HWND wnd = WindowFromDC(dc);
  RECT r{};
  if (!wnd || !GetClientRect(wnd, &r)) return false;
  *w = r.right - r.left;
  *h = r.bottom - r.top;
  return *w > 0 && *h > 0;
}

void ensurePbo(int w, int h) {
  if (gPbo[0] && w == gPboW && h == gPboH) return;
  if (!gPbo[0]) gl.GenBuffers(kRing, gPbo);
  GLint prev = 0;
  gl.GetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &prev);
  for (int i = 0; i < kRing; ++i) {
    gl.BindBuffer(GL_PIXEL_PACK_BUFFER, gPbo[i]);
    gl.BufferData(GL_PIXEL_PACK_BUFFER, static_cast<std::intptr_t>(w) * h * 4, nullptr, GL_STREAM_READ);
  }
  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<GLuint>(prev));
  gPboW = w;
  gPboH = h;
  gLatest = -1;
  gLog("capture: PBO ring %d x %dx%d", kRing, w, h);
}

// SwapBuffers 之前：把后台缓冲异步读进下一个 PBO 槽
void readbackAsync(HDC dc) {
  if (!loadGl() || !gl.pbo) return;
  int w = 0, h = 0;
  if (!clientSize(dc, &w, &h)) return;
  PackState s;
  beginRead(s);
  ensurePbo(w, h);
  const int slot = (gLatest + 1) % kRing;
  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, gPbo[slot]);
  gl.ReadPixels(0, 0, w, h, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
  endRead(s);
  gLatest = slot;
  gLatestSerial = gPresents;
  ++gCapReads;
}

void shotNow(HDC dc) {
  if (!loadGl()) return;
  int w = 0, h = 0;
  if (!clientSize(dc, &w, &h)) return;
  std::string buf(static_cast<std::size_t>(w) * h * 4, '\0');
  PackState s;
  beginRead(s);
  if (gl.BindBuffer) gl.BindBuffer(GL_PIXEL_PACK_BUFFER, 0);
  gl.ReadPixels(0, 0, w, h, GL_BGRA, GL_UNSIGNED_BYTE, buf.data());
  endRead(s);
  const bool ok = writeBmp(gShotNext.c_str(), reinterpret_cast<const std::uint8_t*>(buf.data()), w, h);
  gLog("shot: %s %dx%d ok=%d present=%llu", gShotNext.c_str(), w, h, ok ? 1 : 0, static_cast<unsigned long long>(gPresents));
  gShotNext.clear();
}

bool startFfmpeg(int w, int h) {
  if (gCapFfmpeg.empty() || gCapOut.empty()) { gLog("capture: RR_CAPTURE_FFMPEG/OUT missing"); return false; }
  SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
  HANDLE rd = nullptr, wr = nullptr;
  if (!CreatePipe(&rd, &wr, &sa, 1 << 24)) { gLog("capture: CreatePipe %lu", GetLastError()); return false; }
  SetHandleInformation(wr, HANDLE_FLAG_INHERIT, 0);
  HANDLE lf = INVALID_HANDLE_VALUE;
  if (!gCapLog.empty())
    lf = CreateFileA(gCapLog.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  char cmd[4096];
  std::snprintf(cmd, sizeof(cmd),
                "\"%s\" -hide_banner -nostats -loglevel warning -y -f rawvideo -pix_fmt bgra -s %dx%d -framerate 30 "
                "-i pipe:0 -vf vflip %s \"%s\"",
                gCapFfmpeg.c_str(), w, h, gCapVargs.c_str(), gCapOut.c_str());
  STARTUPINFOA si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = rd;
  si.hStdOutput = lf != INVALID_HANDLE_VALUE ? lf : nullptr;
  si.hStdError = lf != INVALID_HANDLE_VALUE ? lf : nullptr;
  PROCESS_INFORMATION pi{};
  const BOOL ok = CreateProcessA(nullptr, cmd, nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
  const DWORD err = GetLastError();
  CloseHandle(rd);
  if (lf != INVALID_HANDLE_VALUE) CloseHandle(lf);
  gLog("capture: start ffmpeg ok=%d err=%lu pid=%lu cmd=%s", ok ? 1 : 0, ok ? 0ul : err,
       ok ? static_cast<unsigned long>(pi.dwProcessId) : 0ul, cmd);
  if (!ok) { CloseHandle(wr); return false; }
  CloseHandle(pi.hThread);
  gFfIn = wr;
  gFfProc = pi.hProcess;
  gCapW = w;
  gCapH = h;
  gCapActive = true;
  return true;
}

bool writeAll(HANDLE h, const std::uint8_t* p, std::size_t n) {
  while (n) {
    DWORD w = 0;
    const DWORD chunk = n > (1u << 22) ? (1u << 22) : static_cast<DWORD>(n);
    if (!WriteFile(h, p, chunk, &w, nullptr) || w == 0) return false;
    p += w;
    n -= w;
  }
  return true;
}

void writeCaptureJson(DWORD ffExit) {
  if (gCapOut.empty()) return;
  const std::string path = gCapOut + ".dll.json";
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) return;
  std::fprintf(f,
               "{\"frames\":%llu,\"first_frame\":%lld,\"last_frame\":%lld,\"dups\":%llu,\"width\":%d,\"height\":%d,"
               "\"ffmpeg_exit\":%lu,\"failed\":%s,\"readbacks\":%llu,\"write_ms\":%.1f,\"map_ms\":%.1f,\"presents\":%llu,"
               "\"presents_skipped\":%llu}\n",
               static_cast<unsigned long long>(gCapFrames), gCapFirst, gCapLast, static_cast<unsigned long long>(gCapDups),
               gCapW, gCapH, static_cast<unsigned long>(ffExit), gCapFailed ? "true" : "false",
               static_cast<unsigned long long>(gCapReads), gCapWriteUs / 1000.0, gCapMapUs / 1000.0,
               static_cast<unsigned long long>(gPresents), static_cast<unsigned long long>(gSkippedPresents));
  std::fclose(f);
}

LRESULT CALLBACK watchWndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
  if ((m == WM_KEYDOWN || m == WM_SYSKEYDOWN) && w < 256) {
    if (!(l & (1L << 30)) && !gKeyHeld[w] && gKeyEdges.size() < 128) gKeyEdges.push_back(static_cast<char>(w));
    gKeyHeld[w] = true;
  } else if ((m == WM_KEYUP || m == WM_SYSKEYUP) && w < 256) {
    gKeyHeld[w] = false;
  } else if (m == WM_KILLFOCUS) {
    std::memset(gKeyHeld, 0, sizeof(gKeyHeld));
  }
  return CallWindowProcW(gOrigWndProc, h, m, w, l);
}

void hookWindow(HDC dc) {
  if (gHwnd) return;
  HWND w = WindowFromDC(dc);
  if (!w) return;
  gHwnd = w;
  gOrigWndProc = reinterpret_cast<WNDPROC>(SetWindowLongW(w, GWL_WNDPROC, reinterpret_cast<LONG>(&watchWndProc)));
  gLog("watch: window %p subclassed (orig wndproc %p)", reinterpret_cast<void*>(w), reinterpret_cast<void*>(gOrigWndProc));
}

// ------------------------------------------------------------------ 钩子
bool skipDraws() { return gWatch ? !gDrawThis : (gB->flags & kFlagSkipRender) != 0; }

void WINAPI hookDrawElements(unsigned mode, int count, unsigned type, const void* indices) {
  ++gDraws;
  if (skipDraws()) { ++gSkippedDraws; return; }
  gDrawElements(mode, count, type, indices);
}

void WINAPI hookDrawArrays(unsigned mode, int first, int count) {
  ++gDraws;
  if (skipDraws()) { ++gSkippedDraws; return; }
  gDrawArrays(mode, first, count);
}

BOOL WINAPI hookSwap(HDC dc) {
  ++gPresents;
  if (gWatch) {
    if (!gHwnd) hookWindow(dc);
    if (gDrawThis) {
      if (gReadArmed) readbackAsync(dc);
      if (!gShotNext.empty()) shotNow(dc);
    }
    if (gFinalFrame && !gCapClosed && ++gFinalPresents >= 2) {
      const long long f = gFinalFrame;
      gFinalFrame = 0;
      gLog("capture: final frame %lld emitted at its 2nd present (no further logic update)", f);
      rr_turbo::emitLatest(f);
      rr_turbo::captureClose();
    }
    if (!gSwapThis) { ++gSkippedPresents; return TRUE; }
    gLastPresentUs = static_cast<double>(wallMicros());
    return gSwap(dc);
  }
  if (gB->flags & kFlagSkipRender) { ++gSkippedPresents; return TRUE; }
  return gSwap(dc);
}

// libepoxy 首次使用时用真实函数替换 resolver；等它解析完再接管，否则第一次转发会把我们的门控覆盖掉。
void gateResolvedDrawSlots() {
  auto* elements = reinterpret_cast<DrawElementsFn*>(const_cast<std::uint8_t*>(gBase) + kRvaSlotElements);
  auto* arrays = reinterpret_cast<DrawArraysFn*>(const_cast<std::uint8_t*>(gBase) + kRvaSlotArrays);
  if (!gDrawElements && reinterpret_cast<const void*>(*elements) != gBase + kRvaResolverElements) {
    gDrawElements = *elements;
    *elements = &hookDrawElements;
    gLog("turbo: DrawElements slot gated (original=%p)", reinterpret_cast<void*>(gDrawElements));
  }
  if (!gDrawArrays && reinterpret_cast<const void*>(*arrays) != gBase + kRvaResolverArrays) {
    gDrawArrays = *arrays;
    *arrays = &hookDrawArrays;
    gLog("turbo: DrawArrays slot gated (original=%p)", reinterpret_cast<void*>(gDrawArrays));
  }
}

// 观战/导出：迭代开头按速率定步，决定本迭代是否执行原 Manager::Update、是否提交绘制、是否真正呈现
void watchPace() {
  constexpr double kPeriodUs = 1e6 / 60.0;
  const double r = gRate;
  double period = kPeriodUs;
  bool skip = false;
  if (r == 0.0) {
    skip = true;
  } else if (r < 0.0) {
    period = 0.0;
  } else if (r < 1.0) {
    gAcc += r;
    if (gAcc >= 1.0 - 1e-9) gAcc -= 1.0; else skip = true;
  } else {
    period = kPeriodUs / r;
  }
  if (period > 0.0) {
    double now = static_cast<double>(wallMicros());
    if (gDue <= 0.0 || now - gDue > 250000.0) gDue = now;  // 落后太多（加载、断点）就不追
    if (gDue > now) sleepUntil(gDue);
    gDue += period;
  } else {
    gDue = 0.0;
  }
  gSkipThis = skip;
  const double now = static_cast<double>(wallMicros());
  const bool presentDue = gPresentMinUs <= 0.0 || now - gLastPresentUs >= gPresentMinUs || !gShotNext.empty();
  gSwapThis = presentDue;
  // 观战/导出每次迭代都提交绘制，只节流真正的 SwapBuffers。phase6 实测：按迭代门控绘制（不呈现的迭代不画）时，
  // 观战快进的整局回放出现过单帧的网格读数异常（full 层孤立分歧），每次都画的 4 个整局回放（导出 / 参考帧）没有；
  // 而且门控并没有让观战快进更快（39,500 帧都约 97 s）。
  gDrawThis = true;
  ++gIterations;
  if (skip) ++gSkippedIters;
}

// Manager::Update 是 cdecl(void)。提前返回的路径也要推进：外层 60 Hz 调度器必须继续前进。
// 挂起的空迭代：原 Manager::Update 不跑，而 J746 的窗口消息泵在它里面（实测：挂起后窗口收不到键），
// 所以这里自己泵一次（与 glfwPollEvents 相同的 Peek/Translate/Dispatch），窗口不会"未响应"，热键（watchWndProc）照收。
std::uint64_t gPumped = 0;
void pumpMessages() {
  MSG msg;
  int k = 0;
  while (k < 64 && PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
    ++k;
    if (msg.message == WM_QUIT) { PostQuitMessage(static_cast<int>(msg.wParam)); break; }
    TranslateMessage(&msg);
    DispatchMessageW(&msg);
  }
  gPumped += static_cast<std::uint64_t>(k);
}

void __cdecl hookUpdate() {
  if (gWatch) watchPace();
  if (!gSkipThis) gOrigUpdate();
  else pumpMessages();
  // 在更新完成边界切换模式（不在任意一次读时钟时切换）
  const bool want = (gB->flags & kFlagVirtualClock) != 0;
  if (want != gClock.enabled) {
    const double real = realTime();
    if (want) {
      gClock.enable(real);
      gLog("turbo: virtual clock ON base=%.6f real=%.6f", gClock.base, real);
    } else {
      gClock.disable(real);
      gLog("turbo: virtual clock OFF offset=%.6f real=%.6f ticks=%llu", gClock.offset, real,
           static_cast<unsigned long long>(gClock.ticks));
    }
  }
  gClock.tick();
  if (gGate) gateResolvedDrawSlots();
  const std::uint8_t* m = *reinterpret_cast<const std::uint8_t* const*>(gBase + kRvaManagerPtr);
  if (m && !gSkipThis) {
    const std::uint32_t c = *reinterpret_cast<const std::uint32_t*>(m + kManagerCounter);
    if (gCounterChecks && c != gLastCounter + 1) ++gCounterMismatches;
    gLastCounter = c;
    ++gCounterChecks;
  }
  const std::uint64_t calls = ++gB->manager_update_calls;
  gB->virtual_ticks = gClock.ticks;
  gB->wall_us = wallMicros();
  gB->counter_checks = static_cast<std::uint32_t>(gCounterChecks);
  gB->counter_mismatches = static_cast<std::uint32_t>(gCounterMismatches);
  gB->render_calls = gDraws;
  gB->render_skipped = gSkippedDraws;
  if (calls % 36000 == 0) {
    gLog("turbo: manager_updates=%llu virtual_ticks=%llu wall_s=%.1f clock_reads=%llu counter_mismatch=%llu draws=%llu/%llu "
         "skipped presents=%llu/%llu skipped watch_skipped_iters=%llu rate=%.3f cap_frames=%llu",
         static_cast<unsigned long long>(calls), static_cast<unsigned long long>(gClock.ticks), gB->wall_us / 1e6,
         static_cast<unsigned long long>(gB->clock_reads), static_cast<unsigned long long>(gCounterMismatches),
         static_cast<unsigned long long>(gDraws), static_cast<unsigned long long>(gSkippedDraws),
         static_cast<unsigned long long>(gPresents), static_cast<unsigned long long>(gSkippedPresents),
         static_cast<unsigned long long>(gSkippedIters), gRate, static_cast<unsigned long long>(gCapFrames));
  }
}

}  // namespace

extern "C" {
double g_rr_turbo_now = 0.0;

void __cdecl rr_turbo_clock_read(void) {
  g_rr_turbo_now = gClock.read(realTime());
  gB->clock_reads++;
  gB->clock_value = g_rr_turbo_now;
}

// 原函数无参数、结果在 XMM0（LTCG 自定义约定）。C 写不出"返回 double 到 XMM0"，用 naked 桩：保存 ECX/EDX/XMM1–7，
// 调 C 计算，恢复，再把结果装进 XMM0。
__attribute__((naked)) void rr_turbo_hook_clock(void) {
  __asm__ volatile(
      "pushl %ecx\n\t"
      "pushl %edx\n\t"
      "subl $112, %esp\n\t"
      "movups %xmm1, (%esp)\n\t"
      "movups %xmm2, 16(%esp)\n\t"
      "movups %xmm3, 32(%esp)\n\t"
      "movups %xmm4, 48(%esp)\n\t"
      "movups %xmm5, 64(%esp)\n\t"
      "movups %xmm6, 80(%esp)\n\t"
      "movups %xmm7, 96(%esp)\n\t"
      "call _rr_turbo_clock_read\n\t"
      "movups (%esp), %xmm1\n\t"
      "movups 16(%esp), %xmm2\n\t"
      "movups 32(%esp), %xmm3\n\t"
      "movups 48(%esp), %xmm4\n\t"
      "movups 64(%esp), %xmm5\n\t"
      "movups 80(%esp), %xmm6\n\t"
      "movups 96(%esp), %xmm7\n\t"
      "addl $112, %esp\n\t"
      "popl %edx\n\t"
      "popl %ecx\n\t"
      "movsd _g_rr_turbo_now, %xmm0\n\t"
      "ret\n\t");
}
}

namespace rr_turbo {

bool install(const std::uint8_t* base, ControlBlock* block, const Options& opt, LogFn log, char* why, std::size_t whyLen) {
  gBase = base;
  gB = block;
  gLog = log;
  QueryPerformanceFrequency(&gQpf);
  QueryPerformanceCounter(&gQpc0);
  // J746 自己的原始序言（不是从 J460 继承的）；身份门（时间戳/SizeOfImage）由调用方先做
  const unsigned char clockBytes[] = {0x55, 0x8B, 0xEC, 0x83, 0xEC, 0x08};
  const unsigned char updateBytes[] = {0x55, 0x8B, 0xEC, 0x83, 0xE4, 0xF8};
  if (std::memcmp(base + kRvaClock, clockBytes, sizeof(clockBytes)) ||
      std::memcmp(base + kRvaUpdate, updateBytes, sizeof(updateBytes))) {
    std::snprintf(why, whyLen, "ERROR:turbo clock/update prologue mismatch");
    return false;
  }
  void* clockTarget = const_cast<std::uint8_t*>(base + kRvaClock);
  void* updateTarget = const_cast<std::uint8_t*>(base + kRvaUpdate);
  MH_STATUS st = MH_CreateHook(clockTarget, reinterpret_cast<void*>(&rr_turbo_hook_clock), &gOrigClock);
  if (st != MH_OK) { std::snprintf(why, whyLen, "ERROR:turbo clock hook=%d", st); return false; }
  st = MH_CreateHook(updateTarget, reinterpret_cast<void*>(&hookUpdate), reinterpret_cast<void**>(&gOrigUpdate));
  if (st != MH_OK) { std::snprintf(why, whyLen, "ERROR:turbo update hook=%d", st); return false; }
  const bool hookGpu = opt.gpuGate || opt.watch || opt.capture;
  if (hookGpu) {
    auto* slot = reinterpret_cast<SwapFn*>(const_cast<std::uint8_t*>(base) + kRvaSwapIat);
    DWORD oldProtect = 0;
    if (!VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE, &oldProtect)) {
      std::snprintf(why, whyLen, "ERROR:turbo swap IAT protect=%lu", GetLastError());
      return false;
    }
    gSwap = *slot;
    *slot = &hookSwap;
    DWORD ignored = 0;
    VirtualProtect(slot, sizeof(*slot), oldProtect, &ignored);
    gGate = true;
  }
  gWatch = opt.watch || opt.capture;
  gCapture = opt.capture;
  gGateAll = opt.gpuGate && !gWatch;
  if (gWatch) {
    gTimer = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /* CREATE_WAITABLE_TIMER_HIGH_RESOLUTION */,
                                    TIMER_ALL_ACCESS);
    if (!gTimer) gTimer = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
    // 只导出（没有观战）：不限速，每 250 ms 真正呈现一次让窗口能看到进度
    if (opt.capture && !opt.watch) { gRate = -1.0; gPresentMinUs = 250000.0; }
  }
  if (gCapture) {
    gCapOut = envStr("RR_CAPTURE_OUT");
    gCapFfmpeg = envStr("RR_CAPTURE_FFMPEG");
    gCapVargs = envStr("RR_CAPTURE_VARGS");
    gCapLog = envStr("RR_CAPTURE_LOG");
    if (gCapVargs.empty()) gCapVargs = "-c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p";
    // 读回从回放局第一次 RRW_Mark 起才开（菜单与 harness 的第一个局不读）
  }
  // 时钟从下一次 Manager::Update 完成边界起切到虚拟（hookUpdate 里），之前保持真实时间
  block->flags |= kFlagVirtualClock | (gGateAll ? kFlagSkipRender : 0u);
  st = MH_EnableHook(clockTarget);
  if (st == MH_OK) st = MH_EnableHook(updateTarget);
  if (st != MH_OK) { std::snprintf(why, whyLen, "ERROR:turbo enable=%d", st); return false; }
  block->hooks_mask |= kHookGlfwGetTime | kHookManagerUpdate | (hookGpu ? kHookRender : 0u);
  log("turbo: installed clock=5523B0 update=499880 gpu_gate=%d watch=%d capture=%d (DrawElements/DrawArrays slots + "
      "SwapBuffers IAT; CPU render retained) flags=0x%X timer=%p",
      opt.gpuGate ? 1 : 0, opt.watch ? 1 : 0, opt.capture ? 1 : 0, block->flags, gTimer);
  return true;
}

void watchSet(double rate, double presentMinMs) {
  if (!gWatch) return;
  if (rate != gRate) {
    gLog("watch: rate %.3f -> %.3f present_min_ms=%.1f iter=%llu", gRate, rate, presentMinMs,
         static_cast<unsigned long long>(gIterations));
    if (rate > 0.0 && rate < 1.0) gAcc = 1.0 - 1e-9;  // 第一次就执行，单步/慢放立即可见
  }
  gRate = rate;
  gPresentMinUs = presentMinMs > 0 ? presentMinMs * 1000.0 : 0.0;
}

WatchInfo watchInfo() {
  WatchInfo w{};
  w.available = gWatch ? 1 : 0;
  w.skippedThis = gSkipThis ? 1 : 0;
  w.iterations = gIterations;
  w.skipped = gSkippedIters;
  w.presents = gPresents - gSkippedPresents;
  w.presentsSkipped = gSkippedPresents;
  w.wallMs = gQpf.QuadPart ? wallMicros() / 1000.0 : 0.0;
  w.rate = gRate;
  w.capFrames = gCapFrames;
  w.capDups = gCapDups;
  w.capWidth = gCapW;
  w.capHeight = gCapH;
  w.capActive = gCapActive && !gCapClosed ? 1 : 0;
  w.drawThis = gSwapThis ? 1 : 0;  // Lua 用它决定画不画 GUI：这次渲染真正呈现才画
  return w;
}

void watchMark(long long n) {
  if (!gWatch) return;
  if (gCapture && !gCapClosed) gReadArmed = true;
  const long long f = n - 1;  // 刚结束的那一帧（它最后一次渲染 = 最近一次读回）
  if (f < 1) { gLastEmitSerial = gLatestSerial; return; }
  emitLatest(f);
  if (gFinalFrame && f >= gFinalFrame) gFinalFrame = 0;
}

void emitLatest(long long f) {
  auto shot = gShotAt.find(f);
  const bool wantCap = gCapture && !gCapClosed && !gCapFailed;
  if (!wantCap && shot == gShotAt.end()) return;
  if (gLatest < 0 || !gl.pbo) return;
  if (gLatestSerial == gLastEmitSerial) ++gCapDups;  // 两次 Mark 之间没有新的渲染（不应出现）
  gLastEmitSerial = gLatestSerial;
  const std::uint64_t t0 = wallMicros();
  GLint prev = 0;
  gl.GetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &prev);
  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, gPbo[gLatest]);
  const auto* px = static_cast<const std::uint8_t*>(gl.MapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY));
  const std::uint64_t t1 = wallMicros();
  gCapMapUs += t1 - t0;
  if (px) {
    if (wantCap) {
      if (!gCapActive && !startFfmpeg(gPboW, gPboH)) gCapFailed = true;
      if (gCapActive) {
        if (gPboW != gCapW || gPboH != gCapH) {
          gLog("capture: size changed %dx%d -> %dx%d at frame %lld", gCapW, gCapH, gPboW, gPboH, f);
          gCapFailed = true;
        } else if (!writeAll(gFfIn, px, static_cast<std::size_t>(gCapW) * gCapH * 4)) {
          gLog("capture: write failed at frame %lld err=%lu", f, GetLastError());
          gCapFailed = true;
        } else {
          if (gCapFirst < 0) gCapFirst = f;
          if (gCapLast >= 0 && f != gCapLast + 1) gLog("capture: frame gap %lld -> %lld", gCapLast, f);
          gCapLast = f;
          ++gCapFrames;
        }
      }
    }
    if (shot != gShotAt.end()) {
      const bool ok = writeBmp(shot->second.c_str(), px, gPboW, gPboH);
      gLog("shot_at: frame %lld -> %s ok=%d", f, shot->second.c_str(), ok ? 1 : 0);
      gShotAt.erase(shot);
    }
    gl.UnmapBuffer(GL_PIXEL_PACK_BUFFER);
  }
  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<GLuint>(prev));
  gCapWriteUs += wallMicros() - t1;
  if (!gCapture && gShotAt.empty()) gReadArmed = false;
}

// Lua 在最后一帧（帧上限 n）的 MC_POST_UPDATE 调：之后若游戏不再做逻辑更新（录制以暂停菜单退出结束时，回放局关掉后
// 游戏回到主菜单，MC_POST_UPDATE / MC_POST_RENDER 都不再来），就在这一帧之后的第 2 次呈现（正常每个逻辑帧 2 次渲染，
// 第 2 次就是"最后一次渲染"）写出第 n 帧并结束导出；若先来了下一次更新，照常由 RRW_Mark(n+1) 写出。
void captureFinal(long long n) {
  if (!gCapture || gCapClosed) return;
  gFinalFrame = n;
  gFinalPresents = 0;
  gLog("capture: final frame %lld armed", n);
}

long long captureClose() {
  if (!gCapture || gCapClosed) return static_cast<long long>(gCapFrames);
  gCapClosed = true;
  DWORD code = 0xFFFFFFFFu;
  if (gFfIn) { CloseHandle(gFfIn); gFfIn = nullptr; }
  if (gFfProc) {
    WaitForSingleObject(gFfProc, 300000);
    GetExitCodeProcess(gFfProc, &code);
    CloseHandle(gFfProc);
    gFfProc = nullptr;
  }
  gReadArmed = !gShotAt.empty();
  gLog("capture: closed frames=%llu first=%lld last=%lld dups=%llu ffmpeg_exit=%lu failed=%d",
       static_cast<unsigned long long>(gCapFrames), gCapFirst, gCapLast, static_cast<unsigned long long>(gCapDups),
       static_cast<unsigned long>(code), gCapFailed ? 1 : 0);
  writeCaptureJson(code);
  return static_cast<long long>(gCapFrames);
}

std::string watchKeys() {
  std::string out;
  out.swap(gKeyEdges);
  return out;
}

int watchHeld(int vk) { return (vk >= 0 && vk < 256 && gKeyHeld[vk]) ? 1 : 0; }

void shotNext(const char* path) {
  if (!gWatch || !path) return;
  gShotNext = path;
}

void shotAt(long long n, const char* path) {
  if (!gWatch || !path) return;
  gShotAt[n] = path;
  gReadArmed = true;
}

}  // namespace rr_turbo
