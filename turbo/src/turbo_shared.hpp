// isaac_turbo 与控制端（Python）共享的控制块布局。
// 单一事实来源：改这里必须同步改 rl/bridge/python/isaac_bridge/turbo.py 里的 struct 格式。
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace isaac_turbo {

constexpr std::uint32_t kMagic = 0x42525449u;  // 'ITRB'
constexpr std::uint32_t kLayoutVersion = 1;
constexpr std::uint32_t kControlBlockBytes = 4096;

// flags：由控制端写、DLL 每次 Hook 调用时读；可在运行中切换。
constexpr std::uint32_t kFlagVirtualClock = 1u << 0;  // glfwGetTime 返回虚拟时钟（每次 Manager::Update 推进一帧）
constexpr std::uint32_t kFlagSkipRender = 1u << 1;    // 跳过整帧渲染函数（render_every / render_request 例外）
constexpr std::uint32_t kFlagFontGuard = 1u << 2;     // 字体绘制前检查 KAGE_ColorTextureShader 是否可用，不可用则跳过本次绘制（缓解）
constexpr std::uint32_t kFlagFileRetry = 1u << 3;     // 归档容器 fopen 失败时短暂重试（缓解）
constexpr std::uint32_t kFlagProbeDump = 1u << 4;     // 首次入栈失败 / 文件打开失败时写带堆的 minidump（取证）

enum Status : std::uint32_t {
  kStatusLoading = 0,
  kStatusActive = 1,
  kStatusIdentityFailed = 2,  // 宿主不是 J460 isaac-ng.exe，未安装任何 Hook
  kStatusHookFailed = 3,
  kStatusShmFailed = 4,
};

// Hook 安装位掩码（hooks_mask）
constexpr std::uint32_t kHookGlfwGetTime = 1u << 0;
constexpr std::uint32_t kHookManagerUpdate = 1u << 1;
constexpr std::uint32_t kHookGameUpdate = 1u << 2;
constexpr std::uint32_t kHookRender = 1u << 3;
constexpr std::uint32_t kHookPushShader = 1u << 4;
constexpr std::uint32_t kHookFontDraw = 1u << 5;
constexpr std::uint32_t kHookFileOpen = 1u << 6;
constexpr std::uint32_t kHookCrtAccess = 1u << 7;  // ucrtbase!_access（不是 J460 目标，按导出名定位）
constexpr std::uint32_t kHookResolvePath = 1u << 8;
constexpr std::uint32_t kHookSearchPath = 1u << 9;
constexpr std::uint32_t kHookSearchMiss = 1u << 10;
constexpr std::uint32_t kHookCaptureOverlay = 1u << 11;  // startup-only, per-process NVIDIA capture isolation

// Game::Update 单次耗时低于此值的调用计入 "fast" 桶：被 Lua 桥接阻塞等待训练器的那一帧会长达秒级，
// 不能混进纯逻辑成本的统计。
constexpr std::uint64_t kFastGameUpdateMicros = 20000;

constexpr std::uint32_t kMaxProbeDumps = 3;  // 每个进程最多写几份探针转储

#pragma pack(push, 1)
struct ControlBlock {
  std::uint32_t magic;           // 0x00
  std::uint32_t version;         // 0x04
  std::uint32_t pid;             // 0x08
  std::uint32_t flags;           // 0x0C  控制端写
  std::uint32_t render_every;    // 0x10  控制端写：跳渲染时每 N 次"会绘制的"渲染调用放行一次（0 = 不放行）
  std::uint32_t render_request;  // 0x14  控制端递增：请求渲染一帧
  std::uint32_t render_done;     // 0x18  DLL 写：已满足的请求号
  std::uint32_t status;          // 0x1C  DLL 写：Status
  std::uint64_t manager_update_calls;  // 0x20  外层循环迭代数（= Manager::Update 调用数）
  std::uint64_t game_update_calls;     // 0x28  逻辑帧数（Game::Update 调用数）
  std::uint64_t game_update_total_us;  // 0x30
  std::uint64_t game_update_max_us;    // 0x38
  std::uint64_t game_update_last_us;   // 0x40
  std::uint64_t render_calls;          // 0x48  真正调用了原渲染函数的次数
  std::uint64_t render_skipped;        // 0x50
  std::uint64_t clock_reads;           // 0x58  glfwGetTime 被调用次数
  double clock_value;                  // 0x60  最近一次返回给游戏的时钟值（秒）
  std::uint64_t virtual_ticks;         // 0x68  虚拟时钟开启期间累计推进的帧数
  std::uint64_t wall_us;               // 0x70  DLL 初始化以来的真实微秒数（Manager::Update 后更新）
  std::uint32_t hooks_mask;            // 0x78
  std::uint32_t draw_parity_ok;        // 0x7C  1 = Manager 计数器自检通过，可判断本次渲染调用是否真的会绘制
  char status_text[128];               // 0x80
  char log_path[256];                  // 0x100
  std::uint64_t game_update_fast_calls;     // 0x200  耗时 < kFastGameUpdateMicros 的 Game::Update 次数
  std::uint64_t game_update_fast_total_us;  // 0x208
  std::uint64_t game_update_min_us;         // 0x210
  std::uint64_t render_opportunities;       // 0x218  跳渲染期间"游戏本会绘制"的渲染调用数
  std::uint32_t counter_checks;             // 0x220  Manager 计数器自检次数
  std::uint32_t counter_mismatches;         // 0x224
  // ---- 崩溃探针（2026-09-20，rl/docs/NATIVE_CRASH_ANALYSIS.md）
  std::uint64_t push_calls;           // 0x228  入栈函数被调用次数
  std::uint64_t push_failures;        // 0x230  入栈返回 0 的次数（A 类崩溃的直接前提）
  std::uint64_t font_calls;           // 0x238  字体绘制被调用次数
  std::uint64_t font_skipped;         // 0x240  字体守卫跳过的绘制次数
  std::uint64_t file_open_calls;      // 0x248  KAGE 文件打开（带 out 参数）次数
  std::uint64_t file_open_failures;   // 0x250  其中返回失败的次数
  std::uint64_t file_open_retries;    // 0x258  缓解层重试次数
  std::uint64_t file_open_retry_ok;   // 0x260  重试后成功次数
  std::uint32_t probe_dumps_written;  // 0x268
  std::uint32_t registry_size_seen;   // 0x26C  最近一次探针检查看到的 shader 注册表元素数
  std::uint32_t font_shader_flags;    // 0x270  最近一次检查看到的 KAGE_ColorTextureShader 对象 flags；0xFFFFFFFF = 未找到
  std::uint32_t probe_reserved;       // 0x274
  char last_fail_text[256];           // 0x278  最近一次失败的摘要（与日志同内容）
  std::uint64_t access_calls;         // 0x378  ucrtbase!_access 对 *.a 归档路径的调用次数
  std::uint64_t access_failures;      // 0x380  其中返回 -1 的次数（B 类崩溃的真正起点：路径解析随后返回空）
  std::uint64_t access_retries;       // 0x388
  std::uint64_t access_retry_ok;      // 0x390
  std::uint8_t reserved[kControlBlockBytes - 0x398];
};
#pragma pack(pop)

static_assert(sizeof(ControlBlock) == kControlBlockBytes, "control block must be exactly one page");
static_assert(offsetof(ControlBlock, manager_update_calls) == 0x20, "layout drift");
static_assert(offsetof(ControlBlock, clock_value) == 0x60, "layout drift");
static_assert(offsetof(ControlBlock, hooks_mask) == 0x78, "layout drift");
static_assert(offsetof(ControlBlock, status_text) == 0x80, "layout drift");
static_assert(offsetof(ControlBlock, log_path) == 0x100, "layout drift");
static_assert(offsetof(ControlBlock, game_update_fast_calls) == 0x200, "layout drift");
static_assert(offsetof(ControlBlock, counter_checks) == 0x220, "layout drift");
static_assert(offsetof(ControlBlock, push_calls) == 0x228, "layout drift");
static_assert(offsetof(ControlBlock, probe_dumps_written) == 0x268, "layout drift");
static_assert(offsetof(ControlBlock, last_fail_text) == 0x278, "layout drift");
static_assert(offsetof(ControlBlock, access_calls) == 0x378, "layout drift");

// 命名对象名。Python 侧 mmap(tagname=...) 使用完全相同的字符串（默认 Local 命名空间）。
inline void formatMappingName(char* out, std::size_t n, std::uint32_t pid) {
  std::snprintf(out, n, "IsaacTurbo.%u", pid);
}

}  // namespace isaac_turbo
