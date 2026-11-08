// rr_turbo：rr_mt.dll 的可选加速模式（只用于回放验证；录制会话默认不安装，行为与第三阶段完全相同）。
// 实现见 rr_turbo.cpp；来源 analysis/j746/port-20260930/modified/turbo.cpp 的 J746 原生时钟/更新钩子与 GPU 提交门控。
// phase6-watch：观战模式（RR_MT_WATCH=1，速率旋钮 / 整次迭代挂起 / 呈现节流）与实录导出（RR_MT_CAPTURE=1，SwapBuffers 前读像素喂 ffmpeg）。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "turbo_shared.hpp"

namespace rr_turbo {

using LogFn = void (*)(const char* fmt, ...);

struct Options {
  bool gpuGate = false;  // RR_MT_TURBO_GPU_GATE=1：整局不提交 DrawElements/DrawArrays/SwapBuffers（phase4 行为）
  bool watch = false;    // RR_MT_WATCH=1：装 GPU 钩子但按迭代决定是否提交；速率由 Lua（RRW_Set）控制
  bool capture = false;  // RR_MT_CAPTURE=1：每次 SwapBuffers 前异步读回后台缓冲，RRW_Mark 时把"上一逻辑帧最后一次渲染"写给 ffmpeg
};

// 在已 MH_Initialize 的进程里安装：glfwGetTime(0x5523B0) → 虚拟时钟、Manager::Update(0x499880) → 推进时钟；
// gpuGate / watch / capture 时另外接管 DrawElements/DrawArrays（libepoxy 已解析的分发槽）与 SwapBuffers（EXE 的 IAT 槽）。
// 调用方负责 J746 身份门（PE 时间戳/SizeOfImage）。成功返回 true 并在 block->flags 置位；失败写 why。
bool install(const std::uint8_t* base, isaac_turbo::ControlBlock* block, const Options& opt, LogFn log, char* why,
             std::size_t whyLen);

// ---- 观战 / 导出（只在主线程、由 Lua C 函数调用）
// rate：1 = 原速（每秒 60 次迭代），0 = 挂起（之后的 Manager::Update 整次跳过，渲染照常 60 Hz），<0 = 不限速，
// 0<rate<1 = 每秒 60 次迭代里只有 rate 比例真正执行 Manager::Update；presentMinMs：两次真正呈现之间的最小墙钟间隔（0 = 每次）。
void watchSet(double rate, double presentMinMs);
struct WatchInfo {
  int available;          // 观战钩子已安装
  int skippedThis;        // 当前迭代的 Manager::Update 被跳过（本次渲染不属于逻辑 tick 结构）
  std::uint64_t iterations, skipped, presents, presentsSkipped;
  double wallMs, rate;
  std::uint64_t capFrames, capDups;
  int capWidth, capHeight, capActive;
  int drawThis;  // 本迭代的渲染会真正呈现（不呈现时 Lua 可以不画 GUI，省 CPU）
};
WatchInfo watchInfo();
// Lua 在每个 MC_POST_UPDATE 调用：n = 刚完成的逻辑帧号（本局第几帧）。导出模式下把"上一次呈现的画面"作为第 n-1 帧写出。
void watchMark(long long n);
// 最后一帧（帧上限 n）已更新：若之后没有逻辑更新，在它的第 2 次呈现时写出并结束导出（见 rr_turbo.cpp）
void captureFinal(long long n);
// 结束导出：关 ffmpeg 的 stdin、等它退出、写 <out>.dll.json。返回写出的帧数。
long long captureClose();
// 观战热键：自上次调用以来的按下沿（每字节一个 Windows VK 码）与当前按住状态（窗口过程子类化收集，挂起时也有效）
std::string watchKeys();
int watchHeld(int vk);
// 下一次 SwapBuffers 前把整帧（含 Lua GUI）存成 BMP（32 位，自下而上）
void shotNext(const char* path);
// 第 n 帧的"最后一次渲染"（与导出同一口径）存成 BMP
void shotAt(long long n, const char* path);

}  // namespace rr_turbo
