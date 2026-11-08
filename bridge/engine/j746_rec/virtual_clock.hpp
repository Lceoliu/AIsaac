// 虚拟时钟：替代 glfwGetTime 返回值的纯逻辑，无 Windows 依赖，可在 tests/clock_test.cpp 里单独验证。
//
// 设计约束（来自 J460 外层帧循环 RVA 0x00531050 的反汇编，见 analysis/docs/J460_FRAME_LOOP.md）：
//  * 帧循环每次迭代读三处时钟：迭代开头（frame_start、fps 统计）、限帧计算 ms=(period-elapsed)*-1000、
//    自旋 while (period > now - frame_start)。中间平台层 FUN_00a714e0 再读一次算输入设备 delta。
//  * 虚拟时钟只在 Manager::Update 完成时推进一步（tick），读多少次都不推进 —— 这就是
//    "不能每读取一次时钟就推进虚拟时间"。每步比帧期长 1/65536，保证 elapsed >= period 在浮点下恒成立，
//    自旋一次退出、Sleep 毫秒为 0，fps 显示约 60。
//  * 任何时候返回值单调不减：关掉虚拟时钟时用 real+offset 续接，绝不回跳（回跳会让游戏 Sleep 几分钟或自旋等真实时间追上）。
#pragma once

#include <cstdint>

namespace isaac_turbo {

constexpr double kFramePeriodSeconds = 1.0 / 60.0;  // DAT_00baa498 = 0.016666666666666666
constexpr double kVirtualStepSeconds = kFramePeriodSeconds * (1.0 + 1.0 / 65536.0);

struct VirtualClock {
  bool enabled = false;
  double base = 0.0;        // 开启时刻的时钟值
  std::uint64_t ticks = 0;  // 开启以来 Manager::Update 完成次数
  double offset = 0.0;      // 关闭状态下 real + offset，保证续接
  double last = 0.0;        // 最近返回值（单调保护）
  bool ever_read = false;

  double read(double real_now) {
    double v = enabled ? base + static_cast<double>(ticks) * kVirtualStepSeconds : real_now + offset;
    if (ever_read && v < last) v = last;
    last = v;
    ever_read = true;
    return v;
  }

  void tick() {
    if (enabled) ++ticks;
  }

  void enable(double real_now) {
    if (enabled) return;
    double cur = real_now + offset;
    if (ever_read && last > cur) cur = last;
    base = cur;
    ticks = 0;
    enabled = true;
  }

  void disable(double real_now) {
    if (!enabled) return;
    const double cur = base + static_cast<double>(ticks) * kVirtualStepSeconds;
    const double needed = cur - real_now;  // 虚拟时间领先真实时间多少
    if (needed > offset) offset = needed;
    if (!ever_read || cur > last) last = cur;
    ever_read = true;
    enabled = false;
  }
};

// 复刻外层循环限帧的整数运算：uVar6 = (int)((period - elapsed) * -1000)；负数时 Sleep(~uVar6)。
inline std::uint32_t limiterSleepMilliseconds(double elapsed) {
  const double scaled = (kFramePeriodSeconds - elapsed) * -1000.0;  // DAT_00baad98 = -1000.0
  const std::int32_t truncated = static_cast<std::int32_t>(scaled);  // cvttsd2si
  return truncated < 0 ? static_cast<std::uint32_t>(~truncated) : 0u;
}

}  // namespace isaac_turbo
