// 虚拟时钟与外层循环限帧算术的离线验证（不需要游戏）。
// 复刻 RVA 0x00531050 每次迭代的三次时钟读取与 Sleep/自旋判定，证明：
//   * 开启虚拟时钟后 Sleep 毫秒恒为 0，自旋第一次比较即退出；
//   * 读时钟不推进时间（只有 tick 推进）；
//   * 开/关切换与"被 Lua 阻塞数秒后恢复"都不会让时钟回跳。
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../src/virtual_clock.hpp"

using namespace isaac_turbo;

static int gFailures = 0;
#define CHECK(cond, ...)                                     \
  do {                                                       \
    if (!(cond)) {                                           \
      ++gFailures;                                           \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__);       \
      std::printf(__VA_ARGS__);                              \
      std::printf("\n");                                     \
    }                                                        \
  } while (0)

// 模拟一次外层循环迭代（turbo 已开启），返回 frame_start。
static double simulateIteration(VirtualClock& clock, double& real, double prevTop, bool first) {
  real += 40e-6;  // Steam 回调 + 平台层
  const double top = clock.read(real);  // frame_start，同时算 fps
  CHECK(top >= prevTop, "top went backwards %.9f < %.9f", top, prevTop);
  if (!first) {
    const double delta = top - prevTop;
    CHECK(delta >= kFramePeriodSeconds, "frame delta %.9g < period", delta);
    // 连续 30 帧 period/delta > 1.1 会让游戏打开软件限帧；虚拟时钟下比值应恒 <= 1
    CHECK(kFramePeriodSeconds / delta <= 1.0, "ratio %.9g > 1", kFramePeriodSeconds / delta);
  }
  real += 5e-6;
  (void)clock.read(real);  // FUN_00a714e0 输入设备 delta
  real += 250e-6;          // Manager::Update（含 Game::Update）
  clock.tick();
  real += 3e-6;
  const double elapsed = clock.read(real) - top;  // 限帧读取
  CHECK(limiterSleepMilliseconds(elapsed) == 0, "sleep ms=%u elapsed=%.9g", limiterSleepMilliseconds(elapsed),
        elapsed);
  int spins = 0;
  while (kFramePeriodSeconds > clock.read(real) - top) {
    ++spins;
    if (spins > 3) break;
  }
  CHECK(spins == 0, "spin loop did not exit immediately (spins=%d)", spins);
  return top;
}

static void testLimiterArithmetic() {
  CHECK(limiterSleepMilliseconds(kFramePeriodSeconds) == 0, "exact period must not sleep");
  CHECK(limiterSleepMilliseconds(kFramePeriodSeconds * 2) == 0, "late frame must not sleep");
  CHECK(limiterSleepMilliseconds(kFramePeriodSeconds / 2) == 7, "half frame: sleep remaining-1 ms, got %u",
        limiterSleepMilliseconds(kFramePeriodSeconds / 2));
  CHECK(limiterSleepMilliseconds(0.0) == 15, "zero elapsed: got %u", limiterSleepMilliseconds(0.0));
  CHECK(limiterSleepMilliseconds(-1000.0) == 1000015, "clock going backwards would sleep ~16 min (%u ms)",
        limiterSleepMilliseconds(-1000.0));
}

static void testTurboLoop(double realBase, int iterations) {
  VirtualClock clock;
  double real = realBase;
  double prev = clock.read(real);
  clock.enable(real);
  bool first = true;
  for (int i = 0; i < iterations; ++i) {
    prev = simulateIteration(clock, real, prev, first);
    first = false;
  }
  CHECK(clock.ticks == static_cast<std::uint64_t>(iterations), "ticks=%llu", (unsigned long long)clock.ticks);
  // 读时钟不推进：连续读 1000 次值不变
  const double v = clock.read(real);
  for (int i = 0; i < 1000; ++i) CHECK(clock.read(real) == v, "read advanced the clock");
  // 虚拟时间 = base + ticks*step，与真实时间无关
  const double expected = clock.base + static_cast<double>(iterations) * kVirtualStepSeconds;
  CHECK(std::fabs(v - expected) < 1e-9, "value %.9f expected %.9f", v, expected);
}

static void testToggleContinuity() {
  VirtualClock clock;
  double real = 500.0;
  double v0 = clock.read(real);
  CHECK(v0 == 500.0, "passthrough should equal real time before any offset");
  clock.enable(real);
  for (int i = 0; i < 6000; ++i) clock.tick();  // 100 s 虚拟时间，只花了很少真实时间
  real += 0.5;
  const double vOn = clock.read(real);
  CHECK(vOn > 600.0, "virtual clock should be ~100 s ahead, got %.3f", vOn);
  clock.disable(real);
  const double vOff = clock.read(real);
  CHECK(vOff >= vOn, "disable must not go backwards: %.6f < %.6f", vOff, vOn);
  real += 0.02;
  const double vOff2 = clock.read(real);
  CHECK(std::fabs((vOff2 - vOff) - 0.02) < 1e-9, "passthrough must advance at real rate, got %.9f", vOff2 - vOff);
  CHECK(limiterSleepMilliseconds(vOff2 - vOff) == 0, "no sleep after disable");

  // 被 Lua 阻塞 10 s（虚拟时钟落后真实时间）后关闭：允许向前跳，不允许回跳
  VirtualClock c2;
  real = 1000.0;
  (void)c2.read(real);
  c2.enable(real);
  c2.tick();
  real += 10.0;
  const double before = c2.read(real);
  c2.disable(real);
  const double after = c2.read(real);
  CHECK(after >= before, "forward jump allowed, backward not: %.6f -> %.6f", before, after);
  CHECK(after >= 1010.0 - 1e-9, "should follow real time after disable, got %.6f", after);

  // 开启时刻若真实时间已超过上次返回值：base 取真实时间，第一帧不睡眠不自旋
  VirtualClock c3;
  real = 2000.0;
  const double top = c3.read(real);  // frame_start 于阻塞前
  real += 3.0;                       // 在 Game::Update 内被阻塞 3 s，其间控制端打开标志
  c3.tick();
  c3.enable(real);                   // 限帧读取时才发现标志
  const double elapsed = c3.read(real) - top;
  CHECK(elapsed >= kFramePeriodSeconds, "first elapsed after enable %.6f", elapsed);
  CHECK(limiterSleepMilliseconds(elapsed) == 0, "no sleep on the enabling frame");
}

int main() {
  testLimiterArithmetic();
  testTurboLoop(0.0, 100000);
  testTurboLoop(1234.5678, 100000);
  testTurboLoop(3.0e7, 100000);  // 进程跑了一年也不能因浮点误差卡进自旋
  testToggleContinuity();
  if (gFailures) {
    std::printf("FAIL clock_test failures=%d\n", gFailures);
    return 1;
  }
  std::printf("PASS clock_test step=%.12f period=%.12f\n", kVirtualStepSeconds, kFramePeriodSeconds);
  return 0;
}
