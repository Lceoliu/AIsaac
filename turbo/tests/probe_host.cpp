// 在非游戏进程里加载 isaac_turbo.dll：必须被身份门拦住（不装 Hook），但控制块与日志要正常建立。
// 用法: probe_host.exe <isaac_turbo.dll 绝对路径> [hold-ms]
//   hold-ms > 0 时通过后继续存活指定毫秒，供 Python 端（isaac_bridge.turbo）附着到命名共享内存做端到端测试。
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../src/turbo_shared.hpp"

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::printf("FAIL usage\n");
    return 2;
  }
  const HMODULE dll = LoadLibraryA(argv[1]);
  if (!dll) {
    std::printf("FAIL load_library error=%lu\n", GetLastError());
    return 3;
  }
  const auto version = reinterpret_cast<std::uint32_t (*)()>(GetProcAddress(dll, "IsaacTurboVersion"));
  const auto status = reinterpret_cast<const char* (*)()>(GetProcAddress(dll, "IsaacTurboStatus"));
  const auto init = reinterpret_cast<int (*)()>(GetProcAddress(dll, "IsaacTurboInit"));
  const auto block = reinterpret_cast<void* (*)()>(GetProcAddress(dll, "IsaacTurboControlBlock"));
  if (!version || !status || !init || !block) {
    std::printf("FAIL exports\n");
    return 4;
  }
  const int result = init();
  const char* text = status();
  if (result != isaac_turbo::kStatusIdentityFailed || !text || std::strncmp(text, "ERROR:identity", 14) != 0) {
    std::printf("FAIL non_game_host_not_blocked result=%d status=\"%s\"\n", result, text ? text : "null");
    return 5;
  }
  const auto* cb = static_cast<const isaac_turbo::ControlBlock*>(block());
  if (!cb || cb->magic != isaac_turbo::kMagic || cb->version != isaac_turbo::kLayoutVersion ||
      cb->status != isaac_turbo::kStatusIdentityFailed || cb->hooks_mask != 0 ||
      cb->pid != GetCurrentProcessId()) {
    std::printf("FAIL control_block\n");
    return 6;
  }
  char name[64];
  isaac_turbo::formatMappingName(name, sizeof(name), GetCurrentProcessId());
  const HANDLE mapping = OpenFileMappingA(FILE_MAP_READ, FALSE, name);
  if (!mapping) {
    std::printf("FAIL named_mapping_missing name=%s error=%lu\n", name, GetLastError());
    return 7;
  }
  const auto* view = static_cast<const isaac_turbo::ControlBlock*>(
      MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, isaac_turbo::kControlBlockBytes));
  if (!view || view->magic != isaac_turbo::kMagic || view->status != isaac_turbo::kStatusIdentityFailed) {
    std::printf("FAIL named_mapping_content\n");
    return 8;
  }
  std::printf("PASS probe_host pid=%lu version=%u status=\"%s\" mapping=%s log=%s\n",
              static_cast<unsigned long>(GetCurrentProcessId()), version(), text, name, cb->log_path);
  std::fflush(stdout);
  const long holdMs = argc == 3 ? std::strtol(argv[2], nullptr, 10) : 0;
  if (holdMs > 0) {
    // 让控制端有时间写 flags 并读回；退出前把控制端写入的 flags 回显，证明共享是双向的。
    Sleep(static_cast<DWORD>(holdMs));
    std::printf("HOLD_END flags=0x%X render_every=%u render_request=%u\n", view->flags, view->render_every,
                view->render_request);
  }
  UnmapViewOfFile(view);
  CloseHandle(mapping);
  return 0;
}
