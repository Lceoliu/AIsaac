// isaac_turbo_inject.exe <pid> <absolute-dll-path>
// isaac_turbo_inject.exe --launch <absolute-dll-path> <exe> [args...]
//
// 32 位注入器：CreateRemoteThread + LoadLibraryW。与 netfix/src/tools/Injector.cpp 同源，
// 必须编译为 x86，kernel32!LoadLibraryW 的地址才与 32 位目标进程一致。
//
// --launch：以 CREATE_SUSPENDED 创建进程，先注入并等待控制块状态离开 loading，再恢复主线程，
// 使钩子在游戏的第一个归档挂载 / 首帧渲染之前就绪（启动期 B 类崩溃只能这样取证）。
// 进程的工作目录取 exe 所在目录；环境变量原样继承（ISAAC_RL_* / ISAAC_TURBO_* 由调用方设置）。
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "turbo_shared.hpp"

namespace {

std::wstring absolutePath(const wchar_t* input) {
  wchar_t buffer[32768]{};
  const auto length = GetFullPathNameW(input, 32768, buffer, nullptr);
  if (length == 0 || length >= 32768) return {};
  return std::wstring(buffer, length);
}

// 返回 0 = 成功；否则为退出码（与旧行为一致）。
int injectInto(HANDLE process, DWORD pid, const std::wstring& dllPath, DWORD* moduleOut) {
  BOOL wow64 = FALSE;
  if (IsWow64Process(process, &wow64) && !wow64) {
    std::wcerr << L"FAIL target_not_32bit\n";
    return 11;
  }
  const auto bytes = (dllPath.size() + 1) * sizeof(wchar_t);
  const auto remotePath = VirtualAllocEx(process, nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (!remotePath) {
    std::wcerr << L"FAIL virtual_alloc error=" << GetLastError() << L"\n";
    return 6;
  }
  SIZE_T written = 0;
  if (!WriteProcessMemory(process, remotePath, dllPath.c_str(), bytes, &written) || written != bytes) {
    std::wcerr << L"FAIL write_process_memory error=" << GetLastError() << L"\n";
    VirtualFreeEx(process, remotePath, 0, MEM_RELEASE);
    return 7;
  }
  const auto kernel = GetModuleHandleW(L"kernel32.dll");
  const auto loadLibrary = kernel ? GetProcAddress(kernel, "LoadLibraryW") : nullptr;
  if (!loadLibrary) {
    std::wcerr << L"FAIL loadlibrary_address\n";
    VirtualFreeEx(process, remotePath, 0, MEM_RELEASE);
    return 8;
  }
  const auto thread = CreateRemoteThread(process, nullptr, 0,
                                         reinterpret_cast<LPTHREAD_START_ROUTINE>(loadLibrary), remotePath,
                                         0, nullptr);
  if (!thread) {
    std::wcerr << L"FAIL create_remote_thread error=" << GetLastError() << L"\n";
    VirtualFreeEx(process, remotePath, 0, MEM_RELEASE);
    return 9;
  }
  const auto wait = WaitForSingleObject(thread, 15000);
  DWORD module = 0;
  const auto gotExit = GetExitCodeThread(thread, &module);
  CloseHandle(thread);
  VirtualFreeEx(process, remotePath, 0, MEM_RELEASE);
  if (wait != WAIT_OBJECT_0 || !gotExit || module == 0) {
    std::wcerr << L"FAIL remote_load wait=" << wait << L" module=0x" << std::hex << module << L" error="
               << std::dec << GetLastError() << L" pid=" << pid << L"\n";
    return 10;
  }
  if (moduleOut) *moduleOut = module;
  return 0;
}

// 等控制块状态离开 loading（DLL 的初始化线程跑完身份门与 Hook 安装）。返回最终状态；超时返回 0xFFFFFFFF。
std::uint32_t waitControlBlock(DWORD pid, DWORD timeoutMs) {
  char name[64];
  isaac_turbo::formatMappingName(name, sizeof(name), pid);
  const auto deadline = GetTickCount64() + timeoutMs;
  std::uint32_t status = 0xFFFFFFFFu;
  while (GetTickCount64() < deadline) {
    const HANDLE mapping = OpenFileMappingA(FILE_MAP_READ, FALSE, name);
    if (mapping) {
      const auto* view = static_cast<const isaac_turbo::ControlBlock*>(
          MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, isaac_turbo::kControlBlockBytes));
      if (view) {
        if (view->magic == isaac_turbo::kMagic && view->status != isaac_turbo::kStatusLoading) status = view->status;
        UnmapViewOfFile(view);
      }
      CloseHandle(mapping);
      if (status != 0xFFFFFFFFu) return status;
    }
    Sleep(20);
  }
  return status;
}

std::wstring quoteArg(const std::wstring& arg) {
  if (arg.empty()) return L"\"\"";
  if (arg.find_first_of(L" \t\"") == std::wstring::npos) return arg;
  std::wstring out = L"\"";
  for (wchar_t c : arg) {
    if (c == L'"') out += L"\\\"";
    else out += c;
  }
  out += L"\"";
  return out;
}

int launchSuspended(int argc, wchar_t** argv) {
  // argv: --launch <dll> <exe> [args...]
  if (argc < 4) {
    std::wcerr << L"USAGE isaac_turbo_inject.exe --launch <absolute-dll-path> <exe> [args...]\n";
    return 2;
  }
  const auto dllPath = absolutePath(argv[2]);
  if (dllPath.empty() || GetFileAttributesW(dllPath.c_str()) == INVALID_FILE_ATTRIBUTES) {
    std::wcerr << L"FAIL dll_not_found path=\"" << dllPath << L"\"\n";
    return 4;
  }
  const auto exePath = absolutePath(argv[3]);
  if (exePath.empty() || GetFileAttributesW(exePath.c_str()) == INVALID_FILE_ATTRIBUTES) {
    std::wcerr << L"FAIL exe_not_found path=\"" << exePath << L"\"\n";
    return 13;
  }
  std::wstring commandLine = quoteArg(exePath);
  for (int i = 4; i < argc; ++i) {
    commandLine += L" ";
    commandLine += quoteArg(argv[i]);
  }
  const auto slash = exePath.find_last_of(L"\\/");
  const std::wstring workDir = slash == std::wstring::npos ? L"." : exePath.substr(0, slash);
  STARTUPINFOW si{};
  si.cb = sizeof(si);
  PROCESS_INFORMATION pi{};
  std::wstring mutableCommandLine = commandLine;
  if (!CreateProcessW(exePath.c_str(), &mutableCommandLine[0], nullptr, nullptr, FALSE, CREATE_SUSPENDED, nullptr,
                      workDir.c_str(), &si, &pi)) {
    std::wcerr << L"FAIL create_process error=" << GetLastError() << L"\n";
    return 14;
  }
  DWORD module = 0;
  const int rc = injectInto(pi.hProcess, pi.dwProcessId, dllPath, &module);
  if (rc != 0) {
    // 注入失败就不要留下一个挂起的僵尸进程。
    TerminateProcess(pi.hProcess, 0xDEAD);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return rc;
  }
  const std::uint32_t status = waitControlBlock(pi.dwProcessId, 20000);
  const DWORD resumed = ResumeThread(pi.hThread);
  std::wcout << L"PASS launched pid=" << pi.dwProcessId << L" module=0x" << std::hex << module << std::dec
             << L" status=" << status << L" resumed=" << resumed << L" dll=\"" << dllPath << L"\"\n";
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  // 0 = 钩子已激活；12 = 进程在跑但钩子未激活（身份门/安装失败/超时），由调用方决定去留。
  return status == isaac_turbo::kStatusActive ? 0 : 12;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  if (argc >= 2 && std::wcscmp(argv[1], L"--launch") == 0) return launchSuspended(argc, argv);
  if (argc != 3) {
    std::wcerr << L"USAGE isaac_turbo_inject.exe <pid> <absolute-dll-path>\n"
                  L"      isaac_turbo_inject.exe --launch <absolute-dll-path> <exe> [args...]\n";
    return 2;
  }
  wchar_t* end = nullptr;
  const auto pidValue = std::wcstoul(argv[1], &end, 10);
  if (!end || *end != L'\0' || pidValue == 0 || pidValue > 0xFFFFFFFFul) {
    std::wcerr << L"FAIL invalid_pid\n";
    return 3;
  }
  const auto dllPath = absolutePath(argv[2]);
  if (dllPath.empty() || GetFileAttributesW(dllPath.c_str()) == INVALID_FILE_ATTRIBUTES) {
    std::wcerr << L"FAIL dll_not_found path=\"" << dllPath << L"\"\n";
    return 4;
  }
  const auto process = OpenProcess(PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION |
                                       PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ,
                                   FALSE, static_cast<DWORD>(pidValue));
  if (!process) {
    std::wcerr << L"FAIL open_process error=" << GetLastError() << L"\n";
    return 5;
  }
  DWORD module = 0;
  const int rc = injectInto(process, static_cast<DWORD>(pidValue), dllPath, &module);
  CloseHandle(process);
  if (rc != 0) return rc;
  std::wcout << L"PASS injected pid=" << pidValue << L" module=0x" << std::hex << module << L" dll=\""
             << dllPath << L"\"\n";
  return 0;
}
