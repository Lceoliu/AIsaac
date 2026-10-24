"""Compile and exercise the actual loader hook; no game process or injection."""
from pathlib import Path
import subprocess

turbo = Path(__file__).resolve().parents[1]
text = (turbo / 'src/turbo.cpp').read_text(encoding='utf-8')
start = text.index('LONG NTAPI hookLdrLoadDll(')
end = text.index('\nbool installHooks()', start)
out = turbo / 'build/capture-test'
out.mkdir(parents=True, exist_ok=True)
cpp = out / 'capture_overlay_test.cpp'
exe = out / 'capture_overlay_test.exe'
cpp.write_text(r'''
#include <windows.h>
#include <winternl.h>
#include <cwchar>
#include <cstdio>
int forwarded=0;
LONG NTAPI original(PWSTR,ULONG*,UNICODE_STRING*,HMODULE* module){++forwarded;*module=(HMODULE)123;return 17;}
auto gOrigLdrLoadDll=&original;
void logf(const char*,...){}
''' + text[start:end] + r'''
int main(){
 struct Case {const wchar_t* name;bool block;};
 Case cases[]={{L"NvCamera32.dll",true},{L"C:\\Driver\\NVCAMERA32.DLL",true},
   {L"nvspcap.dll",true},{L"nvwgf2um.dll",false},{L"gameoverlayrenderer.dll",false},
   {L"not-nvspcap.dll",false},{L"nvspcap.dll.extra",false}};
 int failures=0;
 for(const auto& c:cases){
   UNICODE_STRING name{};name.Buffer=const_cast<PWSTR>(c.name);name.Length=std::wcslen(c.name)*2;
   name.MaximumLength=name.Length; // no trailing NUL is required by the loader contract
   forwarded=0;HMODULE module=(HMODULE)1;
   LONG r=hookLdrLoadDll(nullptr,nullptr,&name,&module);
   bool ok=c.block?(r==(LONG)0xC0000135u && module==nullptr && forwarded==0)
                  :(r==17 && module==(HMODULE)123 && forwarded==1);
   std::printf("%s block=%d name=%ls\n",ok?"PASS":"FAIL",c.block,c.name);failures+=!ok;
 }
 return failures?1:0;
}
''', encoding='utf-8')
zig = turbo.parents[2] / '.tools/zig-x86_64-windows-0.16.0/zig.exe'
subprocess.run([str(zig), 'c++', '-std=c++17', '-O2', '-Wno-nullability-completeness',
                '-target', 'x86-windows-gnu', str(cpp), '-o', str(exe)], check=True)
raise SystemExit(subprocess.run([str(exe)]).returncode)
