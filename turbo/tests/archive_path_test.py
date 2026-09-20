"""Exercise the actual registered-container resolver hook against a native-call stub."""
from pathlib import Path
import subprocess

turbo = Path(__file__).resolve().parents[1]
source = (turbo / 'src/turbo.cpp').read_text(encoding='utf-8')
start = source.index('char* __fastcall hookResolvePath(')
end = source.index('\n// ---------------------------------------------------------------- Hook：shader', start)
out = turbo / 'build/archive-test'
out.mkdir(parents=True, exist_ok=True)
cpp = out / 'archive_path_test.cpp'
exe = out / 'archive_path_test.exe'
cpp.write_text(r'''
#include <windows.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
DWORD fixtureAttributes(const char* path){return std::strstr(path,"afterbirthp.a")?FILE_ATTRIBUTE_NORMAL:INVALID_FILE_ATTRIBUTES;}
#define GetFileAttributesA fixtureAttributes
namespace j460 { constexpr unsigned kArchiveSlotTableRva=0; }
std::uint32_t slots[64]{};
std::uintptr_t gImageBase=reinterpret_cast<std::uintptr_t>(slots);
bool gArchivePathFix=true, gResolverProbe=false, nativeFail=false;
LONG gArchiveResolveRecoveries=0, gArchiveResolveCalls=0;
void* gLastSearchDirectory=nullptr;
unsigned gLastSearchInArchive=0, gLastSearchKey=0;
char gLastSearchCopy[384]{}, seen[1024]{}, nativeOwned[]="native owned result";
int calls=0, err=0, doserr=0;
int getErr(int* p){*p=err;return 0;} int setErr(int x){err=x;return 0;}
int getDos(int* p){*p=doserr;return 0;} int setDos(unsigned long x){doserr=x;return 0;}
auto gGetErrno=&getErr; auto gSetErrno=&setErr;
auto gGetDosErrno=&getDos; auto gSetDosErrno=&setDos;
void logf(const char*,...){err=900;doserr=901;SetLastError(902);}
void logStackScan(const void*,unsigned,const char*){}
char* __fastcall original(void*,void*,const char* path){
 ++calls;std::strcpy(seen,path?path:"(null)");err=17;doserr=18;SetLastError(19);
 return !nativeFail && path && std::strlen(path)>2 && path[1]==':' ? nativeOwned : nullptr;
}
auto gOrigResolvePath=&original;
''' + source[start:end] + r'''
int main(){
 char relative[]="resources/packed/afterbirthp.a", sameText[]="resources/packed/afterbirthp.a";
 char nested[]="resources/secret.a";
 char absolute[]="D:\\container.a", missing[]="missing.png";
 struct Case {const char* name;const char* input;const char* slot;bool enabled,fail,redirect,success;};
 Case cases[]={
  {"pending-slot-31",relative,relative,true,false,true,true},
  {"equal-text-not-identity",sameText,relative,true,false,false,false},
  {"nested-container",nested,nested,true,false,false,false},
  {"ordinary-asset",missing,relative,true,false,false,false},
  {"absolute-no-rewrite",absolute,absolute,true,false,false,true},
  {"native-failure-preserved",relative,relative,true,true,true,false},
  {"disabled",relative,relative,false,false,false,false},
  {"null-input",nullptr,relative,true,false,false,false}};
 int failures=0;
 for(auto& c:cases){
  std::memset(slots,0,sizeof(slots));slots[62]=reinterpret_cast<std::uintptr_t>(c.slot);
  calls=0;gArchivePathFix=c.enabled;nativeFail=c.fail;
  char expected[1024]{};
  if(c.redirect)GetFullPathNameA(c.input,sizeof(expected),expected,nullptr);
  else std::strcpy(expected,c.input?c.input:"(null)");
  char* result=hookResolvePath(nullptr,nullptr,c.input);
  bool ok=calls==1 && std::strcmp(seen,expected)==0 && result==(c.success?nativeOwned:nullptr)
       && err==17 && doserr==18 && GetLastError()==19;
  std::printf("%s %s native_calls=%d\n",ok?"PASS":"FAIL",c.name,calls);failures+=!ok;
 }
 return failures?1:0;
}
''', encoding='utf-8')
zig = turbo.parents[2] / '.tools/zig-x86_64-windows-0.16.0/zig.exe'
subprocess.run([str(zig), 'c++', '-std=c++17', '-O2', '-Wno-nullability-completeness',
                '-target', 'x86-windows-gnu', str(cpp), '-o', str(exe)], check=True)
raise SystemExit(subprocess.run([str(exe)]).returncode)
