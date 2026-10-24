"""Compile the actual access hook against deterministic CRT/diagnostic stubs.

The game identity gate is not bypassed: no DLL is injected and no game is needed.
Usage: python access_probe_test.py [source.cpp] [output-directory]
"""
from pathlib import Path
import subprocess
import sys

turbo = Path(__file__).resolve().parents[1]
source = Path(sys.argv[1]) if len(sys.argv) > 1 else turbo / 'src/turbo.cpp'
out = Path(sys.argv[2]) if len(sys.argv) > 2 else turbo / 'build/access-test'
out.mkdir(parents=True, exist_ok=True)
text = source.read_text(encoding='utf-8')
functions = []
for signature in ('bool looksLikeArchive(', 'int __cdecl hookCrtAccess('):
    start = text.index(signature)
    end = text.index('{', start)
    depth = 1
    while depth:
        end += 1
        depth += (text[end] == '{') - (text[end] == '}')
    functions.append(text[start:end + 1])
prelude = r'''
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
struct Block { unsigned flags=0; unsigned long long access_calls=0, access_failures=0,
 access_retries=0, access_retry_ok=0; } block;
Block* gBlock=&block;
constexpr unsigned kFlagProbeDump=1, kFlagFileRetry=2;
volatile LONG gAccessTotal=0, gAccessFailed=0;
int error_no=0, dos_error=0, calls=0; bool succeed_on_retry=false;
int orig(const char*,int) { ++calls; bool ok=succeed_on_retry && calls>1;
 error_no=ok?17:2; dos_error=ok?18:3; SetLastError(ok?19:4); return ok?0:-1; }
int get_e(int* p){*p=error_no;return 0;} int get_d(int* p){*p=dos_error;return 0;}
int set_e(int e){error_no=e;return 0;} int set_d(unsigned long e){dos_error=e;return 0;}
auto gOrigAccess=&orig; auto gGetErrno=&get_e; auto gGetDosErrno=&get_d;
auto gSetErrno=&set_e; auto gSetDosErrno=&set_d;
void pollute(){error_no=91;dos_error=92;SetLastError(93);}
void logf(const char*,...){pollute();}
void diagnoseAccessFailure(const char*,int,int,int,DWORD){pollute();}
void logStackScan(const void*,unsigned,const char*){pollute();}
void writeProbeDump(const char*){pollute();}
bool shouldLogOccurrence(unsigned long long n){return n<=20 || n%100==0;}
'''
main = r'''
int main(){
 int failed=0;
 for(int i=0;i<3;++i){
   block={};block.flags=i==2?kFlagFileRetry:0;
   calls=0;gAccessTotal=0;gAccessFailed=0;succeed_on_retry=i==2;
   const char* path=i==1?"resources/optional.a":"D:/fixture.a";
   int result=hookCrtAccess(path,0);DWORD last=GetLastError();
   bool ok=result==(i==2?0:-1) && error_no==(i==2?17:2)
       && dos_error==(i==2?18:3) && last==DWORD(i==2?19:4) && calls==(i==2?2:1);
   printf("%s case=%d result=%d errno=%d doserrno=%d lastError=%lu calls=%d total=%ld\n",
       ok?"PASS":"FAIL",i,result,error_no,dos_error,last,calls,gAccessTotal);
   failed+=!ok;
 }
 return failed?1:0;
}
'''
cpp = out / 'access_probe_test.cpp'
exe = out / 'access_probe_test.exe'
cpp.write_text(prelude + '\n'.join(functions) + main, encoding='utf-8')
zig = turbo.parents[2] / '.tools/zig-x86_64-windows-0.16.0/zig.exe'
subprocess.run([str(zig), 'c++', '-std=c++17', '-O2', '-Wno-nullability-completeness', '-target', 'x86-windows-gnu',
                str(cpp), '-o', str(exe)], check=True)
raise SystemExit(subprocess.run([str(exe)]).returncode)
