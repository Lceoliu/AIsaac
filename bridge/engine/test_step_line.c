// abp_step_line.h by itself (schema_v3 batch 4, the step line's drop): a line without a 7th number is the same step as
// one with a 7th number 0, and without a 6th the same as a 6th 0. gcc -O1 -o t test_step_line.c && ./t (exit 0: holds).
// tests/test_drop_protocol.py compiles and runs it where gcc exists.
#include <stdio.h>
#include <string.h>
#include "abp_step_line.h"

static int same(const char *a, const char *b) {
    long long va[7], vb[7];
    memset(va, 0x55, sizeof va);
    memset(vb, 0x77, sizeof vb);   // (different garbage: every field must be written)
    int ra = abp_step_line_parse(a, va), rb = abp_step_line_parse(b, vb);
    return ra == 1 && rb == 1 && memcmp(va, vb, sizeof va) == 0;
}

int main(void) {
    int bad = 0;
#define CHECK(x) do { if (!(x)) { printf("FAIL %s\n", #x); bad++; } } while (0)
    CHECK(same("S 4 1 2 0 1", "S 4 1 2 0 1 0 0"));
    CHECK(same("S 4 1 2 0 1", "S 4 1 2 0 1 0"));
    CHECK(same("S 4 1 2 0 1 3", "S 4 1 2 0 1 3 0"));
    CHECK(same("S 1 0 0 0 0", "S 1 0 0 0 0 0 0"));
    CHECK(!same("S 4 1 2 0 1", "S 4 1 2 0 1 0 1"));
    long long v[7];
    CHECK(abp_step_line_parse("S 4 1 2 0 1 2 1", v) == 1 && v[0] == 4 && v[1] == 1 && v[2] == 2 && v[3] == 0 &&
          v[4] == 1 && v[5] == 2 && v[6] == 1);
    CHECK(abp_step_line_parse("S 4 1 2 0 1", v) == 1 && v[5] == 0 && v[6] == 0);
    CHECK(abp_step_line_parse("S 4 1 2 0", v) == 3);              // (5 numbers at least)
    CHECK(abp_step_line_parse("S 4 1 2 0 1 0 0 0", v) == 3);      // (7 at most)
    CHECK(abp_step_line_parse("S 4 1 2 0 1 ", v) == 3);
    CHECK(abp_step_line_parse("S 4 1 2 0 1  0", v) == 3);
    CHECK(abp_step_line_parse("S 4 1 2 0 x", v) == 3);
    CHECK(abp_step_line_parse("S 1234567890 1 2 0 1", v) == 3);   // (9 digits at most)
    printf(bad ? "step_line: %d failed\n" : "step_line: ok\n", bad);
    return bad != 0;
}
