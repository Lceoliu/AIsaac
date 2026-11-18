// The step line's numbers (abp_turbo.c step_read; the bridge's Lua parse the same): "S <repeat> <move> <shoot> <bomb>
// <item>[ <pill>[ <drop>]]" after the "S", NUL-terminated, no newline. v[0..6] = repeat, move, shoot, bomb, item, pill,
// drop; a missing 6th / 7th number is 0 (2026-10-06 pill, 2026-11-17 drop: visible-v3 F2 -- schema_v3 batch 4: a line
// without the 7th number is the same step as one with 0). Numbers of up to 9 digits. 1 parsed, 3 malformed.
#ifndef ABP_STEP_LINE_H
#define ABP_STEP_LINE_H
static int abp_step_line_parse(const char *line, long long v[7]) {
    const char *c = line + 1;
    v[5] = 0;
    v[6] = 0;
    for (int i = 0; i < 7; i++) {
        if (i >= 5 && *c == 0) break;
        if (*c != ' ') return 3;
        c++;
        if (*c < '0' || *c > '9') return 3;
        long long x = 0;
        int digits = 0;
        while (*c >= '0' && *c <= '9') {
            x = x * 10 + (*c - '0');
            c++;
            if (++digits > 9) return 3;
        }
        v[i] = x;
    }
    return *c == 0 ? 1 : 3;
}
#endif
