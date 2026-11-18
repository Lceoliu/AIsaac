# B58e: the main thread's Holding GIL intervals of an nsys sqlite (python-gil): count, durations, gaps, the ranges they fall in,
# the bursts inside c.wait. usage: python3 b58e_gilq.py <report.sqlite>
import sqlite3, sys, collections
c = sqlite3.connect(sys.argv[1])
S = {i: v for i, v in c.execute('select id, value from StringIds')}
txt = "coalesce(text, (select value from StringIds where id = textId))"
main = c.execute("select globalTid from NVTX_EVENTS where text = 'actor.collect' group by globalTid order by count(*) desc limit 1").fetchone()[0]
hold = [k for k, v in S.items() if v == 'Holding GIL'][0]
ev = c.execute("select start, end from NVTX_EVENTS where globalTid=? and textId=? and end is not null order by start", (main, hold)).fetchall()
print('holdings', len(ev))
gaps = [ev[i+1][0]-ev[i][1] for i in range(len(ev)-1)]
durs = [e-s for s, e in ev]
import statistics as st
print('dur us p50 %.1f p90 %.1f' % (st.median(durs)/1e3, sorted(durs)[int(.9*len(durs))]/1e3))
print('gap ms p10 %.2f p50 %.2f p90 %.2f' % tuple(sorted(gaps)[int(q*len(gaps))]/1e6 for q in (.1, .5, .9)))
rng = c.execute(f"select start, end, {txt} from NVTX_EVENTS where globalTid=? and end is not null and coalesce(textId,-9) != ? and {txt} in ('c.wait','actor.collect','h.wait_learner','h.job','h.gc')", (main, hold)).fetchall()
cnt = collections.Counter()
for s, e in ev:
    names = [n for a, b, n in rng if a <= s and e <= b]
    cnt[','.join(sorted(set(names))) or '-'] += 1
print(cnt.most_common(8))
# which other threads hold the GIL right before each main holding (handoff partner)
cw = [(a, b) for a, b, n in rng if n == 'c.wait']
inside = [(s, e) for s, e in ev if any(a <= s and e <= b for a, b in cw)]
g2 = sorted(inside[i+1][0]-inside[i][1] for i in range(len(inside)-1))
print('inside c.wait', len(inside), 'gap ms p10 %.2f p25 %.2f p50 %.2f p75 %.2f p90 %.2f' % tuple(g2[int(q*len(g2))]/1e6 for q in (.1, .25, .5, .75, .9)))
d2 = sorted(e-s for s, e in inside)
print('dur us p50 %.1f p90 %.1f max %.1f' % (d2[len(d2)//2]/1e3, d2[int(.9*len(d2))]/1e3, d2[-1]/1e3))
bursts, cur = [], [inside[0]]
for a, b in zip(inside, inside[1:]):
    if b[0]-a[1] < 1e6: cur.append(b)
    else: bursts.append(cur); cur = [b]
bursts.append(cur)
sz = sorted((len(x), (x[-1][1]-x[0][0])/1e6, sum(e-s for s, e in x)/1e6) for x in bursts)
print('bursts', len(bursts), 'largest (holdings, span ms, held ms):', sz[-6:])
print('singletons', sum(1 for x in bursts if len(x) == 1))
