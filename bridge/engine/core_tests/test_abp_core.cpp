// B58c: the torch-free core (abp_core.h) under -fsanitize=thread with random interleavings. No CUDA, no torch.
//
//   g++ -std=c++17 -O1 -g -fsanitize=thread -pthread test_abp_core.cpp -o test_abp_core && ./test_abp_core 20000 <seed>
//
// Part 1, the page store: a writer (the driver) allocates and writes decision rows stamped with its record sequence and
// publishes them through a queue; a consumer takes them and holds each for a while (the logs / snapshots), acking the
// last stamp it took; readers pin held rows and check every byte (the learner reading its job's pages); a collector
// scans the used flags (under the lock) and frees what is neither held nor pinned, limited to rows stamped before the
// consumer's ack (the B59 rule: rows whose records wait in the queue are not visible to the scan). A freed-and-reused
// row read by a reader shows up as a wrong byte and, without the stamp rule, as a TSAN race.
// Part 2, the native consumer: one thread runs consume() over random records into a rollout slot (rows, length / open
// published atomically, the frame arena under its mutex) while another reads length / open and the arena.
// Exit code 0 when no check failed (TSAN reports races on stderr and makes the exit code non-zero).
#include "../abp_core.h"

#include <chrono>
#include <cstdio>
#include <deque>
#include <map>
#include <random>
#include <set>
#include <thread>

using namespace abpc;

static uint8_t pat(i64 seq, int b) { return (uint8_t)((seq*131+b*7+(seq >> 8)) & 0xff); }

struct Item { i64 id, seq, stamp; };

int part1(int iters, unsigned seed, bool use_stamp) {
  const i64 KV = 32;
  PageCore pc(16, KV, 4, 4, 2, 2, 512, 64);
  std::mutex qm; std::deque<Item> q; std::atomic<bool> done{false};
  std::mutex hm; std::map<i64, Item> held; std::multiset<i64> pinned; i64 consumed = 0;
  std::atomic<i64> errors{0}, reads{0}, frees{0}, scans{0};
  std::thread writer([&] {
    std::mt19937 r(seed);
    uint8_t frame[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (i64 s = 1; s <= iters; s++) {
      int n = 1+r()%6;
      pc.set_stamp(s);
      std::vector<uint8_t> kv(n*KV); std::vector<float> z(n*4, (float)s);
      std::vector<i64> steps(n, s), dec(n), kvs(n), pend(n*2, 0), nb(n, 8);
      std::vector<uint8_t> first(n, 0); std::vector<int32_t> sch(n, 1); std::vector<const uint8_t *> fr(n, frame);
      for (int j = 0; j < n; j++) for (int b = 0; b < KV; b++) kv[j*KV+b] = pat(s*10+j, b);
      pc.write_decisions_raw(n, kv.data(), KV, z.data(), 4, steps.data(), s, first.data(), sch.data(), fr.data(), nb.data(),
                             pend.data(), dec.data(), kvs.data());
      { std::lock_guard<std::mutex> l(qm); for (int j = 0; j < n; j++) q.push_back({kvs[j], s*10+j, s}); }
      if (r()%64 == 0) std::this_thread::yield();
    }
    done = true;
  });
  std::thread consumer([&] {
    std::mt19937 r(seed+1);
    std::deque<i64> order;
    while (true) {
      std::deque<Item> got;
      { std::lock_guard<std::mutex> l(qm); got.swap(q); }
      if (got.empty()) {
        if (done) { std::lock_guard<std::mutex> l(qm); if (q.empty()) break; }
        std::this_thread::yield();
        continue;
      }
      if (r()%4 == 0) std::this_thread::sleep_for(std::chrono::microseconds(r()%200));   // (the consumer lags)
      std::lock_guard<std::mutex> l(hm);
      for (auto &it : got) { held[it.id] = it; order.push_back(it.id); consumed = it.stamp; }
      while ((i64)order.size() > 150+(i64)(r()%100)) { held.erase(order.front()); order.pop_front(); }
    }
  });
  auto reader = [&](unsigned s) {
    std::mt19937 r(s);
    while (!done || reads < 1000) {
      std::vector<Item> mine;
      {
        std::lock_guard<std::mutex> l(hm);
        if (held.empty()) continue;
        for (int k = 0; k < 6; k++) {
          auto it = held.begin(); std::advance(it, r()%held.size());
          mine.push_back(it->second); pinned.insert(it->first);
        }
      }
      for (auto &it : mine) {
        const uint8_t *row = pc.kv_page(it.id/pc.page)+(it.id % pc.page)*KV;
        for (int b = 0; b < KV; b++) if (row[b] != pat(it.seq, b)) { errors++; break; }
        reads++;
      }
      { std::lock_guard<std::mutex> l(hm); for (auto &it : mine) pinned.erase(pinned.find(it.id)); }
      if (done && reads > 1000) break;
    }
  };
  std::thread r1(reader, seed+2), r2(reader, seed+3);
  std::thread collector([&] {
    std::mt19937 r(seed+4);
    std::vector<uint8_t> du(pc.cap_d), bu(pc.cap_b);
    while (!done) {
      std::set<i64> live; i64 limit;
      {
        std::lock_guard<std::mutex> l(hm);
        for (auto &kv : held) live.insert(kv.first);
        for (i64 id : pinned) live.insert(id);
        limit = consumed+1;
      }
      pc.used_snapshot(du.data(), bu.data());
      std::vector<i64> dead;
      for (i64 id = 0; id < pc.cap_d; id++) if (du[id] && !live.count(id)) dead.push_back(id);
      pc.free_rows(dead.data(), (i64)dead.size(), nullptr, 0, nullptr, 0, use_stamp ? limit : 0);
      frees += (i64)dead.size(); scans++;
      std::this_thread::sleep_for(std::chrono::microseconds(50+r()%300));
    }
  });
  writer.join(); consumer.join(); r1.join(); r2.join(); collector.join();
  std::printf("part 1 (stamp rule %s): %d records, %lld row reads, %lld scans, %lld freed, %lld kept by the stamp, %lld reallocs, "
              "%lld wrong bytes\n", use_stamp ? "on" : "off", iters, (long long)reads.load(), (long long)scans.load(),
              (long long)pc.freed_kv, (long long)pc.stamp_kept, (long long)pc.reallocs, (long long)errors.load());
  return errors.load() == 0 && pc.reallocs == 0 ? 0 : 1;
}

int part2(int iters, unsigned seed) {
  const i64 W = 4, CAP = 1 << 16, STRIDE = 256, NH = 2;
  ConsumerCfg c;
  c.on = true; c.o_first = 0; c.o_done = 1; c.o_damage = 4; c.o_hurt = 8; c.o_events = 12; c.o_stage = 13; c.o_pforce = 14;
  c.o_player = 16; c.row_frame_at = 152; c.w_damage = 1; c.w_hurt = .75; c.w_time = .002; c.w_death = 2; c.w_timeout = 1;
  c.w_room = 1; c.w_explore = .5; c.w_boss = 2; c.w_exit = 3; c.w_heal = .5; c.w_resource = .25; c.end_bonus = 3;
  c.floor = c.run = c.items = c.gains_on = c.plr_on = c.pick_on = true;
  std::vector<double> ret(W), hurt(W), dmg(W), gains(W*2), healed(W), gathered(W), v0(W), gret(W);
  std::vector<i64> rooms(W), boss(W), extra(W*6), steps(W), pick(W);
  c.ret = ret.data(); c.ep_hurt = hurt.data(); c.ep_damage = dmg.data(); c.gains = gains.data(); c.healed = healed.data();
  c.gathered = gathered.data(); c.v0 = v0.data(); c.gret = gret.data(); c.rooms = rooms.data(); c.boss = boss.data();
  c.extra = extra.data(); c.steps = steps.data(); c.pick_steps = pick.data();
  RollSlot R;
  R.cap = CAP; R.rowc_size = 160; R.off_visible_off = 152; R.priv_size = 0;
  std::vector<uint8_t> rows(W*CAP*160), term(W*CAP), cdone(W*CAP), cev(W*CAP), boot(W*CAP);
  std::vector<float> rew(W*CAP), chu(W*CAP), logp(W*CAP), val(W*CAP), vg(W*CAP);
  std::vector<i64> open(W, -1), length(W, 0), pend(W*CAP*NH), act(W*CAP*NH), ver(W*CAP);
  R.rows = rows.data(); R.terminal = term.data(); R.close_done = cdone.data(); R.close_ev = cev.data(); R.boot = boot.data();
  R.reward = rew.data(); R.close_hurt = chu.data(); R.logp = logp.data(); R.value = val.data(); R.value_g = vg.data();
  R.open = open.data(); R.length = length.data(); R.pending = pend.data(); R.actions = act.data(); R.version = ver.data();
  R.arena.ensure(1 << 12); R.set = true;
  RowLayout L{STRIDE, 144, 152, 64, NH, 0};
  std::mutex amu; std::atomic<bool> done{false}; std::atomic<i64> stored{0}, errors{0};
  std::thread writer([&] {
    std::mt19937 r(seed);
    std::vector<uint8_t> rc(W*STRIDE);
    for (int it = 0; it < iters; it++) {
      i64 n_all = 1+r()%W;
      std::vector<i64> ws, go_idx;
      for (i64 w = 0; w < W && (i64)ws.size() < n_all; w++) if (r()%2 || W-w <= n_all-(i64)ws.size()) ws.push_back(w);
      for (size_t i = 0; i < ws.size(); i++) {
        uint8_t *s = &rc[i*STRIDE];
        memset(s, 0, STRIDE);
        s[0] = r()%20 == 0; s[1] = r()%15 == 0 ? (uint8_t)(1+r()%4) : 0;
        float f = (float)(r()%100)/10.0f; memcpy(s+4, &f, 4); f = (float)(r()%2); memcpy(s+8, &f, 4);
        s[12] = (uint8_t)(r()%128); s[13] = (uint8_t)(1+r()%8); s[14] = r()%10 == 0;
        for (int q = 0; q < 31; q++) { f = (float)(r()%1000)/1000.0f; memcpy(s+16+q*4, &f, 4); }
        int32_t nb = 8+(int32_t)(r()%48); memcpy(s+144, &nb, 4);
        for (int q = 0; q < 64; q++) s[152+q] = (uint8_t)(it+q);
        if (s[1] == 0 || s[1] == 4) go_idx.push_back((i64)i);
      }
      std::vector<float> out(go_idx.size()*(NH+3), .5f); std::vector<i64> a(go_idx.size()*NH, 1), p(go_idx.size()*NH, 0), kk(go_idx.size());
      std::vector<double> snap(ws.size()*N_SNAP);
      consume(c, R, amu, L, rc.data(), (i64)ws.size(), ws.data(), go_idx.data(), (i64)go_idx.size(), out.data(), a.data(), p.data(),
              it, nullptr, kk.data(), snap.data());
      for (i64 k : kk) stored += k >= 0;
      if (r()%64 == 0) std::this_thread::yield();
    }
    done = true;
  });
  std::thread reader([&] {
    i64 last = 0;
    while (!done) {
      i64 tot = 0;
      for (i64 w = 0; w < W; w++) {   // (open is published after length: open first, then length, never open >= length)
        i64 o = load_i64(&R.open[w]); i64 l = load_i64(&R.length[w]); tot += l; if (o >= l) errors++;
      }
      if (tot < last) errors++;
      last = tot;
      { std::lock_guard<std::mutex> l(amu); volatile uint8_t x = R.arena.used ? R.arena.cur[R.arena.used-1] : 0; (void)x; }
    }
  });
  writer.join(); reader.join();
  i64 tot = 0; for (i64 w = 0; w < W; w++) tot += length[w];
  std::printf("part 2: %d records, %lld stored rows (lengths %lld), arena %lld bytes grown %lld times, %lld errors\n", iters,
              (long long)stored.load(), (long long)tot, (long long)R.arena.used, (long long)R.arena.grown, (long long)errors.load());
  return errors.load() == 0 && tot == stored.load() ? 0 : 1;
}

int main(int argc, char **argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 20000;
  unsigned seed = argc > 2 ? (unsigned)atoi(argv[2]) : 1;
  bool stamp = !(argc > 3 && std::string(argv[3]) == "nostamp");
  int rc = part1(iters, seed, stamp);
  rc |= part2(iters, seed);
  std::printf("%s\n", rc == 0 ? "OK" : "FAILED");
  return rc;
}
