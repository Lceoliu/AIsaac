-- 找隐藏房: the shared scoreboard on Supabase. Run once in the project's SQL Editor.
--
-- The page talks to it with the project's anon public key (web/static/js/scoreboard.js). That key is
-- meant to be public; what it can do is set here: add a score and read the boards, nothing else (no
-- update, no delete). One row per player (a random id kept in the browser), game, seed and floor: the
-- first score counts, a replay of the same floor is refused.
--
-- The page computes the points itself, so a determined player could send made-up scores; the checks
-- below only keep them within what a floor can give.

create table if not exists public.hr_scores (
  id         bigint generated always as identity primary key,
  client_id  uuid        not null,
  player     text        not null check (char_length(player) between 1 and 16),
  game       text        not null check (game in ('abplus', 'repplus')),
  mode       text        not null check (mode in ('daily', 'random')),
  day        date,
  seed       bigint      not null check (seed between 1 and 4294967295),
  floor      smallint    not null check (floor between 0 and 15),
  points     smallint    not null check (points between 0 and 500),
  bombs      smallint    not null check (bombs between 0 and 300),
  keys       smallint    not null check (keys between 0 and 300),
  hints      boolean     not null default false,
  created_at timestamptz not null default now(),
  unique (client_id, game, seed, floor),
  check ((mode = 'daily') = (day is not null))
);

alter table public.hr_scores enable row level security;

drop policy if exists "read scores" on public.hr_scores;
create policy "read scores" on public.hr_scores for select to anon, authenticated using (true);

-- a daily score only for today or yesterday (China time), so old challenges cannot be filled in later
drop policy if exists "add a score" on public.hr_scores;
create policy "add a score" on public.hr_scores for insert to anon, authenticated
  with check (day is null or day between (now() at time zone 'Asia/Shanghai')::date - 1
                                     and (now() at time zone 'Asia/Shanghai')::date);

grant select, insert on public.hr_scores to anon, authenticated;

-- the boards: one row per player, the latest name they used
create or replace view public.hr_board_total with (security_invoker = true) as
  select game, client_id,
         (array_agg(player order by created_at desc))[1] as player,
         sum(points)::int as points, count(*)::int as floors, max(created_at) as last_at
  from public.hr_scores
  group by game, client_id;

create or replace view public.hr_board_daily with (security_invoker = true) as
  select day, game, client_id,
         (array_agg(player order by created_at desc))[1] as player,
         sum(points)::int as points, count(*)::int as floors, max(created_at) as last_at
  from public.hr_scores
  where mode = 'daily'
  group by day, game, client_id;

grant select on public.hr_board_total, public.hr_board_daily to anon, authenticated;
