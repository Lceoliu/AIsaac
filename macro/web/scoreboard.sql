-- 找隐藏房: the shared scoreboard on Supabase. Run once in the project's SQL Editor.
--
-- The page talks to it with the project's anon public key (web/static/js/scoreboard.js). That key is
-- meant to be public; what it can do is set here: add a score and read the boards, nothing else (no
-- update, no delete). One row per player (a random id kept in the browser), game, seed and floor: the
-- first score counts, a replay of the same floor is refused.
--
-- The page computes the points itself, so a determined player could send made-up scores; the checks
-- below only keep them within what a floor can give. Since the 2026-09-29 rules (js/score.js) a floor
-- gives up to 800; a table made before then needs its cap raised once:
--   alter table public.hr_scores drop constraint hr_scores_points_check;
--   alter table public.hr_scores add constraint hr_scores_points_check check (points between 0 and 1000);
--
-- Since 2026-10-21 a daily board breaks ties by time: each floor's score carries the time it took
-- (ms, at most an hour), and hr_board_daily sums it. A table made before then needs the column once
-- (then run the hr_board_daily view below again):
--   alter table public.hr_scores add column if not exists ms integer check (ms between 0 and 3600000);

create table if not exists public.hr_scores (
  id         bigint generated always as identity primary key,
  client_id  uuid        not null,
  player     text        not null check (char_length(player) between 1 and 16),
  game       text        not null check (game in ('abplus', 'repplus')),
  mode       text        not null check (mode in ('daily', 'random')),
  day        date,
  seed       bigint      not null check (seed between 1 and 4294967295),
  floor      smallint    not null check (floor between 0 and 15),
  points     smallint    not null check (points between 0 and 1000),
  bombs      smallint    not null check (bombs between 0 and 300),
  keys       smallint    not null check (keys between 0 and 300),
  hints      boolean     not null default false,
  ms         integer     check (ms between 0 and 3600000),     -- the floor's time; null from older pages
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

-- the boards: one row per player, the latest name they used; a daily board's time is the sum of the
-- floors' times when every floor has one
create or replace view public.hr_board_total with (security_invoker = true) as
  select game, client_id,
         (array_agg(player order by created_at desc))[1] as player,
         sum(points)::int as points, count(*)::int as floors, max(created_at) as last_at
  from public.hr_scores
  group by game, client_id;

create or replace view public.hr_board_daily with (security_invoker = true) as
  select day, game, client_id,
         (array_agg(player order by created_at desc))[1] as player,
         sum(points)::int as points, count(*)::int as floors, max(created_at) as last_at,
         case when count(ms) = count(*) then sum(ms)::int end as ms
  from public.hr_scores
  where mode = 'daily'
  group by day, game, client_id;

grant select on public.hr_board_total, public.hr_board_daily to anon, authenticated;

-- The replays (2026-10-21): each daily floor's moves, "b43 k57 …" (bomb or Red Key, cell), sent with
-- its score. They give a day's answers away, so nobody reads the table: hr_replay hands a day's moves
-- to a player who has finished that day's challenge, shown by the 7 floors of moves sent under their
-- own secret token (kept in their browser, never shown on a board).
create table if not exists public.hr_moves (
  client_id  uuid        not null,
  token      uuid        not null,
  game       text        not null check (game in ('abplus', 'repplus')),
  day        date        not null,
  floor      smallint    not null check (floor between 0 and 15),
  moves      text        not null check (char_length(moves) <= 2000 and moves ~ '^([bk][0-9]{1,3}( [bk][0-9]{1,3})*)?$'),
  gave_up    boolean     not null default false,
  created_at timestamptz not null default now(),
  primary key (client_id, game, day, floor)
);

alter table public.hr_moves enable row level security;

drop policy if exists "add moves" on public.hr_moves;
create policy "add moves" on public.hr_moves for insert to anon, authenticated
  with check (day between (now() at time zone 'Asia/Shanghai')::date - 1
                      and (now() at time zone 'Asia/Shanghai')::date);

grant insert on public.hr_moves to anon, authenticated;     -- no select: read through hr_replay only

create or replace function public.hr_replay(p_game text, p_day date, p_player uuid, p_client uuid, p_token uuid)
returns table (floor smallint, moves text, gave_up boolean)
language plpgsql stable security definer set search_path = public as $$
#variable_conflict use_column
begin
  if (select count(*) from public.hr_moves m
      where m.game = p_game and m.day = p_day and m.client_id = p_client and m.token = p_token) < 7 then
    raise exception 'finish this day''s challenge first' using errcode = 'P0001';
  end if;
  return query
    select m.floor, m.moves, m.gave_up from public.hr_moves m
    where m.game = p_game and m.day = p_day and m.client_id = p_player
    order by m.floor;
end
$$;

revoke all on function public.hr_replay(text, date, uuid, uuid, uuid) from public;
grant execute on function public.hr_replay(text, date, uuid, uuid, uuid) to anon, authenticated;
