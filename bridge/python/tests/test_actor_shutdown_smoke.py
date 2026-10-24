"""Exercise the trainer's actual last-half handoff with a saturated process queue."""
import ast
import multiprocessing as mp
from pathlib import Path


def actor_end(jobs, ready, last, stop):
    source = Path(__file__).resolve().parents[1] / 'train_tok.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'actor is not None and (not stopped())')
    last_at = next(i for i, stmt in enumerate(branch.body)
                   if ast.unparse(stmt) == "actor['last'].set()")
    jobs.put(b'x' * (4 * 1024 * 1024))  # larger than the pipe; feeder must wait for the learner
    ready.set()
    exec(compile(ast.Module(body=branch.body[:last_at + 1], type_ignores=[]), str(source), 'exec'),
         dict(jobs=jobs, actor={'last': last}))
    assert stop.wait(15)


def main():
    ctx = mp.get_context('spawn')
    jobs, ready, last, stop = ctx.Queue(), ctx.Event(), ctx.Event(), ctx.Event()
    actor = ctx.Process(target=actor_end, args=(jobs, ready, last, stop))
    actor.start()
    assert ready.wait(15)
    announced_before_drain = last.wait(0.3)
    payload = jobs.get(timeout=15)
    assert last.wait(15)
    stop.set()
    actor.join(15)
    assert actor.exitcode == 0
    jobs.close(); jobs.join_thread()
    assert len(payload) == 4 * 1024 * 1024
    assert not announced_before_drain, 'last-half event fired before queue feeder delivered the payload'
    print('ok last-half event waits for full payload; actor exits cleanly, no tail dropped')


if __name__ == '__main__':
    main()
