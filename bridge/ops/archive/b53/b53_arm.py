"""B53 helper for b53_run_arm.sh: the train_tok command of an arm, fresh or resumed.

  python b53_arm.py <arms.json> <arm> <total_game_hours> <out_dir>   -> prints the train_tok arguments

Resume (out_dir/checkpoints/last.pt exists): the game hours already done are progress.csv's last game_hours; the
remaining ones are trained with --resume, and a linear LR schedule continues where it stopped (--lr = the schedule's
value at the hours done, --lr-final unchanged, --game-hours = the remainder; with the arms' constant 1e-4 this changes
nothing). Prints DONE when nothing is left."""
import csv
import json
import shlex
import sys
from pathlib import Path


def flag(args, name, default):
    if name in args:
        return type(default)(args[args.index(name)+1])
    return default


def main():
    spec_path, arm, total, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), Path(sys.argv[4])
    spec = json.load(open(spec_path))
    a = spec['arms'][arm]
    args = shlex.split(spec['common'])+shlex.split(a['flags'])
    done = 0.0
    last = out/'checkpoints'/'last.pt'
    if last.is_file():
        prog = out/'progress.csv'
        rows = list(csv.DictReader(open(prog))) if prog.is_file() else []
        done = float(rows[-1]['game_hours']) if rows and rows[-1].get('game_hours') else 0.0
    left = total-done
    if left <= 1e-6:
        print('DONE')
        return
    lr, lr_final = flag(args, '--lr', 1e-4), flag(args, '--lr-final', 1e-4)
    if done > 0:
        lr_now = lr+(lr_final-lr)*min(1.0, done/total)
        i = args.index('--lr'); args[i+1] = repr(lr_now)
        args += ['--resume', str(last)]
    args += ['--game-hours', repr(left), '--port', str(a['port']), '--name', f'b53{arm}', '--out', str(out)]
    print(' '.join(shlex.quote(x) for x in args))


if __name__ == '__main__':
    main()
