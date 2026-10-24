"""C39 figures for EXPERIMENTS.md from c39_curves.json (host 2 c39_curves.py): fresh-seed outcomes, damage and clear
time, evaluations, training statistics and the Room Buffer over the 1000 game hours. usage: plot_c39.py data.json out_dir"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'Source Han Sans CN', 'SimHei', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150,
                     'savefig.bbox': 'tight', 'legend.frameon': False,
                     'axes.titlepad': 16})   # room for the run-segment labels between the axes and the title

data = json.load(open(sys.argv[1], encoding='utf8'))
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
GROUPS = data['groups']
NAMES = {'normal': '普通房', 'horf': '单 Horf', 'horf_rocks': '石头 Horf', 'spawners': '生怪房'}
COLORS = {'normal': '#1f77b4', 'horf': '#ff7f0e', 'horf_rocks': '#2ca02c', 'spawners': '#d62728'}
BOUNDS = [s[2] for s in data['segments'][:-1]]          # 122.6, 245.2, 337.4
SEG_LABELS = ['t3', 't3r', 't3r2', 't3r3']
bins = data['bins']
mid = np.array([(b['lo'] + b['hi']) / 2 for b in bins])


def segments(ax, labels=True):
    for x in BOUNDS:
        ax.axvline(x, color='0.6', lw=0.8, ls=':')
    if labels:
        edges = [0] + BOUNDS + [1000]
        for a, b, name in zip(edges, edges[1:], SEG_LABELS):
            ax.text((a + b) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                    fontsize=8, color='0.45')


def series(g, kind, key):
    ys = []
    for b in bins:
        v = b['groups'][g][kind]
        ys.append(np.nan if v is None or v.get(key) is None else v[key])
    return np.array(ys, float)


def hours_axis(ax, log=False):
    if log:
        ax.set_xscale('log')
        ax.set_xlim(2, 1000)
        ax.set_xticks([2, 5, 10, 20, 50, 100, 200, 500, 1000])
        ax.set_xticklabels(['2', '5', '10', '20', '50', '100', '200', '500', '1000'])
    else:
        ax.set_xlim(0, 1000)
    ax.set_xlabel('游戏小时')


# 1. Fresh-seed clear rate, linear and log hours.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, log in zip(axes, (False, True)):
    for g in GROUPS:
        x = mid if not log else np.array([max(np.sqrt(max(b['lo'], 1) * b['hi']), 2.5) for b in bins])
        ax.plot(x, series(g, 'fresh', 'win'), color=COLORS[g], lw=1.6, marker='o', ms=2.5, label=NAMES[g])
    hours_axis(ax, log)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel('新种子清房率')
    if not log:
        segments(ax)
axes[0].legend(loc='lower right')
axes[0].set_title('新种子清房率（线性横轴）', loc='left', fontsize=10)
axes[1].set_title('同上，对数横轴', loc='left', fontsize=10)
fig.savefig(out / 'c39_fresh_clear.png')
plt.close(fig)

# 2. Fresh-seed death and timeout rates.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
for ax, key, title in zip(axes, ('death', 'timeout'), ('死亡率', '超时率（180 秒）')):
    for g in GROUPS:
        ax.plot(mid, series(g, 'fresh', key), color=COLORS[g], lw=1.5, marker='o', ms=2.5, label=NAMES[g])
    hours_axis(ax)
    segments(ax)
    ax.set_ylim(0, None)
    ax.set_ylabel(title)
    ax.set_title(f'新种子{title}', loc='left', fontsize=10)
axes[0].legend(loc='upper right')
fig.savefig(out / 'c39_fresh_death_timeout.png')
plt.close(fig)

# 3. Damage per episode and clear time.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
for g in GROUPS:
    axes[0].plot(mid, series(g, 'fresh', 'hurt'), color=COLORS[g], lw=1.5, marker='o', ms=2.5, label=NAMES[g])
    axes[1].plot(mid, series(g, 'fresh', 'clear_s'), color=COLORS[g], lw=1.5, marker='o', ms=2.5, label=NAMES[g])
for ax in axes:
    hours_axis(ax)
    segments(ax)
axes[0].set_ylim(0, None)
axes[0].set_ylabel('半心 / 局')
axes[0].set_title('新种子每局掉血（开局 6 个半心）', loc='left', fontsize=10)
axes[1].set_yscale('log')
axes[1].set_yticks([2, 5, 10, 20, 50])
axes[1].set_yticklabels(['2', '5', '10', '20', '50'])
axes[1].set_ylabel('秒')
axes[1].set_title('新种子清房用时中位（对数纵轴）', loc='left', fontsize=10)
axes[0].legend(loc='upper right')
fig.savefig(out / 'c39_fresh_damage_time.png')
plt.close(fig)

# 4. Evaluations (32 held-out seeds per group, full HP, 1 bomb).
evals = data['evals']
eh = np.array([e['hours'] for e in evals])


def ev(g, mode, key, frac=True):
    ys = []
    for e in evals:
        v = e['groups'][g].get(mode)
        if v is None:
            ys.append(np.nan)
        else:
            ys.append(v[key] / v['n'] if frac else v[key])
    return np.array(ys, float)


fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
panels = [(axes[0, 0], 'greedy', 'win', True, '取最大动作：清房比例'),
          (axes[0, 1], 'sampled', 'win', True, '采样：清房比例'),
          (axes[1, 0], 'greedy', 'timeout', True, '取最大动作：超时比例'),
          (axes[1, 1], 'greedy', 'clean', True, '取最大动作：无伤清房比例')]
for ax, mode, key, frac, title in panels:
    for g in GROUPS:
        ax.plot(eh, ev(g, mode, key, frac), color=COLORS[g], lw=1.5, marker='o', ms=3.5, label=NAMES[g])
    hours_axis(ax)
    segments(ax, labels=False)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, loc='left', fontsize=10)
axes[0, 0].legend(loc='lower right')
fig.suptitle('评估：每组 32 个 held-out 种子，满血、1 个炸弹（第 200 轮的生怪房只有取最大的 25 局）', x=0.02, ha='left',
             fontsize=10)
fig.tight_layout()
fig.savefig(out / 'c39_evals.png')
plt.close(fig)

# 5. Training statistics per rollout (rolling median over 9 rollouts for the noisy ones).
roll = data['rollouts']
rh = np.array([r['hours'] for r in roll])


def col(k):
    return np.array([np.nan if r.get(k) is None else r[k] for r in roll], float)


def smooth(y, w=9):
    out_ = np.full_like(y, np.nan)
    for i in range(len(y)):
        seg = y[max(0, i - w // 2): i + w // 2 + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out_[i] = np.median(seg)
    return out_


fig, axes = plt.subplots(2, 3, figsize=(13, 7.0))
spec = [(axes[0, 0], ['train/kl_full'], ['kl_full'], '每轮 KL（完整分布，C32）', None),
        (axes[0, 1], ['train/entropy_move', 'train/entropy_shoot'], ['移动头', '射击头'], '策略熵（最大 ln9=2.20 / ln5=1.61）', None),
        (axes[0, 2], ['train/explained_variance'], ['解释方差'], '价值的解释方差', (0, 1.02)),
        (axes[1, 0], ['train/learning_rate'], ['学习率'], '学习率（余弦 1e-3 → 1e-4）', None),
        (axes[1, 1], ['train/clip_fraction'], ['裁剪比例'], 'PPO 裁剪比例', None),
        (axes[1, 2], ['game/speed_x_realtime'], ['速度'], '速度（× 实时，含评估期间）', None)]
for ax, keys, labels, title, ylim in spec:
    for k, lab, c in zip(keys, labels, ('#444444', '#9467bd')):
        y = col(k)
        ax.plot(rh, y, color=c, lw=0.5, alpha=0.35)
        ax.plot(rh, smooth(y), color=c, lw=1.6, label=lab)
    hours_axis(ax)
    segments(ax, labels=False)
    ax.set_title(title, loc='left', fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    if len(keys) > 1:
        ax.legend(loc='upper right')
axes[0, 0].set_ylim(0, None)
axes[1, 2].set_ylim(0, None)
fig.suptitle('训练指标：每个点是一轮（32,768 步），粗线是 9 轮滚动中位；虚线是四段运行的分界', x=0.02, ha='left', fontsize=10)
fig.tight_layout()
fig.savefig(out / 'c39_training.png')
plt.close(fig)

# 6. Room Buffer per group.
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
for g in GROUPS:
    axes[0].plot(rh, col(f'buffer/{g}/p_mean'), color=COLORS[g], lw=1.4, label=NAMES[g])
    axes[1].plot(rh, col(f'buffer/{g}/p_frontier_share'), color=COLORS[g], lw=1.4, label=NAMES[g])
    axes[2].plot(rh, col(f'buffer/{g}/p_solved_share'), color=COLORS[g], lw=1.4, label=NAMES[g])
for ax, title in zip(axes, ('缓冲区种子的 p 均值（清房率 EMA）', '0.2 ≤ p ≤ 0.8 的份额', 'p > 0.9 的份额')):
    hours_axis(ax)
    segments(ax, labels=False)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, loc='left', fontsize=10)
axes[0].legend(loc='lower right')
fig.tight_layout()
fig.savefig(out / 'c39_buffer.png')
plt.close(fig)

print('written', sorted(p.name for p in out.glob('c39_*.png')))
