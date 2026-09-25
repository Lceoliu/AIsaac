"""FrameSampler whose per-step work runs as captured CUDA graphs (AB+ collection speed-up).

Measured on the RTX 3080 Ti for 16 environments (abplus_bench_sampler.py): the eager step (encode
the newest frames, 64-frame temporal Transformer, heads, masked sampling) takes ~10.8 ms, of which
the GPU is busy ~2 ms. encode_frames compacts entities with boolean indexing, .item() and nonzero(),
and SB3's masked distributions validate their arguments: ~37 host-device synchronisations and ~500
kernel launches per step. The same step with fixed shapes and no synchronisation, replayed as a
graph, takes ~1.7 ms.

Per chunk two graphs are captured once, at the start of a rollout when nothing else is in flight:
  encode  CombatTransformer.encode_frame_static on the chunk's newest frames (copied into fixed
          input tensors first); the raw frames and features are then written to the buffer as
          FrameSampler.encode does.
  act     the chunk's causal windows from the buffer at a position held in a device tensor, the
          temporal Transformer, heads, the same action masks as FrameSampler.action, Gumbel-max
          sampling of each action component (the categorical distribution) and its log-probability.
The graphs read the sampling policy's parameters in place (the learner's, or the actor copy with
asynchronous training), so weight updates need no re-capture; a different policy object is
re-captured at the next rollout start. Everything else
(episode resets of single environments, value estimates, deterministic actions, prefix
re-encoding) stays on the eager FrameSampler path.

Self-check: every check_every-th graph step also runs the eager path on the same inputs and
records the largest difference of the new features, of the log-probability of the sampled
actions and of the values (diffs, logged by train_abplus as sampler/*). Differences come from
float summation order only (~1e-5).
"""
import torch
import torch.nn.functional as F

from .gpu_env import decode_frame
from .gpu_ppo import FrameSampler

HUGE_NEG = -1e8   # sb3_contrib MaskableCategorical's masked logit


class GraphFrameSampler(FrameSampler):
    def __init__(self, model, check_every=1024):
        super().__init__(model)
        self.check_every = check_every
        self.nvec = [int(n) for n in self.env.action_space.nvec]
        self.device = model.device
        self.graphs = None                      # per chunk: dict(encode=..., act=...)
        self.captured = None                    # the policy the graphs read
        self.chunk_of = {id(ids): i for i, ids in enumerate(self.workers)}
        self.graph_steps = 0
        self.diffs = {'features': 0.0, 'log_prob': 0.0, 'value': 0.0, 'checks': 0}

    # ---- capture --------------------------------------------------------------------------------
    def begin(self):
        super().begin()
        if self.graphs is None or self.captured is not self.policy:
            torch.cuda.synchronize(self.device)
            self.graphs = [self._capture_chunk(i) for i in range(len(self.workers))]
            self.captured = self.policy
            torch.cuda.synchronize(self.device)

    def _graph(self, fn):
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode='thread_local'):
            out = fn()
        return graph, out

    def _capture_chunk(self, i):
        chunk, ids = self.env.chunks[i], self.workers[i]
        raw = decode_frame(chunk.slots[0].device, self.env.observation_space)
        inputs = {k: v.clone() for k, v in raw.items()}
        encode = self._graph(lambda: self.encoder.encode_frame_static(inputs))
        # Shape (1,), not a 0-dim tensor: indexing with a 0-dim integer tensor turns it into a Python int
        # (a host synchronisation, not allowed while capturing); (1,) broadcasts against ids.
        position = torch.full((1,), self.buffer.frame_pos, dtype=torch.long, device=self.device)
        act = self._graph(lambda: self._act(position, ids))
        return {'inputs': inputs, 'encode': encode, 'position': position, 'act': act}

    def _act(self, position, ids):
        """FrameSampler.action with the position as a device tensor and Gumbel-max sampling."""
        b, policy = self.buffer, self.policy
        length = b.lengths[position, ids]
        j = torch.arange(b.history, device=self.device)[None, :]
        valid = j < length[:, None]
        times = torch.where(valid, position - length[:, None] + 1 + j, 0)
        fused = self.features[times, ids[:, None]].masked_fill(~valid[:, :, None], 0)
        elapsed = b.frames['time'][times, ids[:, None]].masked_fill(~valid, 0)
        seq = self.encoder.temporal_features(fused, elapsed, valid)
        latent = seq[torch.arange(len(ids), device=self.device), valid.long().sum(-1) - 1]
        masks = torch.ones((len(ids), sum(self.nvec)), dtype=torch.bool, device=self.device)
        masks[:, 46] = b.frames['player'][position, ids, 9] > 0
        masks[:, 48] = False
        pi, vf = policy.mlp_extractor(latent)
        logits = torch.where(masks, policy.action_net(pi), torch.full_like(masks, HUGE_NEG, dtype=torch.float32))
        noise = torch.rand(logits.shape, device=self.device).clamp_(1e-10, 1 - 1e-7)
        actions, log_prob = [], 0
        for part, u in zip(torch.split(logits, self.nvec, -1), torch.split(noise, self.nvec, -1)):
            a = torch.argmax(part - torch.log(-torch.log(u)), -1)
            actions.append(a)
            log_prob = log_prob + F.log_softmax(part, -1).gather(-1, a[:, None])[:, 0]
        return torch.stack(actions, -1), policy.value_net(vf).flatten(), log_prob, masks

    # ---- per-step use ---------------------------------------------------------------------------
    def encode(self, position, workers, raw):
        i = self.chunk_of.get(id(workers))
        if self.graphs is None or i is None:
            return super().encode(position, workers, raw)
        g = self.graphs[i]
        for k, v in g['inputs'].items():
            v.copy_(raw[k], non_blocking=True)
        graph, out = g['encode']
        graph.replay()
        for k, v in raw.items():
            self.buffer.frames[k][position, workers] = v
        self.features[position, workers] = out
        self.encoded_frames += len(workers)
        if self.check_every and self.graph_steps % self.check_every == 0:
            eager = self.encoder.encode_frames({k: v[:, None] for k, v in raw.items()})[:, 0]
            self._note('features', (eager - out).abs().max())

    def action(self, position, ids, deterministic=False):
        i = self.chunk_of.get(id(ids))
        if self.graphs is None or i is None or deterministic:
            return super().action(position, ids, deterministic)
        g = self.graphs[i]
        g['position'].fill_(position)
        graph, out = g['act']
        graph.replay()
        actions, values, log_prob, masks = out
        if self.check_every and self.graph_steps % self.check_every == 0:
            policy = self.policy
            pi, vf = policy.mlp_extractor(self.latent(position, ids))
            dist = policy._get_action_dist_from_latent(pi)
            dist.apply_masking(masks)
            self._note('log_prob', (dist.log_prob(actions) - log_prob).abs().max())
            self._note('value', (policy.value_net(vf).flatten() - values).abs().max())
            self.diffs['checks'] += 1
        self.graph_steps += 1
        return actions, values, log_prob, masks

    def _note(self, key, value):
        self.diffs[key] = max(self.diffs[key], float(value))
