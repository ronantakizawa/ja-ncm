"""Token read-out from the tokenizer decoder's logits (codes -> pre-tokens).

Upstream reads tokens with a plain argmax. Its released LM1B generation config instead uses
``eval_shard.pick_tokens``: a left-to-right greedy pass with an adjacency penalty (lower the
logit of the last few emitted tokens) and an optional frequency penalty. We port the
adjacency penalty and add a UTF-8 constraint for byte-level BPE vocabularies (Qwen3):

* adjacency "a,w,d": subtract a * d**j from the logit of the token emitted j+1 steps ago,
  for the last w tokens (released LM1B value: "10,4,0.5"). Parallel decoding of
  neighbouring positions otherwise produces doubled tokens ("事業事業").
* utf8: allow only tokens whose bytes continue the pending multi-byte character exactly
  and start a valid one otherwise. Qwen3 splits many CJK characters into 2-3 byte tokens;
  one wrong byte token renders as U+FFFD and garbles its neighbours.

Both are greedy, deterministic and cost one [B, V] pass per position.
"""
import torch

# GPT-2 / Qwen byte-level BPE: printable stand-in characters for raw bytes.


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _utf8_step(state, b):
    """state = continuation bytes still owed; returns the new state or None if invalid."""
    if state > 0:
        return state - 1 if 0x80 <= b <= 0xBF else None
    if b < 0x80:
        return 0
    if 0xC2 <= b <= 0xDF:
        return 1
    if 0xE0 <= b <= 0xEF:
        return 2
    if 0xF0 <= b <= 0xF4:
        return 3
    return None


def token_bytes(hf_tokenizer):
    """Byte string of every token id (special / added tokens -> b"")."""
    dec = _bytes_to_unicode()
    special = set(hf_tokenizer.all_special_ids)
    V = len(hf_tokenizer)
    out = []
    for i in range(V):
        if i in special:
            out.append(b"")
            continue
        s = hf_tokenizer.convert_ids_to_tokens(i)
        if s is None:
            out.append(b"")
            continue
        if all(ch in dec for ch in s):
            out.append(bytes(dec[ch] for ch in s))
        else:   # added token stored as plain text
            out.append(s.encode("utf-8"))
    return out


class Utf8Constraint:
    """allowed[s, v]: token v may follow pending-continuation state s; nxt[s, v]: next state."""

    def __init__(self, tok_bytes, vocab_size=None):
        V = vocab_size or len(tok_bytes)
        tok_bytes = list(tok_bytes) + [None] * max(0, V - len(tok_bytes))   # padded logit columns
        allowed = torch.zeros(4, V, dtype=torch.bool)
        nxt = torch.zeros(4, V, dtype=torch.long)
        for v, bs in enumerate(tok_bytes[:V]):
            if bs is None:          # not a real token: never allowed
                continue
            for s0 in range(4):
                s = s0
                if not bs:          # special token: only between whole characters
                    ok = s0 == 0
                else:
                    ok = True
                    for b in bs:
                        s = _utf8_step(s, b)
                        if s is None:
                            ok = False
                            break
                if ok:
                    allowed[s0, v] = True
                    nxt[s0, v] = s
        self.allowed, self.nxt = allowed, nxt

    @classmethod
    def from_pretokenizer(cls, pretok, vocab_size):
        return cls(token_bytes(pretok.hf_tokenizer), vocab_size)

    def to(self, device):
        self.allowed, self.nxt = self.allowed.to(device), self.nxt.to(device)
        return self


def parse_adjacency(spec):
    if not spec:
        return None
    v = [float(x) for x in str(spec).split(",")]
    return (v[0], int(v[1]) if len(v) > 1 else 1, v[2] if len(v) > 2 else 1.0)


@torch.no_grad()
def sample_tokens(logits, temperature, top_p=1.0, chunk=16):
    """Sample ids from softmax(logits / T) (optionally nucleus), chunked over the batch so a
    [B, T, V] float copy of a 152k vocabulary is never held at once."""
    out = []
    for i in range(0, logits.shape[0], chunk):
        lg = logits[i:i + chunk].float() / temperature
        if top_p < 1.0:
            sl, si = torch.sort(lg, descending=True, dim=-1)
            p = torch.softmax(sl, -1)
            sl = sl.masked_fill(p.cumsum(-1) - p > top_p, float("-inf"))
            lg = torch.full_like(lg, float("-inf")).scatter(-1, si, sl)
        b, t, v = lg.shape
        out.append(torch.multinomial(torch.softmax(lg, -1).view(-1, v), 1).view(b, t))
    return torch.cat(out)


def pick_tokens(logits, adjacency=None, utf8=None):
    """logits [B, T, V] -> ids [B, T]. Plain argmax when no option is set."""
    if adjacency is None and utf8 is None:
        return logits.argmax(-1)
    B, T, V = logits.shape
    dev = logits.device
    lg = logits.float()
    out = torch.empty(B, T, dtype=torch.long, device=dev)
    hist = []
    state = torch.zeros(B, dtype=torch.long, device=dev)
    ar = torch.arange(B, device=dev)
    for t in range(T):
        step = lg[:, t, :V].clone()
        if adjacency is not None:
            a, w, d = adjacency
            for j, h in enumerate(reversed(hist[-w:])):
                step[ar, h] -= a * d ** j
        if utf8 is not None:
            mask = utf8.allowed[state, :V]                       # [B, V]
            if t == T - 1:                                       # end on a whole character
                mask = mask & (utf8.nxt[state, :V] == 0)
            step = step.masked_fill(~mask, float("-inf"))
        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        if utf8 is not None:
            state = utf8.nxt[state, p]
    return out


def readout_from_config(G, pretok, vocab_size, device):
    """(adjacency, utf8) from generation.readout: {adjacency: "10,4,0.5", utf8: true}."""
    R = G.get("readout", None) if G is not None else None
    if not R:
        return None, None
    adj = parse_adjacency(R.get("adjacency", ""))
    u = Utf8Constraint.from_pretokenizer(pretok, vocab_size).to(device) if bool(R.get("utf8", False)) else None
    return adj, u
