# 딥러닝 A to Z · 글 안상선 · CC BY-NC 4.0 (비상업적 이용, 출처 표기 필수)
# ===============================================================
# 특별편 3 마인크래프트 속 초파리 뇌 · 코드 2: 일반 딥러닝 다섯 가지 대 초파리 모델 세 가지 (PyTorch)
#  - 라이브러리: torch, numpy, matplotlib (Colab 기본 설치)
#  - 실행 확인: 2026-10-04, torch 2.x · numpy 2.4 · matplotlib 3.10 (CPU)
#  - 하는 일:
#      ① MLP·CNN·RNN·GRU·LSTM과 초파리 모델 세 가지(초파리 방식, 기억+본능, 배선 고정 스파이킹)를 만든다
#      ② 같은 데이터(그림 16×16)와 같은 분할로 학습하고 연결 수·정확도를 표로 비교한다
#      ③ 기억 인출 차단, 스파이크 기록, 케니언 세포 희소성 실험을 한다
# ===============================================================
# ---------- 0. 준비: 라이브러리와 재현성 ----------
import json, math, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| torch", torch.__version__)

FOCUS = "ConnectomeLIF"   # 이 코드를 내려받은 페이지의 모델 (표에서 ★로 표시)

# ---------- 1. 데이터: 학습기에서 내려받은 JSON이 있으면 쓰고, 없으면 예시 데이터를 만든다 ----------
DATA_FILE = "fly_lab_data.json"

try:                                   # 코랩이면 파일 올리기 창을 띄운다 (건너뛰려면 취소)
    from google.colab import files     # type: ignore
    if not os.path.exists(DATA_FILE):
        print("학습기에서 내려받은 fly_lab_data.json을 올리세요. 없으면 '취소'를 누르면 예시 데이터를 씁니다.")
        try:
            up = files.upload()
            for name in up:
                if name.endswith(".json"):
                    os.replace(name, DATA_FILE)
        except Exception as e:
            print("업로드 건너뜀:", e)
except ImportError:
    pass

S, FS = 64, 16                          # 그림 버퍼 64×64 → 특징 16×16 (페이지 학습기와 같은 규칙)

def draw_shape(name, rng):
    """페이지의 예시 생성기와 같은 방식으로 흔들린 모양을 64×64 버퍼에 그린다."""
    b = np.zeros((S, S), np.float32)
    cx, cy = 32 + (rng.random() - .5) * 16, 32 + (rng.random() - .5) * 16
    s, a, r = 12 + rng.random() * 10, (rng.random() - .5) * 1.6, 1.2 + rng.random() * 1.6
    sx_, sy_ = 0.7 + rng.random() * 0.6, 0.7 + rng.random() * 0.6        # 가로세로 비율 흔들기
    def rot(P):
        c, sn = math.cos(a), math.sin(a)
        P = [(x * sx_ + (rng.random() - .5) * s * .35, y * sy_ + (rng.random() - .5) * s * .35) for x, y in P]  # 손떨림
        return [(cx + x * c - y * sn, cy + x * sn + y * c) for x, y in P]
    if name == "circle":
        P = [(math.cos(t) * s * (1 + (rng.random() - .5) * .12), math.sin(t) * s) for t in np.linspace(0, 2 * math.pi, 25)]
        segs = [rot(P)]
    elif name == "cross":
        segs = [rot([(-s, -s), (s, s)]), rot([(s, -s), (-s, s)])]
    elif name == "hline":
        segs = [rot([(-s * 1.2, 0), (s * 1.2, (rng.random() - .5) * 6)])]
    elif name == "tri":
        P = [(0, -s), (s * .95, s * .75), (-s * .95, s * .75)]; segs = [rot(P + P[:1])]
    else:  # square
        P = [(-s, -s), (s, -s), (s, s), (-s, s)]; segs = [rot(P + P[:1])]
    yy, xx = np.mgrid[0:S, 0:S]
    for P in segs:
        for (x0, y0), (x1, y1) in zip(P[:-1], P[1:]):
            for t in np.linspace(0, 1, max(2, int(math.hypot(x1 - x0, y1 - y0) * 1.5))):
                d = np.hypot(xx - (x0 + (x1 - x0) * t), yy - (y0 + (y1 - y0) * t))
                b = np.maximum(b, np.clip(r + 1 - d, 0, 1))
    return b

def featurize(b):
    """그린 부분을 찾아 정사각형으로 잘라 16×16으로 줄인다 (위치·크기 차이를 없앰)."""
    ys, xs = np.where(b > .15)
    if len(xs) == 0:
        return np.zeros(FS * FS, np.float32)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.15 + 4
    f = np.zeros((FS, FS), np.float32)
    for j in range(FS):
        for i in range(FS):
            sx, sy = cx - side / 2 + (i + .5) * side / FS, cy - side / 2 + (j + .5) * side / FS
            acc = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    X, Y = round(sx + dx * side / FS / 3), round(sy + dy * side / FS / 3)
                    acc.append(b[Y, X] if 0 <= X < S and 0 <= Y < S else 0.)
            f[j, i] = np.mean(acc)
    return (f / f.max()).ravel() if f.max() > 0 else f.ravel()

if os.path.exists(DATA_FILE):
    d = json.load(open(DATA_FILE, encoding="utf-8"))
    CLASSES, X, Y = d["classes"], np.array(d["X"], np.float32), np.array(d["Y"], np.int64)
    print(f"학습기 데이터: {len(X)}개, 클래스 {CLASSES}")
else:
    rng = np.random.default_rng(SEED)
    CLASSES = ["동그라미", "네모", "세모", "엑스"]          # 동그라미·네모·세모는 서로 닮아서 적당히 어렵다
    shapes = ["circle", "square", "tri", "cross"]
    X = np.stack([featurize(draw_shape(shapes[c], rng)) for c in range(4) for _ in range(40)])
    Y = np.repeat(np.arange(4), 40)
    print(f"예시 데이터: {len(X)}개, 클래스 {CLASSES}")

C = len(CLASSES)
fig, ax = plt.subplots(1, min(8, len(X)), figsize=(12, 2))
for k, i in enumerate(np.linspace(0, len(X) - 1, len(ax)).astype(int)):
    ax[k].imshow(X[i].reshape(FS, FS), cmap="magma"); ax[k].set_title(str(Y[i])); ax[k].axis("off")
plt.show()

# ---------- 2. 훈련 / 검증 / 테스트 나누기 — 정확도는 반드시 학습에 쓰지 않은 데이터로 잰다 ----------
idx = np.random.default_rng(SEED).permutation(len(X))
n_te, n_va = max(C, len(X) // 5), max(C, len(X) // 5)
te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]
Xt = torch.tensor(X, dtype=torch.float32, device=device)
Yt = torch.tensor(Y, dtype=torch.long, device=device)
print(f"훈련 {len(tr)} · 검증 {len(va)} · 테스트 {len(te)}")

# ---------- 3. 일반 딥러닝 모델 다섯 가지 — 모든 가중치를 역전파로 학습한다 ----------
class MLP(nn.Module):                      # 범용 단일 신경망 (완전 연결)
    def __init__(s, h=64):
        super().__init__(); s.net = nn.Sequential(nn.Linear(256, h), nn.ReLU(), nn.Linear(h, C))
    def forward(s, x): return s.net(x)

class CNN(nn.Module):                      # 같은 필터를 이미지 전체에 공유
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        s.head = nn.Linear(16 * 4 * 4, C)
    def forward(s, x): return s.head(s.f(x.view(-1, 1, 16, 16)).flatten(1))

class SeqNet(nn.Module):                   # 그림의 16개 행을 "시간 순서"로 읽는다
    def __init__(s, kind="rnn", h=64):
        super().__init__()
        s.rnn = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind](16, h, batch_first=True)
        s.head = nn.Linear(h, C)
    def forward(s, x):
        out, _ = s.rnn(x.view(-1, 16, 16))   # (배치, 시간 16, 입력 16)
        return s.head(out[:, -1])            # 마지막 시점의 기억으로 판단

# ---------- 4. 초파리에서 온 모델 세 가지 — 대부분의 가중치가 고정이다 ----------
def sparse_fixed(n_out, n_in, k, g):
    """각 출력 뉴런이 입력 k개를 무작위로 골라 1로 연결 (학습하지 않음)."""
    W = torch.zeros(n_out, n_in)
    for i in range(n_out):
        W[i, torch.randperm(n_in, generator=g)[:k]] = 1.0
    return W

class FlyRandomProjection(nn.Module):      # ①번 페이지: 투사 뉴런 → 케니언 세포(고정) → 출력(학습)
    def __init__(s, n_kc=1000, k=10, top=0.08):
        super().__init__(); g = torch.Generator().manual_seed(7)
        s.register_buffer("W_kc", sparse_fixed(n_kc, 256, k, g))   # buffer = 학습 안 되는 텐서
        s.top = top; s.readout = nn.Linear(n_kc, C)
    def features(s, x): return s.kc(x)           # 고정된 부분
    def head(s, f): return s.readout(f)           # 학습되는 부분
    def kc(s, x):
        a = x @ s.W_kc.T
        th = a.topk(max(1, int(a.shape[1] * s.top)), dim=1).values[:, -1:]   # 상위 8%만 (APL 억제)
        return (a >= th.clamp_min(1e-6)).float()
    def forward(s, x): return s.head(s.features(x))

class MemoryInnate(FlyRandomProjection):   # ②번 페이지: 합류 = 1.0·본능 + 0.6·기억
    def __init__(s):
        super().__init__()
        s.register_buffer("innate", torch.zeros(C))   # 측각: 진화가 정한 값 (학습 안 함)
        s.retrieve = True                               # False = 합류 뉴런의 기억 입력 차단 실험
        s.gain = 0.6                                    # 기억 경로의 무게
    def head(s, f):
        memory = s.readout(f)
        return 1.0 * s.innate + (s.gain * memory if s.retrieve else 0 * memory)

class ConnectomeLIF(nn.Module):            # ③번 페이지: 배선 고정 스파이킹 + 해독층만 학습
    def __init__(s, n=160, T=40):
        super().__init__(); g = torch.Generator().manual_seed(11)
        s.n, s.T = n, T
        s.register_buffer("W_in", sparse_fixed(n, 256, 24, g) * 0.28)
        sign = torch.where(torch.rand(n, generator=g) < 0.8, 0.18, -0.5)   # 데일의 법칙: 뉴런마다 부호 하나
        s.register_buffer("W_rec", sparse_fixed(n, n, 8, g).T * sign[:, None])   # (보내는 뉴런, 받는 뉴런)
        s.readout = nn.Linear(n, C)
    @torch.no_grad()
    def rates(s, x, seed=5, keep_raster=False):
        g = torch.Generator(device=x.device).manual_seed(seed)
        B = x.shape[0]; v = torch.zeros(B, s.n, device=x.device); ref = torch.zeros_like(v)
        syn = torch.zeros_like(v); cnt = torch.zeros_like(v); raster = []
        for t in range(s.T):
            spk_in = (torch.rand(x.shape, generator=g, device=x.device) < 0.45 * x).float()   # 푸아송 입력
            I = syn + spk_in @ s.W_in.T
            v = torch.where(ref > 0, torch.zeros_like(v), 0.9 * v + I)   # 새는 양동이
            spk = ((v > 1) & (ref <= 0)).float()
            v = v * (1 - spk); ref = torch.clamp(ref - 1, min=0) + 2 * spk
            syn = spk @ s.W_rec; cnt += spk
            if keep_raster: raster.append(spk[0].cpu())
        r = cnt / (s.T * 0.5)
        return (r, torch.stack(raster)) if keep_raster else r
    def features(s, x): return s.rates(x)
    def head(s, f): return s.readout(f)
    def forward(s, x): return s.head(s.features(x))

# ---------- 5. 같은 학습 루프 — 학습되는 파라미터(requires_grad=True)만 옵티마이저에 넣는다 ----------
def count(m):
    # 고정 연결(buffer)은 0이 아닌 값, 즉 실제로 이어진 연결만 센다
    total = sum(p.numel() for p in m.parameters()) + sum(int((b != 0).sum()) for b in m.buffers())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, train

def evaluate(m, ids):
    m.eval()
    with torch.no_grad():
        return (m(Xt[ids]).argmax(1) == Yt[ids]).float().mean().item()

def fit(m, epochs=40, lr=3e-3, bs=16):
    m = m.to(device)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    hist = {"train_loss": [], "val_acc": []}
    for ep in range(epochs):
        m.train(); perm = tr[np.random.permutation(len(tr))]; tot = 0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = F.cross_entropy(m(Xt[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        hist["train_loss"].append(tot / len(tr)); hist["val_acc"].append(evaluate(m, va))
    return m, hist

@torch.no_grad()
def fit_local(m, epochs=40, lr=0.05, bs=16):
    """초파리 모델용: 역전파 없이 한 층만 고친다. ΔW = lr · (정답 − 예측) ⊗ 특징 (델타 규칙).
    고정된 부분(케니언 세포·스파이킹 신경망)의 특징은 한 번만 계산한다.
    ※ MemoryInnate의 '정답 − 예측'은 실제 도파민(보상·처벌 한 가지 값)을 정답 라벨로 단순화한 것이다."""
    m = m.to(device); m.eval()
    feats = m.features(Xt); onehot = F.one_hot(Yt, C).float()
    scale = getattr(m, "gain", 1.0)
    hist = {"train_loss": [], "val_acc": []}
    for ep in range(epochs):
        perm = tr[np.random.permutation(len(tr))]; tot = 0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; f = feats[b]; z = m.head(f)
            err = (onehot[b] - F.softmax(z, 1)) * scale          # 오차 신호
            m.readout.weight += lr * err.T @ f / len(b)          # 국소 규칙: 오차 × 앞 뉴런 활동
            m.readout.bias += lr * err.mean(0)
            tot += F.cross_entropy(z, Yt[b], reduction="sum").item()
        hist["train_loss"].append(tot / len(tr))
        hist["val_acc"].append((m.head(feats[va]).argmax(1) == Yt[va]).float().mean().item())
    return m, hist

USE_LOCAL_RULE = True    # False로 바꾸면 초파리 모델도 Adam + 역전파로 학습 (결과 비교용)

# ---------- 6. 여덟 모델을 모두 학습하고 표로 비교 ----------
MODELS = {
    "MLP": lambda: MLP(), "CNN": lambda: CNN(),
    "RNN": lambda: SeqNet("rnn"), "GRU": lambda: SeqNet("gru"), "LSTM": lambda: SeqNet("lstm"),
    "FlyRandomProjection": lambda: FlyRandomProjection(),
    "MemoryInnate": lambda: MemoryInnate(), "ConnectomeLIF": lambda: ConnectomeLIF(),
}
SEEDS = [0, 1]          # 시드 두 개로 따로 학습해 흔들림을 본다
results, trained = [], {}
for name, make in MODELS.items():
    fly = name in ("FlyRandomProjection", "MemoryInnate", "ConnectomeLIF")
    row = {"va": [], "te": []}
    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd); t0 = time.time()
        if fly and USE_LOCAL_RULE:
            m, h = fit_local(make(), epochs=200, lr=0.5)   # 역전파 없음 (특징을 미리 계산해 에포크가 싸다)
        else:
            m, h = fit(make(), lr=1e-2 if fly else 3e-3)
        row["va"].append(h["val_acc"][-1]); row["te"].append(evaluate(m, te))
        if sd == SEEDS[0]: trained[name] = (m, h)
    tot, trn = count(m)
    results.append((name, tot, trn, row["va"], row["te"]))

print("초파리 모델 학습 방식:", "델타 규칙 (역전파 없음)" if USE_LOCAL_RULE else "Adam + 역전파")
print(f"{'모델':<22}{'전체 연결':>10}{'학습되는 연결':>12}{'비율':>6}   검증 정확도(시드0/1)   테스트 정확도(시드0/1)")
for n, tot, trn, va_acc, te_acc in results:
    star = " ★" if n == FOCUS else ""
    print(f"{n + star:<22}{tot:>10,}{trn:>12,}{trn / tot:>6.0%}   {va_acc[0]:.2f} / {va_acc[1]:.2f}            {te_acc[0]:.2f} / {te_acc[1]:.2f}")
print(f"(훈련 {len(tr)} · 검증 {len(va)} · 테스트 {len(te)}개. 그림 수가 적어 정확도는 ±0.1 정도 흔들린다)")

# ---------- 7. 학습 곡선: 훈련 손실은 계속 내려가도 검증 정확도가 멈추면 과적합이다 ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, (m, h) in trained.items():
    lw = 3 if name == FOCUS else 1.3
    ax[0].plot(h["train_loss"], label=name, lw=lw); ax[1].plot(h["val_acc"], label=name, lw=lw)
ax[0].set_title("train loss"); ax[1].set_title("validation accuracy"); ax[1].legend(fontsize=8)
plt.show()

# ---------- 8. 초파리 모델만의 실험 ----------
# (1) ②번: 기억 인출 차단 — 합류 뉴런에 들어가는 기억 신호를 끄면 본능만 남는다
mi = trained["MemoryInnate"][0]
mi.retrieve = False; off = evaluate(mi, te); mi.retrieve = True; on = evaluate(mi, te)
print(f"기억 인출 켬: {on:.2f}  →  차단: {off:.2f}")
print("차단하면 본능 값만 남는다. 본능이 모두 0이면 모든 점수가 같아져 첫 클래스만 고르게 된다.")

# (2) ③번: 배선 고정 스파이킹 신경망의 스파이크 기록 (첫 테스트 그림)
lif = trained["ConnectomeLIF"][0]
_, raster = lif.rates(Xt[te[:1]], keep_raster=True)
plt.figure(figsize=(10, 3)); t_, n_ = torch.nonzero(raster, as_tuple=True)
plt.scatter(t_.numpy(), n_.numpy(), s=3, c="k"); plt.xlabel("time step"); plt.ylabel("neuron")
plt.title("ConnectomeLIF spikes (weights fixed, only readout trained)"); plt.show()

# (3) ①번: 케니언 세포 코드는 얼마나 희소한가
fr = trained["FlyRandomProjection"][0]
with torch.no_grad():
    k = fr.kc(Xt[te[:20]])
print(f"케니언 세포 중 켜진 비율: {k.mean().item():.1%} (상위 8% 규칙)")

# ---------- 9. 새 그림 판단하기 · 모델 저장 ----------
def predict(model, feature_256):
    x = torch.tensor(np.asarray(feature_256, np.float32)[None], device=device)
    with torch.no_grad():
        p = F.softmax(model(x), 1)[0].cpu().numpy()
    return {c: round(float(v), 3) for c, v in zip(CLASSES, p)}

best = trained[FOCUS][0] if FOCUS in trained else trained["MLP"][0]
print("테스트 첫 그림 판단:", predict(best, X[te[0]]), "| 정답:", CLASSES[Y[te[0]]])
torch.save(best.state_dict(), "fly_model.pt"); print("저장: fly_model.pt")

# ---------- 더 해 보기 ----------
# ## 더 해 보기
# - `MLP(h=8)`과 `MLP(h=512)`의 검증 정확도를 비교해 과적합을 관찰한다.
# - `FlyRandomProjection(top=0.5)`로 희소성을 없애면 무엇이 달라지나?
# - `MemoryInnate`의 `innate`에 값을 넣어(예: `mi.innate[:] = torch.tensor([0., 0., 2.])`) 본능이 판단을 어떻게 끌어당기는지 본다.
# - `ConnectomeLIF`의 `W_rec` 부호를 모두 +로 바꾸면(억제 제거) 발화가 어떻게 되는가? 마인크래프트 모드의 "이득 0.75 폭주"와 비교한다.
