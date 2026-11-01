# 딥러닝 A to Z · 글 안상선 · CC BY-NC 4.0 (비상업적 이용, 출처 표기 필수)
# ===============================================================
# 특별편 3 마인크래프트 속 초파리 뇌 · 코드 1: 배선도로 고정된 누설 적분 발화(LIF) 신경망을 넘파이로
#  - 라이브러리: numpy (Colab 기본 설치)
#  - 실행 확인: 2026-10-04, numpy 2.4 (CPU)
#  - 하는 일:
#      ① 시냅스 수 × 0.275 mV × 이득 0.65로 고정 가중치를 만든다(부호는 신경전달물질)
#      ② 감각 뉴런 20개를 120 Hz로 자극하고 0.5 ms씩 100번(게임 1틱) 계산한다
#      ③ 발화한 뉴런 수와 평균 발화 빈도를 센다 — 학습하는 가중치는 0개
# ===============================================================
import numpy as np
rng = np.random.default_rng(0)
# ---------- 1) 배선도로 고정 가중치 만들기 ----------
# [내 데이터로 바꾸려면] syn(시냅스 수 행렬)을 실제 커넥톰 파일에서 읽어 온다.
N = 500                                              # 실제 모드는 176,422개
syn = rng.poisson(0.02, (N, N)) * rng.integers(20, 150, (N, N))   # 시냅스 수 (실제로는 커넥톰 파일에서 읽음)
sign = np.where(rng.random(N) < 0.66, 1.0, -1.0)     # 신경전달물질로 부호 결정 (데일의 법칙)
W = syn * sign[:, None] * 0.275 * 0.65               # mV = 시냅스 수 × 0.275 × 이득 0.65 — 고정

# ---------- 2) 0.5 ms씩 100번 계산 (게임 1틱 = 50 ms) ----------
dt, tau_m, tau_s = 0.5, 20.0, 5.0                    # ms
v = np.full(N, -52.0); g = np.zeros(N); refr = np.zeros(N); spiked = np.zeros(N, bool)
sensory = np.arange(20); rate_hz = 120               # 감각 뉴런 20개를 120 Hz로 자극
count = np.zeros(N)
for step in range(100):                              # 0.5 ms × 100 = 게임 1틱(50 ms)
    poisson = np.zeros(N, bool); poisson[sensory] = rng.random(20) < rate_hz * dt / 1000
    g += W.T @ (spiked | poisson)                    # 이전 스텝에 쏜 뉴런이 연결을 따라 전달
    v = np.where(refr > 0, -52.0, v + dt / tau_m * ((-52.0 - v) + g))
    g *= np.exp(-dt / tau_s)
    spiked = v > -45.0
    v[spiked] = -52.0; refr = np.where(spiked, 2.2, refr - dt); count += spiked

# ---------- 3) 발화 빈도 세기 ----------
rates = count / 0.05                                 # Hz
print(f"발화한 뉴런 {int((count > 0).sum())}개 · 평균 {rates.mean():.1f} Hz · 학습된 가중치 0개")
# 실제 모드는 여기서 하강 뉴런 무리의 발화 빈도를 "우선순위 사다리"(사람이 만든 규칙)로 행동에 옮긴다
