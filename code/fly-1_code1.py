# 딥러닝 A to Z · 글 안상선 · CC BY-NC 4.0 (비상업적 이용, 출처 표기 필수)
# ===============================================================
# 특별편 1 초파리 뇌 탐사 · 코드 1: 초파리 방식(고정 무작위 확장 + 출력층만 학습)을 넘파이로
#  - 라이브러리: numpy (Colab 기본 설치)
#  - 실행 확인: 2026-10-04, numpy 2.4 (CPU)
#  - 하는 일:
#      ① 케니언 세포 1,000개가 입력 10개씩 무작위로 받는 고정 연결을 만든다
#      ② 상위 8%만 켜는 희소 코드를 만든다
#      ③ 출력층 하나만 델타 규칙으로 학습하고, 학습·고정 연결 수를 센다
# ===============================================================
import numpy as np
rng = np.random.default_rng(0)
N_IN, N_KC, N_OUT = 256, 1000, 3          # 투사 뉴런(입력) · 케니언 세포 · 출력(MBON)
K_IN, TOP = 10, 0.08                      # 케니언 세포 하나가 받는 입력 수 · 켜지는 비율

# ---------- 1) 고정 연결 만들기 ----------
# ① 연결은 태어날 때 무작위로 정해지고 학습하지 않는다
W_kc = np.zeros((N_KC, N_IN))
for i in range(N_KC):
    W_kc[i, rng.choice(N_IN, K_IN, replace=False)] = 1.0

def kc_code(x):
    a = W_kc @ x                          # 무작위로 골라 더하기
    thr = np.sort(a)[-int(N_KC * TOP)]    # 상위 8%만 남기고 끄기 (APL 억제 뉴런)
    return (a >= max(thr, 1e-6)).astype(float)

def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()

# ---------- 2) 출력층만 학습하기 ----------
# ② 바뀌는 것은 출력층 하나뿐: 오차 × 입력 (델타 규칙, 역전파 없음)
W_out = np.zeros((N_OUT, N_KC))
def train(X, Y, epochs=20, lr=0.05):
    global W_out
    for _ in range(epochs):
        for x, y in zip(X, Y):
            k = kc_code(x)
            p = softmax(W_out @ k)
            W_out += lr * np.outer(np.eye(N_OUT)[y] - p, k)

# ---------- 3) 장난감 데이터로 학습하고 세기 ----------
# [내 데이터로 바꾸려면] X(샘플 × 256)와 Y(정답 번호)를 내 데이터로 바꾼다.
X = rng.random((60, N_IN)) * (rng.random((60, N_IN)) < 0.2); Y = np.arange(60) % 3
X[Y == 0, :80] += 0.8; X[Y == 1, 80:160] += 0.8; X[Y == 2, 160:] += 0.8   # 장난감 데이터
train(X, Y)
acc = np.mean([softmax(W_out @ kc_code(x)).argmax() == y for x, y in zip(X, Y)])
print(f"훈련 정확도 {acc:.2f} · 학습된 가중치 {W_out.size:,}개 · 고정 연결 {int(W_kc.sum()):,}개")
