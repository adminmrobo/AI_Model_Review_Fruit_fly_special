# 딥러닝 A to Z · 글 안상선 · CC BY-NC 4.0 (비상업적 이용, 출처 표기 필수)
# ===============================================================
# 특별편 2 기억 인출 논문 도해 · 코드 1: 기억 + 본능 합류 모형을 넘파이로
#  - 라이브러리: numpy (Colab 기본 설치)
#  - 실행 확인: 2026-10-04, numpy 2.4 (CPU)
#  - 하는 일:
#      ① 측각(본능)은 고정, 버섯체 출력(기억)만 바뀌는 회로를 만든다
#      ② 냄새 A에 전기충격(처벌)을 주는 훈련을 5번 한다
#      ③ 합류 뉴런의 기억 입력을 막으면 본능만 남는지 확인한다
# ===============================================================
import numpy as np
rng = np.random.default_rng(0)
N_IN, N_KC = 256, 1000
ACTIONS = ["다가감", "피함"]

# ---------- 1) 두 갈래 회로 만들기 ----------
W_kc = np.zeros((N_KC, N_IN))                         # 케니언 세포 입력: 고정 무작위
for i in range(N_KC):
    W_kc[i, rng.choice(N_IN, 10, replace=False)] = 1.0
def kc_code(x):
    a = W_kc @ x; return (a >= max(np.sort(a)[-80], 1e-6)).astype(float)

innate = np.array([0.5, 0.0])        # 측각: 냄새에 상관없이 조금 다가가려는 타고난 성향 (진화가 정함)
W_mb = np.zeros((2, N_KC))           # 기억 출력 뉴런: 경험으로 바뀌는 유일한 가중치

# ---------- 2) 합류 뉴런과 도파민 규칙 ----------
def decide(x, retrieve=True):
    memory = W_mb @ kc_code(x)
    z = 1.0 * innate + (0.6 * memory if retrieve else 0)   # 합류 뉴런 = 본능 + 기억
    return z

def dopamine(x, action, outcome, lr=0.2):
    """세 요소 규칙: 켜진 케니언 세포 × 고른 행동 × 도파민(결과 +1 보상 / −1 처벌)"""
    W_mb[action] += lr * outcome * kc_code(x)

# ---------- 3) 훈련 전후와 기억 입력 차단 ----------
# [내 데이터로 바꾸려면] odor_A(냄새 패턴)와 훈련 횟수, innate 값을 바꾼다.
odor_A = (rng.random(N_IN) < 0.2).astype(float)       # 전기충격과 짝지을 냄새
print("훈련 전:", ACTIONS[decide(odor_A).argmax()])
for _ in range(5):
    dopamine(odor_A, action=0, outcome=-1)            # 다가갔더니 전기충격 → 처벌
print("훈련 후:", ACTIONS[decide(odor_A).argmax()])
print("합류 뉴런의 기억 입력 차단:", ACTIONS[decide(odor_A, retrieve=False).argmax()], "(본능만 남음)")
