# Ant: 처음 보는 지형에서 걷기

Isaac-Ant-v0로 학습한 Ant가 학습 때 보지 못한 지형에서도 걷도록 만든 과제 코드입니다.

## 설치 방법

준비물은 아래와 같습니다.
[IsaacLab_RS](https://github.com/cailab-hy/IsaacLab_RS)가 설치되어 있고, 그 conda 환경에서 Isaac-Ant-v0 학습과 재생이 되는 상태면 됩니다.

**1. 이 저장소를 받습니다.** 위치는 어디든 괜찮습니다.

```bash
git clone https://github.com/brownporkcutlet/robotic_simulation_assignment_ant_rough_walk.git
```

**2. 받은 폴더 3개를 IsaacLab_RS 루트에 복사합니다.** `<IsaacLab_RS 경로>`는 사용하시는 경로로 바꿔 주세요 (예: `~/IsaacLab_RS`).

```bash
cp -r robotic_simulation_assignment_ant_rough_walk/source \
      robotic_simulation_assignment_ant_rough_walk/scripts \
      robotic_simulation_assignment_ant_rough_walk/checkpoints  <IsaacLab_RS 경로>/
```

복사되는 파일은 아래와 같습니다. 모두 새 파일이라 IsaacLab_RS의 기존 파일은 덮어쓰지 않습니다.

| 경로 | 내용 |
|---|---|
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_rough/` | 학습 환경과 PPO 설정 |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_unseen/` | 시험 지형 |
| `scripts/reinforcement_learning/rsl_rl/eval_unseen.py` | 평가 스크립트 |
| `checkpoints/ant_walk_final.pt` | 학습된 최종 정책 |

**3. 등록을 확인합니다.** 추가 설치 명령은 필요하지 않습니다. 아래 명령을 실행했을 때 태스크 6개가 보이면 정상입니다.

```bash
cd <IsaacLab_RS 경로>
python scripts/environments/list_envs.py | grep -E "Ant-Rough|Ant-Unseen"
```

```
Isaac-Ant-Rough-Walk-v0        학습용
Isaac-Ant-Unseen-Flat-v0       시험 지형
Isaac-Ant-Unseen-Waves-v0
Isaac-Ant-Unseen-Boxes-v0
Isaac-Ant-Unseen-Stairs-v0
Isaac-Ant-Unseen-Mixed-v0
```

재생, 평가, 학습 커맨드는 [eval_commands.txt](eval_commands.txt)에 있습니다.
