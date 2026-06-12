# 最强可玩策略（目标线①交付物）

> 8×8 Block Blast 上目前**得分最高的、训练出来的**策略。不靠预知未来。

## 它是什么

**价值引导前瞻（value-guided lookahead）= 训练好的价值网络 V + 前瞻搜索**（AlphaZero 思路）：
- 每回合 `beam_hand` 枚举 3 块的联合落点候选；
- 对每个候选末态做 `S` 条 `D` 手的蒙特卡洛 rollout（启发式 greedy），叶子接**训练好的 V** 当低方差尾值；
- 选 `hand_score + mean_S[rollout + V(leaf)]` 最高者。

价值网络 `models/strongest_v.pt`：8×8 CNN afterstate 价值函数，经
**mode-T Fitted Value Iteration + on-policy 策略迭代**训练（γ=0.98 当训练稳定器）。

## 实战得分（M=200，同一批 CRN 发牌流，T=50，均不预知未来）

| 策略 | 平均总分 | 存活轮 | 说明 |
|---|---|---|---|
| **本策略（vla D2 S30 + V）** | **~2950** | 41 | 训练 + 搜索 |
| 纯前瞻搜索（无训练） | 2842 | 42 | look D3 S20 |
| 纯训练价值贪心（greedy-on-V） | 2353 | — | = 打平 strong |
| strong（手内 beam，旧最强） | 2364 | 42 | 启发式搜索 |
| 早期未修的 RL | 1888 | 33 | 早死（已修） |

→ 比旧最强 strong **+25%**，比早期 RL **+56%**。

**为什么早期 RL 弱、怎么修的**：早期纯价值贪心(1888)在训练没覆盖的局面上高估→早死。
**on-policy 策略迭代**（反复"用当前策略玩→收集途经局面→重训 V"）让 V 在策略真正会遇到的
局面上变准 → 纯贪心 1888→2353 打平 strong；再接前瞻搜索 → 2950。

## 怎么用

> 需在仓库根目录、用 `.venv` 的 python 跑。Windows 设 `PYTHONIOENCODING=utf-8`。

**看它玩一整局（逐回合显示落点 + 棋盘 + 累计分）：**
```bash
.venv/Scripts/python rl8.py play 0          # seed=0
.venv/Scripts/python rl8.py play 7 40       # seed=7, S=40（更强更慢）
```

**单步推荐（代码内调用）：** `best_move(net, engine, board, combo, hand, k)` 返回
`{board, combo, hand_score, placements:[{piece_id,row,col,cells}], value}`：
```python
import rl8, torch
rl8.T_MAX = 50; rl8.GAMMA = 0.98
eng = rl8.Engine8()
net = rl8.RateNet(hidden=128).to(rl8.DEVICE)
net.load_state_dict(torch.load("models/strongest_v.pt", map_location=rl8.DEVICE))
# board: 8×8 bitboard(bit r*8+c)；combo: 当前连击；hand: [piece_id×3]；k: 剩余轮
mv = rl8.best_move(net, eng, board=0, combo=0, hand=[6, 30, 30], k=50)
for p in mv["placements"]:
    print(f"放 piece {p['piece_id']} 于 (行{p['row']},列{p['col']})")
```

**复跑得分对比：**
```bash
.venv/Scripts/python rl8.py vbench rate 50 200 pi    # 用本模型跑 M=200 擂台
python bench_players.py 200 50                        # 纯搜索玩家对比（torch-free）
```

## 参数与权衡
- `D`（rollout 深度）=2 最佳；D≥3 反而略降（叶子 V 尾值已补远期，深 rollout 引入噪声）。
- `S`（rollout 采样数）：S20→S30 仍涨，S30→S40 收益递减（~2950 封顶）。S 是速度/分数旋钮。
- `B`（beam 候选数）=12 足够，加大无明显收益。
- piece_id 目录见 `pieces.py` / `memory/game_rules.md`（38 种方块）。

## 诚实边界
- 不预知未来；rollout 用随机未来牌（蒙特卡洛），不是作弊前瞻。
- "最高分"= 在本评测口径（均匀发牌、T=50、上述计分）下、所测玩家集合里最高；非数学最优（8×8 不可精确解）。
- 模型设备无关：有 CUDA 用 GPU，否则 CPU（慢）。
