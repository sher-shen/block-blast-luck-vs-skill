# block-blast-sim 记忆索引

> 8×8 Block Blast 类游戏的"运气 vs 技能"量化项目。按需查，不预读。

- **[../GOAL.md](../GOAL.md)（根）— 权威双目标：①最强可玩策略(②a) + ②运气/技能分解；定位+成功标准**
- **[../ROADMAP.md](../ROADMAP.md)（根）— ②a/②b 阶段路线，②a 先；[../PLAN_8x8_RL.md] 是 /loop 可执行计划**
- [project_goal.md](project_goal.md) — 早期目标+私有 repo(数字部分 legacy,以 GOAL.md 为准；78% 已撤回)
- [game_rules.md](game_rules.md) — 游戏规则定义 + 38 种方块目录（均匀假设）
- [luck_vs_skill_results.md](luck_vs_skill_results.md) — 新手技能主导(地板113×)/高手~7成运气；线性版CV=91%已legacy,现行见oracle两通道
- [strategy_design.md](strategy_design.md) — strong 策略怎么做的、是不是最优、通往最优的路线图
- [oracle_immortality_reframe.md](oracle_immortality_reframe.md) — seer近乎不死→78%口径坏掉→固定T+两通道(hazard/EVPI)重设计+六大陷阱+可靠性修正
- [rl_plan.md](rl_plan.md) — RL afterstate值迭代当"在线天花板压力测试"，三轮审核定稿(γvs固定-T/双gate/combo变换)
- [rl4_gate_results.md](rl4_gate_results.md) — 方向①双gate皆PASS(认证4×4 FVI,放行8×8)+winner配置+5条工程教训(高估偏置随网宽降/mode-T早停)
- [rl8_phase1.md](rl8_phase1.md) — 8×8 RL Phase1：§3.0 gate PASS/§3.6 变换未定盘；6 条工程教训(★×k放大陷阱/前向闭包防崩塌/MPS 10×/冻结目标/exact_vtot_anchor 真值锚)
- [log.md](log.md) — 时间线流水

## 文件地图
- `pieces.py` — 方块目录（旋转生成 + 去重）
- `sim.py` — 引擎 + 三策略(random/greedy/strong) + 同种子双因素方差分解
- `RESULTS.md` — 结论速览
