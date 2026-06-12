# block-blast-sim 记忆索引

> 8×8 Block Blast 类游戏。两条线：①最强可玩策略 ②运气/技能分解。按需查，不预读。
>
> **▸ 当前焦点（2026-06-06 末态）= 目标线① 最强策略已交付 + "2950 墙"已被重框**。链路：RL 价值贪心线收口
> （RL≪strong，诊断=OOD高估早死，稳健非池假象）→ on-policy 策略迭代修好（greedy-on-V 1888→2353 打平 strong）
> → **价值引导前瞻（训练V+搜索）= 交付 2924±49 @T=50**（比 strong 2364 +25%；models/strongest_v.pt）。
> 续14：CEM-Dellacherie 落地(cem.py)→ CEM-beam 2458≈strong（独立强基线）；CEM-vla 2918 ≈ CNN-vla 2924（TIE）。
> **续16：唯一候选破墙杠杆（强 beam rollout 基）实测无效**（hbeam 同-S d=+12 打平）→ 搜索族内旋钮穷尽。
> **★★续17/18 重框（最新权威）："2950 分低" = benchmark 定义问题，非策略弱**：T=50 封顶截断 62% 的局
> （旧"surv≈42"=删失均值非自然死）；取消封顶(T=500) surv 42→85、分 2950→**5954**（endless_survival.json）；
> 均匀 38 型发牌过硬是第二主因（仅去 3×3 一项 surv×2.7，deal_audit.json）；计分公式不是因
> （real_approx/assumed=0.71× 同量级，calibrate_scoring.json）。旧"~2950 结构性天花板"的**绝对数解读已推翻**
> （T=50 内的相对配对结论仍立）。详见 log.md 续16–18 + [strongest_policy.md](strongest_policy.md) 顶部修正块。
> **▸ 续19（2026-06-12）✅ 目标线② 收口 verdict(i) 命中 + 连击豪赌计算器**（`evpi_vla.py`/`combo_gamble.py`，详见 log.md 续19 + `../EVPI_VLA_RESULTS.md`，**未提交**）：
> 用 vla(2924) 当 EVPI 占比新分母在冻结 cohort4 重算 ⇒ 占比 **69/65/57% → 109/106/96%（≈100%）**（strong-分母 cohort3 逐点复现 channelB=自洽 sanity；抬升来自换分母非 cohort）。含义=面对最强可玩"无真未来"策略，到 oracle 残差几乎全是不可约运气，技能通道被 vla 吃满；**>100% 是信号(vla>blind=更强边缘化器)非 bug**。诚实边界：cohort4 小(39/26/20)→CI 宽跨100%，方向性结论稳/点值≈100%带宽 CI。连击计算器：**combo 占 strong 总分 53%**(超线性=运气引擎) 但维持概率随层级 0.52→0.16 速降 ⇒ **常数-p 高估深连击赌局 4×、追深连击是亏本押注**(价值来自大量短链)。
> **▸ 仍未结**：② 真实化 benchmark（endless + real_approx 计分 + 善意发牌）下重测 luck/skill；③ **大量产物未提交**
> （续14 起：cem.py / models/ / STRONGEST_POLICY.md / NEXT_STEP_* / endless·deal·calib + 续19 evpi_vla/combo_gamble 等，commit 先问用户）。

- **[../NEXT_STEP_beam_rollout.md](../NEXT_STEP_beam_rollout.md)（根）✅ 已执行（续16，2026-06-06，负结果：强 beam 基救不了 → 结构墙第二次坐实）— 留档**
- **[../NEXT_STEP_endless_survival.md](../NEXT_STEP_endless_survival.md)（根）✅ 已执行（续18，2026-06-06，Step A/B/C 全跑 → 重框 2950）— 留档**
- **[../GOAL.md](../GOAL.md)（根）— 权威双目标：①最强可玩策略(②a) + ②运气/技能分解；定位+成功标准**
- **[../ROADMAP.md](../ROADMAP.md)（根）— ②a/②b 阶段路线，②a 先；[../PLAN_8x8_RL.md] 是 /loop 可执行计划**
- [project_goal.md](project_goal.md) — 早期目标+私有 repo(数字部分 legacy,以 GOAL.md 为准；78% 已撤回)
- [game_rules.md](game_rules.md) — 游戏规则定义 + 38 种方块目录（均匀假设）
- [luck_vs_skill_results.md](luck_vs_skill_results.md) — 新手技能主导(地板113×)/高手~7成运气；线性版CV=91%已legacy,现行见oracle两通道
- [strategy_design.md](strategy_design.md) — strong 策略怎么做的、是不是最优、通往最优的路线图
- [oracle_immortality_reframe.md](oracle_immortality_reframe.md) — seer近乎不死→78%口径坏掉→固定T+两通道(hazard/EVPI)重设计+六大陷阱+可靠性修正
- [rl_plan.md](rl_plan.md) — RL afterstate值迭代当"在线天花板压力测试"，三轮审核定稿(γvs固定-T/双gate/combo变换)
- [rl4_gate_results.md](rl4_gate_results.md) — 方向①双gate皆PASS(认证4×4 FVI,放行8×8)+winner配置+5条工程教训(高估偏置随网宽降/mode-T早停)
- [rl8_phase1.md](rl8_phase1.md) — 8×8 RL Phase1：§3.0 gate PASS；变换线终局=logv/rate 在无折扣 T=50 都发散→γ=0.95 当训练稳定器（判强仍用无折扣实战分）；7 条工程教训(★×k放大陷阱/前向闭包防崩塌/MPS 10×/冻结目标/exact_vtot_anchor 真值锚/高-k 自洽)
- **[strongest_policy.md](strongest_policy.md) ★当前焦点 — 价值引导前瞻交付(~2950)+怎么炼成(on-policy PI 修早死)+多角度评测+2048/Tetris/AlphaZero 更强方法调研(n-tuple/expectimax/CEM)+下一步**
- [log.md](log.md) — 时间线流水

## 文件地图
- `pieces.py` — 方块目录（旋转生成 + 去重）
- `sim.py` — 引擎 + 三策略(random/greedy/strong) + 同种子双因素方差分解
- `RESULTS.md` — 结论速览
