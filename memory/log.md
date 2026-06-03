# block-blast-sim 日志

## 2026-05-31 项目启动
- 起因：用户玩 Block Blast 类游戏，疑惑"运气 vs 技能各占几成"。
- 调研：确认是单人随机生存 MDP（≠五子棋）；Tetris NP-hard + Burgiel 杀手序列。
- 建模：pieces.py 生成 38 种方块目录（含朝向，均匀等概率）。
- 引擎：sim.py 8×8 规则 + random/greedy/strong 三策略 + 同种子双因素方差分解。
- 结果(300局)：random 0.9 / greedy 17.4 / strong 102.9；熟练间运气≥技能，strong CV=91%。
- 结论：新手技能主导(100×)，高手运气主导(~7成)。写入 RESULTS.md + memory/。
- 厘清：strong 是启发式非最优；通往最优靠 4×4 精确DP基准 + 技能阶梯收敛 + 离线上界。
- 下一步候选：(A)换真实方块集/概率 (B)加MCTS/RL看运气占比收敛 (C)分数分布图 (D)4×4 DP基准

## 2026-05-31 (续) 接入非线性计分
- 用户补充真实计分：multi-clear(同时消多行)、combo(连续消除连击)、all-clear(清盘) 都有奖励。
- 新增 scoring.py（可调参数：line_base=10/三角数翻倍, combo_unit=50, all_clear=300）。
- 重写 sim.py：连击状态贯穿整局，策略改为追逐真实得分。
- 300局结果：random58/greedy668/strong3888；熟练间 技能31%/运气35%/交互34%；CV85%。
- 发现：奖励小幅抬高技能(29→31%)，但 strong 只一手贪心、无多手前瞻，吃不到 combo/清盘的规划红利
  → 真实技能天花板更高，**需 MCTS/前瞻才看得见**。交互项(34%)是"会玩才兑得出的机会"。
- 下一步定调：上 MCTS 前瞻玩家，预期非线性计分下技能占比明显上升。

## 2026-05-31 (续2) 执行前瞻玩家 + 独立 agent 审核
- 先开独立 agent 审核方案，得 GO-WITH-CHANGES，7 条改动全采纳(combo贯穿/λ=1/bitboard/beam/CRN/独立RNG/收敛非循环)。
- 新建 fast.py(bitboard 64位int引擎 + beam-strong + lookahead) / compare.py(配对CRN对比) / run_lookahead.py(阶梯+oracle)。
- 踩坑并修：(1)初版 rollout 每候选用不同未来牌→违反CRN→排序噪声大，改为同批未来牌配对；
  (2)种子太少(8-12)且 strong/LA 用不同种子集→均值纯噪声→改配对同种子。
- **K=20 配对结论**：LA_greedy−strong=−3640±1717(greedy基更差,rollout回归);
  LA_strongbase−LA_greedy=+2427±1007(换strong基回血,证实基策略须≥被改进者);
  LA_strongbase−strong=−1214±1814(打平→**收敛**)。
- 最终结论：beam-strong 已近此启发式族天花板；天花板被极少数爆炸局主导(seed8 strong=33315);
  oracle(真未来牌)~12000 vs strong~6100，缺口≈一半=信息缺失代价=运气量级。
- 三方印证"高端运气主导"：搜索收敛 + 天花板巨大方差(CV~120%) + oracle缺口约一半。
- 真天花板需 RL 价值函数(非更深搜索)，列为可选下一步。

## 2026-05-31 (续3) 做成创新小项目 + 推送 GitHub
- 用户定：做成有创新性的小项目，搜现有研究找空缺，推 GitHub，记目标到 memory。
- 搜网确认空缺：现有全是 solver/RL-agent("怎么玩好")，无人做"运气-技能定量分解"；
  skill-vs-luck 文献几乎全多人博弈。理论锚点 Aleph Insights "luck=lack of information"。
- 创新主线(用户选)：**运气=信息价值(oracle 缺口/value of hindsight)**。
- 新增：experiments.py(一键 results.json) / plots.py(零依赖 SVG) / figures / README(writeup) / LICENSE / .gitignore。
- 最终(200/24 种子)：地板113×；ANOVA技能34/运气34/交互32；交叉点≈3408；
  收敛 LA−strong=−1619±1545；oracle≈27475 vs strong≈6066 → 缺口≈78%。
- **私有 repo: https://github.com/sher-shen/block-blast-luck-vs-skill**（git 身份 sher-shen / 用户邮箱）。
- memory/ 已 .gitignore，不进 repo。

## 2026-05-31 (续4) 严谨化 oracle 缺口 → 重大发现 + 三轮审核重设计
- 优先级建议(2 个独立审核 agent 都同意)：**方向4(缺口严谨化)>3(4×4 DP)>2(更多种子)>1(RL)**。
  理由：78% 是 headline 唯一新贡献，却是【对一个 D=3 greedy 偷看"oracle"算的 ratio-of-means】，
  最该补可信度；RL 是瞄准天花板，得先把天花板量准。估计量先修、种子后加、4×4 DP 早做当校准锚。
- 新建 oracle_analysis.py：把 oracle 换成 beam-rollout + 真实未来，做 per-seed 配对 + bootstrap CI + 分解。
- **重大发现**：beam-rollout seer 看到真实未来后【近乎不死】，无界 horizon 下分数爆炸到 1e5–1e6
  (连击链复利)。→ 旧"运气=seer−strong 分差"被存活长度主导、本质无界，**78% 口径坏掉**(不只长尾)。
- **重设计(用户要求≥3 轮审核)**：固定 horizon T，运气拆两通道——
  (A) 存活运气=Burgiel 杀手序列【每轮 hazard】(seer 死亡率=真最优 hazard 上界)；
  (B) 计分运气=EVPI=seer−blind，在【四玩家都活到 T 的 cohort】上算(否则死亡冻结分会把存活漏进计分)。
  4 玩家：strong(地板)/blind(采样未来均值=无信息 Bayes 臂)/seer(真未来=完美信息臂)/anti(单条假未来=诊断)。
- 三轮审核结论(全过)：
  R1 抓到 blind 可能 < strong(rollout 回归)→ procedure_cost 可为负，份额会 >100%；
  R2 抓到死亡冻结分再污染计分→改 cohort 条件化；clip 掩盖 procedure 符号→改带符号堆叠分解，
     headline 用 seer−blind(非 seer−anti，anti 是"自信地猜错"会高估)；加 S-stability + 多 T；
  R3(执行门)抓到 cohort 选择偏差(只留 strong 能活的易种子→份额是运气【下界】)；raw=EVPI+procedure
     是代数恒等(报"如何切分"非"发现恒等")；EVPI 两端都有偏→headline 说"≈"，靠 S/D 趋平背书；
     4×4 减子集 DP 是"可解类比"非 8×8 校准。**APPROVE，可执行**。
- 已执行并推送(commit 900d63b)：oracle_analysis.py 两通道全跑完，README 撤回78%改两通道+fig3/4，push。
  最终数：存活 hazard≤7e-5/轮；EVPI 943[816,1082]@T40 / 1427[1221,1662]@T60；信息占比 57–69%；S=8 够，D=3 plateau。

## 2026-05-31 (续5) 方向3：4×4 精确 DP 锚点（三轮审核 APPROVE + 已推送）
- 用户选做方向3。三轮独立审核(全 APPROVE/APPROVE-WITH-CHANGES)关键修正：
  R1: 3块手牌 exact-DP 在 k=32 不可行且 16格近退化 → 改 1块/回合(状态=棋盘2^16)；玩家诚实叫"greedy"非beam；
      VOI 要 per-seq 配对+CI(非 offline_mean−V0)；framing=启发式标定非EVPI迁移。
  R2(BLOCKING): monomino 永远放得下 → 无折扣 SSP 的 V* 可能【发散】(a.s.终止≠期望有限,正是近immortal现象)。
      改 **γ-折扣(γ=0.95)**，γ-收缩保证有限 V*。VOI 改 discounted。
  R3(放行): 锁折扣时序(γ^t 三处一致)；offline/online 同截断 L(γ^L·maxR/(1−γ)<ε)；硬断言 mean(online)=V*(empty)。
- 新建 dp4.py(engine4 4×4 + reachable BFS=41503 + γ-VI + 离线DP + greedy + bootstrap)。
- **最终数(M=200,γ=0.95,断言✓)**：V*(empty)=3.91；**近视贪心达最优≈86%**(缺口0.82[0.37,1.27])→启发式在线近最优；
  **online 只兑现上帝视角25%**，**discounted VOI=12.6[12.0,13.3]**(≈在线3倍)→信息价值主导。独立佐证8×8主线。
- README 加 §⑤ + fig5；push。**4 个方向收口**：4(严谨化)✓ 3(4×4DP)✓ 2(N=120已足)✓ 1(RL)仍最后/未做。
- 关键工程教训见 [[oracle_immortality_reframe]] 第6条 + 新：4×4 1块/回合 monomino 致无折扣VI发散→必须γ折扣。

## 2026-05-31 (续6) 方向1 RL：优先级判断 + 设计 + 三轮审核(全过)
- 优先级判断：RL **GO 但须 reframe**——不是造 bot(=solver 框架/弱agent陷阱)，而是**EVPI 占比载重假设"strong=在线天花板"的压力测试**(收敛三脚架第3腿)。
- 设计：**afterstate 拟合值迭代**(非DQN/PPO)，学 V(末手棋盘,combo,rounds-left)，argmax `hand_score+V`(目标不含 heuristic，但 warm-start 用它)。详见 [[rl_plan]]。
- 三轮审核(每轮换新 agent、轮间改)：
  R1(GO-WITH-CHANGES)：Part A 抓通道A hazard 只基于1死(须 Poisson 95%UB=3.3e-4)、EVPI 占比缺 CI/2点线段/sstab 各 S 不同 cohort、**dp4 assert 用 3·SE 太松疑似藏偏差**。
  R2(GO-WITH-CHANGES，跑码验证)：**证伪 dp4 系统偏差**(32k种子缺口−0.0096，Round1 被单窗骗)→V*=3.9157 可信；
    抓**最严重新问题**：γ=0.95(RL) vs 固定-T(channelB) **apples-to-oranges**→8×8 须用无折扣有限-T 值(条件 rounds-left)，γ 只留给 4×4 dp4 gate。
  R3(APPROVE-WITH-CHANGES，执行门)：**4×4 γ-gate 只认证折扣机器，不认证无折扣-T 目标**→须加**第二个 4×4 微 gate**(无折扣有限-T 后向DP)；
    combo 无界奖励→目标须 log1p/Huber；warm-start 须配"V 已离开 init"位移检查；预算须含 ~T× 样本膨胀；RL>strong 重算占比须用**冻结 intersection cohort**。
- 三轮全过(无 BLOCK)。Part A 修正(改已发布结论)与 Part B(RL pipeline)的最优执行链：先修 dp4 收敛V*+Part A → 建 4×4 双 gate harness → 才 8×8 训练。

## 2026-05-31 (续7) 用户拍板"按推荐执行"→ 第4轮审核 + Part A 全部落地
- 用户选 Option3(Part A 修正 + 4×4 双gate)，要求"推荐后再开1个 agent 审查才执行"→ 第4轮审核(APPROVE-WITH-CHANGES)。
  审核抓: (a) 我 spec 的 Poisson 常数 2.3724 错(应 4.7439=0.5·χ²_{0.95}(4) 单侧)；(b) 误把 README "86%" 当陈旧(其实**反了**)。
  我自验: 单侧95% UB=**3.30e-4**(≤1死/3028轮)，非双侧3.88e-4；greedy ratio fresh=**0.859**=匹配README。
- **Part A 全部执行 + 验证(已就绪待commit)**：
  - survival hazard: 撤回误导的"≤7e-5 上界"→ 点估计 6.96e-5 + **单侧95% Poisson UB 3.30e-4(≤1死/3028轮)** + 独立"seer≤真最优"建模界。survival.json/README/fig3 同步。
  - channelB N=120 D=3: **精确复现** baseline(T40 EVPI 943[816,1082] 69%; T60 1427[1221,1662] 57%)+ 新增**份额bootstrap CI**(69%[53,73]/65%[54,71]/57%[54,74])+ **第3个T=50**(EVPI1180 65%)。
  - s_stability: 修 per-S-cohort 假象 → **固定 intersection cohort=24**(≥20✓)，EVPI across S=1085/980/1055/1065 ~±10% 真平。
  - dp4: 加 backward_dp_T(无折扣有限-T 真值,T=1=0.0625=2/32✓) + run_blocks(收敛V*=3.9157, online block-mean gap **−0.0009** 证无系统偏差) + γ docstring 修。
  - README 加"份额分母载重于 strong=天花板,更强玩家会抬升占比"caveat(=方向①压力测试目标)。
- **dp4 "0.78 vs 0.86 vs 5.26" 三值之谜根因 = stale __pycache__**(非代码bug)：编辑 .py 后旧 .pyc 被某次调用加载→幻象不可复现。
  `rm -rf __pycache__` 后一致 0.859/VOI12.6=匹配README。committed dp4.json(0.78)是旧码产物,已重生。
  **教训**：诊断"数值不可复现/疑似非确定"前先清 __pycache__；committed JSON 与现码不符时优先怀疑产物陈旧而非代码。见 [[oracle_immortality_reframe]] 工程节。
- **踩坑**：① 同时用 `nohup &` + 工具 run_in_background = 双重后台,状态混乱→只用其一。② channel/sstab 误用 D=5/N=80(默认值)跑→与 baseline(D=3/N=120)不可比,发现后 kill 重跑。
- 4×4 双gate harness(torch venv) + 8×8 训练 = 下一chunk,未做(等 Part A commit 后)。

## 2026-06-01 续8（规划：4×4 双 gate 执行计划 + 3 轮审核全过）
- 产出 `PHASE_EXEC_PLAN.md`（项目根）：① 4×4 双 gate 可执行方案（torch venv 安装 / rl4.py 网络+FVI+两模式 / 五条 gate 判据 / 执行链）+ ② 长期 (a)8×8 RL (b)胜率可设计性 DDA 研究提纲。本轮不写训练代码。
- **3 轮独立 adversarial 审核全过**（每轮换 agent，轮间改）。抓出并落地的硬修正：
  - R1：.venv→.gitignore 强制前置；T-gate 在线须实现**无折扣 T-rollout**（无 γ/L）+ 预注册等价界；greedy ratio 关 0.86–0.95 limbo；位移检查改条件 gate。
  - R2：greedy ratio 改 **paired-CRN bootstrap 95% CI 下界≥0.92**（非点 ratio-of-means，dir④）；T-在线 M=50k + 精度护栏 2·SE<2.5%·bdp_T；位移 corr 阈 0.9→**0.92**（因实测 corr(heuristic4,V\*)=0.870，0.9 会假阴）；warm-start **仿射映射**到 V\* 量纲（heuristic4∈[−24,−8] 负、V\*∈[0.07,4.56] 正，不能直拟）+ fallback；补 dir⑦ 后台不叠加。
  - R3：清除残留旧阈值（执行链/USER_GATE 里漏改的 ≥0.95 / corr≥0.9）；位移检查 binding 语义写死（主判据 V-单位变化>τ_disp 为 PASS 必需，corr<0.92 为 advisory→USER_GATE 人工核查）；τ_disp 用 V-单位（非 ratio 缺口 0.14，量纲不同）。
- 状态：计划 APPROVED，**等用户放行**才从"装 torch venv + 建 4×4 双 gate"开始执行。

## 2026-06-01 续9（执行：4×4 双 gate harness 全部跑通，双 gate 皆 PASS，停在放行②a）
- 用户 `/loop /bsa...` 风格启动执行对话。装 py3.13 venv + torch 2.12.0（py3.14 无 arm64 wheel，降 3.13）。写 `rl4.py`（afterstate MLP + 向量化 FVI + 无折扣 T-rollout + 仿射 warm-start + 位移检查）。
- **双 gate 皆 PASS（rl4_gate.json）**：
  - γ-gate：值 V_net(empty)=3.9144 Δ=0.0013<0.05 ✓；策略 greedy-on-V_net ratio=1.005 paired-CRN bootstrap CI[0.997,1.013] 下界≥0.92 ✓（碾压近视贪心 0.859）；位移 disp 0.74>τ_disp 0.22 binding✓，corr(V_net,heuristic4)=0.871<0.92 advisory OK（实测 0.870 预测精准命中）。
  - T-gate：值 T=8 rel 2.2% / T=16 rel 3.9% <5% ✓；在线 M=50k 无折扣 rollout T=8 rel 1.6% / T=16 rel 1.5% + 精度护栏 2·SE<2.5%·bdp_T 全过 ✓。
- **关键工程发现（见 [[rl4_gate_results]]）**：
  ① FVI + max 算子 + 函数逼近 = **系统性高估偏置**，且**偏置随网宽递减**：hidden=64 → V(empty) 高估 +3.9%（4.07 vs 3.92，FAIL 紧 gate）；hidden=256 → +0.03%（PASS）。紧的绝对值 gate 必须用够大的网压偏置。
  ② mode-T 偏置容忍松（5% 相对）→ hidden=128 足够且每 sweep 快 ~2.5×；256 跑 mode-T 142 sweep 撞 1800s wall budget 仍未收敛（值还在从高位下降途中），128 跑 102 sweep / 637s 干净 plateau 且 PASS。**网宽是 accuracy↔wall-clock 的旋钮，按 gate 紧度选**。
  ③ warm-start fallback（sweep1 MSE >10×warm末）在大网上"过敏"：256 warm 拟合太好(MSE 0.0096)→10× 太小→误触发冷启；但冷启照样收敛到 3.9144（FVI 对 init 不敏感，PASS 不受影响）。
  ④ mode-T 是网络版后向归纳，值逐层从 k=1 传到 k=16，**早停 plateau 是陷阱**：首版 sweep18 plateau→严重欠收敛(undershoot)；加 min_sweeps=80 解决。
- 踩坑：rl4_gate.json 被 T-only 运行覆盖丢了 gamma 结果 → 改 main **merge 进已有 JSON**。py3.14 torch 无 wheel → 用 py3.13。
- README 加"方向① 4×4 afterstate-FVI 双 gate 认证"节（gate 表 + **不认证项声明**：4×4 单块/回合无 combo；不认证 8×8 beam_hand 三块手牌×afterstate，那层靠 8×8 competence gate）+ rl4.py 文件行 + venv 复现命令。
- **状态：双 gate 皆过 ⇒ 认证 4×4 pipeline，停在"放行②a"等用户确认。本 loop 不碰 8×8 训练。**

## 2026-06-01 续10（系统审查 A–F + 目标升级双目标 + ②a 计划落定，未碰训练）
- 6 维并行 sub-agent 审查：A 新颖性"部分站得住"（最危险对标 arXiv:2402.12874 advantage-based 单智能体分解；通道A支柱 Gehnen FUN2024；方向①先例 2306.14626；DDA 先例 FLAIRS2016）；B 方法严谨——headline 全稳，仅 dp4 两处内部不一致（value_iteration 默认 γ=0.99 vs 调用 0.95；run() 残留单窗 3·SE assert）不污染数字；C 现码现跑 13/13 全绿、JSON 无陈旧、零依赖隔离干净；D 记忆无一超 150 行（"大概率超标"预设不成立），唯一硬错 = project_goal 第41行 78% 残留 + MEMORY 索引 oracle 钩子 155 字超标；E 起草 GOAL/ROADMAP；F 拍板 **②a 先做**（②b 卖点全建立在 headline 站稳上，②a 是其前置）。
- **目标升级（用户对齐）**：从"运气/技能研究"升级为**双目标**——①最强可玩 8×8 策略(②a RL)②运气/技能分解；二者在 ②a 汇合。新建根 `GOAL.md`/`ROADMAP.md`。改 project_goal(撤 78%)、MEMORY 索引(加 GOAL/ROADMAP 指针+压 oracle 钩子)、luck_vs_skill_results/RESULTS 标 legacy(CV=91% 线性版已被两通道取代)。dp4 两小修 + README 三引用 = 排进 PLAN Phase 0。
- **`PLAN_8x8_RL.md`**（②a /loop source-of-truth）落定，经 1 轮独立 adversarial 复审改进：① 加**致命前置迁移 gate**（rl8 的 CNN+采样 FVI 须先在 4×4 复现 bdp_T，rel<5%——4×4 双 gate 是 rl4 的 MLP+全枚举认证的，新架构/采样/损失三换须重认证）；② combo 重尾默认从 Huber-on-raw **改 per-round-rate×k**（Huber 不压 1e4-1e5 值域）+ 微检移到 8×8 held-out（4×4 触发不了重尾）；③ competence 判等价改 **TOST**（"CI 含 0"≠相等是统计错）；④ buffer 覆盖率护栏防 on-policy 漂移 + strong/seer 轨迹 off-policy 补采；⑤ (ii) 三角验证措辞降级——rl/strong 共享 beam 候选+heuristic，仅 value 层正交、不得称"独立"。
- **状态：审查+规划完成，全部落盘（未 commit/push）。等用户新开对话 `/loop` 跑 PLAN_8x8_RL.md。**

## 2026-06-01 续11（独立重审 2026-06-01 产出 + 落实修复，未碰代码逻辑）
- 4 个并行 sub-agent 独立重审续10 产出（AUDIT-1 记忆/2 GOAL-ROADMAP 诚实度/3 PLAN 科学性/4 方向拍板）。结论：记忆+对外口径**通过**（数字与 channelB/rl4_gate/dp4.json 现读精确吻合、78% 已彻底撤回、4 篇对标文献全真实存在）；方向拍板 **②a 先/②b 后正确**；PLAN **需改后可放行**——5 个复审修复全落实，但 AUDIT-3 adversarial 挖出 1 致命运营缺陷 + 3 实现空白。
- **已落实修复（纯文档预注册，未改 .py、未 commit）**：
  ① PLAN §1+§3.7 完成判据：加 **HARD_CEILING=7 天**预注册硬上限，Phase 1 末「单 sweep×min_sweeps×1.5」外推 >7 天 → BLOCKED_BUDGET、不进 Phase 2 USER_GATE（防废算力跑到撞墙）。
  ② PLAN §3.6：写死 **rate↔总值 Bellman 三步**（V_total=rate×(k−1) 还原→max→/k 回归），消除 rate-vs-总值混用 bug。
  ③ PLAN §3.0：写死 **4×4→8×8-CNN 适配**（padding='same' + zero-pad 到 8×8 左上角 + 同一 forward 代码路径、仅重训权重），使迁移 gate 真测同一 pipeline。
  ④ §3.7 位移阈**去循环化**（不得用含 final-V 的 RMS 差，改纯 warm_init 量纲）；§1 Δ_eq 单位写死=T=50 总分 5%；§1 EVAL_SEEDS 加 M 超限 fallback 优先级；§2 校正 2306.14626 引用定位（更贴 ②b DDA，勿当 ②a 先例）。
  ⑤ 记忆：3.9158→**3.9157**（rl_plan/log/rl_gate_spec 共 5 处，dp4.json 实读 V_star_converged=3.91573 为权威）；MEMORY.md 4 行超 150 字符 → 全压到 ≤150；ROADMAP:47"检索不到先例"软化为"据我们检索未见直接先例"+ 标注非对外 headline；GOAL:35"已部分声明"→"已明确声明"。
- **状态：PLAN 已补齐预注册护栏，可放行 /loop 执行（未 commit/push）。**

## 2026-06-01 续12（/loop 跑 PLAN_8x8_RL：Phase 0 DONE + Phase 1 rl8.py 落地，迁移 gate 收敛中）
- **Phase 0 全清（已验证）**：`rm -rf __pycache__`；dp4 两修—`value_iteration` 默认 γ 0.99→0.95（跑 `value_iteration(tol=1e-5)` 复现 V[0]=3.9157 ✓）、`run()` 移除单窗 3·SE assert 输出字段（稳健比较走 run_blocks/block_stats）；README 补 3 对标引用（2402.12874 advantage-分解机制别 / Gehnen FUN2024 通道A支柱 / 2306.14626 标注"agent表现作难度代理"先例，勿当②a强策略先例）+ "空白点"措辞降级（"首次将 EVPI/survival 双通道用于单人随机生存 block-puzzle 并精确 DP 锚定"，不吹"首次用信息价值定义运气"）；torch 2.12.0 OK、分析管线 0 import torch。状态块 PHASE_0_PREFLIGHT=DONE，CURRENT_PHASE=1。
- **Phase 1 新文件 `rl8.py` 落地**（仅 rl8/rl4 依赖 torch）：8×8 小 CNN（2 conv padding=1 全卷积→flatten+concat(combo,k)→MLP→per-round-rate 标量，**140,897 参数**，CPU）；引擎抽象 Engine4(dp4 单块)/Engine8(fast beam_hand 3块)；self-play(+ε) 采样 buffer；mode-T FVI 预计算转移表(rl4.build_transitions 的 buffer 版)+向量化 scatter-amax；per-round-rate 严格三步（V_total=rate×(k−1) 还原→max→/k 回归）。
- **预注册设计决策（§3.1 内部张力定调）**：afterstate-V 条件于 **(board,combo,k)，不含手牌 glyph 平面**——§3.3 权威 Bellman 已由 mean_p 平均掉手牌，与 rl4 mode-T(board+k)+beam_hand(board,combo,score) 粒度一致。写进 rl8.py docstring。
- **迁移 gate 关键工程教训（实测）**：首版只在 self-play buffer 的 src(930) 上回归、但续值在更大的 dst(20072) 上 bootstrap → **值崩塌向"只剩即时分"**（sweep0 Vtot(0,8)=1.08→sweep1 0.79，靶 3.59）。**修复 = 训练集取 buffer 的前向闭包**（4×4=全 reachable 41503，续值查询全覆盖；8×8 撞 cap 由 BUFFER_COVERAGE_τ 单独 gate）→ 值改为**向上爬**（0.57→0.68→…）。self-play buffer 仍跑、用作覆盖率诊断（cover≈0.49%）。教训：FVI 必须"在 bootstrap 续值的态上训练"，buffer 覆盖不足会静默崩塌成贪心。
- **状态**：迁移 gate 收敛运行中（detached pid，220 sweep / 2700s budget，写 rl8_gate.json）。架构已验证（值向 bdp_T 爬升）。剩余 Phase 1：gate 收敛到 rel<5% + 8×8 单 sweep wall-clock 外推（HARD_CEILING 7 天判据）。未 commit/push。

### 续12 补（迁移 gate 性能墙 → MPS 解，2026-06-02）
- 首个收敛运行 FAIL **但非代码 bug**：CPU 上 CNN 全 reachable(41503) full-batch FVI = **2187ms/fwd+bwd → ~180s/sweep**，2700s 只跑 15 sweep（值还在 1.15，靶 3.59，远未收敛）= inconclusive 欠训，非 gate FAIL。
- **MPS 解（§3.2 "CPU 太慢再试 MPS" 触发）**：实测 MPS=211ms/fwd+bwd（**10× 加速**）→ 13.7s/sweep → 100+ sweep ~25-35min 可收敛。改 `DEVICE = "mps" if available`。
- **MPS 两个坑（已修）**：① `board_plane` 逐格赋值必须**在 CPU 构建再整批 .to(DEVICE)**——在 MPS 上逐格赋值是上千次慢分发；② probe 用的 `torch.zeros(1)`/`torch.full` 漏了 device → 必须显式 `device=DEVICE`（否则 input 在 CPU、权重在 MPS 报错）。`scatter_reduce(amax)` 在 MPS **支持**，无需回退。
- 重启收敛运行：hidden=128（rl4 教训 hidden=64 mode-γ 有 +3.9% FVI 高估偏置，gate 要 rel<5% → 用够宽的网），max_sweeps=250 / min_sweeps=120 / 2700s budget，harness-tracked。

### 续12 补2（迁移 gate 收敛工程：冻结目标 + 断点续训，2026-06-02）
- MPS 收敛运行 144 sweep（19s/sweep, hidden=128）：值稳定爬升 v8 0.69→2.65 / v16 1.34→3.66，loss 0.34→0.007，**未 plateau，撞 2700s wall 时仍 rel≈25%**（靶 3.59/4.86）→ 需更多 sweep（外推 ~250-300 sweep ≈ 80-95min）。非代码 bug，是 backward-induction 逐层传播慢。
- **两处工程改进**：① **冻结每-sweep 目标**（rl4 式：先用冻结 net 建全 k 目标再训）——原实现在一个 sweep 内逐 k 边算目标边训、共享权重的 k 轴有 catastrophic interference / moving-target；冻结后更稳（代价：传播 ~1 层/sweep，需 ≥16 sweep 填满各 k 层）。② **断点续训** ckpt_path/resume_ckpt（save/load state_dict），长收敛跑可跨 loop 轮次续，不从零重来。均已 smoke 验证（save+resume OK）。
- 重启长跑：max_sweeps=300 / min_sweeps=200 / 5400s budget / inner_steps=3 / lr=2e-3 / hidden=128 / ckpt 每 20 sweep → /tmp/rl8_gate.pt，harness-tracked。

### 续12 补3（迁移 gate 300 sweep：T=8 PASS / T=16 10% 续训中，2026-06-02）
- 冻结目标长跑 300 sweep（4315s, hidden=128, lr=2e-3）：**T=8 rel 3.84% → PASS ✓**（与 rl4 mode-T hidden=128 的 2-4% 一致）；**T=16 rel 10.3% → 仍 climbing**（sweep270=4.30, 末 4.36, plateau=False, 噪声 ±0.4），长 horizon k=16 层数多收敛慢。
- **断点续训生效**：从 /tmp/rl8_gate.pt(sweep300) resume，lr→1.2e-3 + inner_steps=4 压噪声续推 T=16<5%。注：4×4 closure 恒=全 reachable，recollect 无意义 → 关掉（recollect_every=9999）。
- Adam moment 未存（仅 net 权重）→ resume 有短暂 transient，影响小。

### 续12 补4（迁移 gate PASS + 8×8 wall-clock 实测，2026-06-02）
- **§3.0 迁移 gate PASS**：resume 续训 250 sweep（4587s, lr1.2e-3, inner4）→ T=8 rel **1.04%** / T=16 rel **2.74%**，both<5% ✓。值噪声 ±0.3 但 trailing-mean 稳在容差内（非 lucky snapshot）。pipeline 认证，放行 8×8。rl8_gate.json pass=true。PLAN 状态块 PHASE_1_INFRA=IN_PROGRESS。
- **8×8 单 sweep wall-clock 实测**（T=50, hidden=128, inner=3, MPS）：buffer=3054(100 rollouts) → eval-dst=**198k** trans=289k，setup=31.7s，**per-sweep≈15.2s**。
- **HARD_CEILING 初算**：15.2s × ~1700 收敛 sweep(4×4 经验 T=16→~550 sweep 线性外推 T=50) × 1.5 ≈ **11h ≪ 7天**。**但**这是 buffer=3054 的欠覆盖配置（eval-dst 198k vs buf 3054 = 续值在 65× 未训态上 bootstrap）——4×4 已证欠覆盖会值崩塌。真收敛需更大 buffer → per-sweep 涨 → 可能逼近/超 7天。⏳ 正跑 40-sweep 8×8 收敛行为测（climb vs collapse）定夺。
- **未决 Phase 1 缺口**：① combo 未接线（8×8 buffer/编码现 combo=0，需存 (board,combo) 并编码——计算开销可忽略但科学正确性必须）；② 覆盖率→收敛→真 per-sweep→HARD_CEILING 终判。

### 续12 补5（8×8 收敛行为测：覆盖 OK 但 §3.6 重尾暴露，2026-06-02）
- 40-sweep 8×8 测（buffer=2944, eval-dst=193k, T=50, 878s=22s/sweep）：**值 climb 不 collapse**（v16 8.8→215）→ 覆盖在 buffer~3k 不崩塌，**预算非瓶颈**（22s×~1700 sweep×1.5≈18h<7天，HARD_CEILING 不触发）。
- **但暴露 §3.6 重尾**：loss 巨大（13461→7092 仍数千量级）、值**指数上冲不收敛**（v16 sweep30=70→sweep40=215）。combo 计分无界 → V_total 达数百且仍涨，**per-round-rate 单独不足以压重尾**（高分态等权淹没回归）。且本测 combo=0（每轮重置）已如此，真 combo（跨轮串联）更重尾 → 更糟。
- **结论**：§3.0 gate PASS + 预算可行（~18h）；瓶颈转为 **§3.6 变换选择**（plan 预注册要在 8×8 held-out 高分态对比 per-round-rate / log1p / Huber-augment 选最稳）+ **combo 接线**（buffer 存 (board,combo)+编码）。二者是 Phase 1 收尾前置，且变换是"实质方案"决策。

### 续12 补6（combo 接线完成 + 验证中，2026-06-02）
- **用户授权自主跑 combo 接线 + §3.6 变换 micro-check**（AskUserQuestion 选项1）。
- **combo 接线落地**（buffer/转移表/编码全线）：① collect_buffer 存 **(board,combo)** tuple（combo 跨轮 real threading）；② build_buffer_transitions eval 集按 (dst_board,dst_combo) 去重，src/eval 都带真实 combo；③ RateNet.forward combo **/10 归一化**（保 O(1)，combo=0 不变 → 4×4 gate 不受影响）；④ train_modeT 用 tr['src_combo'] 训练输入；make_buf 统一 4×4 闭包(combo≡0)/8×8 采样集。修 _smoke 漏 .to(DEVICE)（DEVICE=mps 后暴露）。smoke 验证 combo 真流入（max combo 3-5）。
- 验证中：4×4 回归（combo≡0 路径不应崩）+ 8×8 real-combo 特征化（真 combo 比 combo=0 测更重尾？）。下一步：§3.6 变换 micro-check（per-round-rate / log1p(rate) / Huber-augment 在 8×8 held-out 高分态对比选最稳）。

### 续12 补7（real-combo 重尾确认 + §3.6 micro-check 运行中，2026-06-02）
- 4×4 回归（combo 接线后）：10 sweep v8 0.43→1.17 健康爬升、不崩 → combo 接线没破坏认证路径 ✓。
- 8×8 **real-combo** 特征化（T=50, 30 sweep）：loss **20953**（vs combo=0 测 13461）→ 真 combo **更重尾**，per-round-rate+MSE 确认不足。
- **§3.6 变换 micro-check 落地**：train_modeT 加 `transform` 参（decode_vtot/encode_target/loss_fn 三处一致切换 + collect/_greedy_pick 同步），对比 `rate`(默认 rate+MSE) / `rate_huber`(rate+Huber) / `lograte`(log1p(rate)+MSE 压重尾)。combo /10 归一。正跑 8×8 三变换 18-sweep 对比（loss 量级 + 值收敛稳定性）选最稳。
- 预期：lograte 把 O(10²) rate 压到 O(1) → loss O(1)、值有界收敛；rate/rate_huber 仍上冲。待验证。

### 续12 补8（§3.6 micro-check 定论：lograte 胜，2026-06-02）
- 三变换 8×8 real-combo 18-sweep 对比：**rate** loss 19022→16424（病态，重尾淹没低分态）/ **rate_huber** loss 496→507（flat，卡住没学）/ **lograte** loss **104→59（降 43%，well-conditioned）**。三者 v16>v8（k 单调对）。
- **§3.6 winner = lograte**（log1p(rate)+MSE）：log 压缩把 O(10²) rate 映到 O(1-5)→回归良态。**实证推翻 plan 的 rate 默认**（plan §3.6 本就预留 micro-check override）。
- **下一步严谨性**：§3.0 gate 是用 rate 认证的；lograte 是新变换的 encode/decode（log1p/expm1）未对真值验证。→ 重跑 4×4 迁移 gate transform=lograte，确认也复现 bdp_T(8/16) 才认证 lograte 无 bug，再用于 8×8。架构+buffer+FVI 已认证不变，仅验变换层。

### 续12 补9（lograte 4×4 gate 验证：T=8 PASS / T=16 续训中，2026-06-02）
- lograte 4×4 gate 首跑 304 sweep（5402s, loss 良态 ~0.03）：T=8 rel **2.51% PASS** / T=16 rel 11.4% climbing（plateau=False）——与 rate gate 首跑**同模式**（T=8 先收敛、T=16 层多滞后、需 resume），**非变换 bug**，lograte 朝 bdp_T 正确收敛。
- resume /tmp/rl8_lograte.pt 续 250 sweep（lr1.2e-3）推 T=16<5% 完成认证。认证后 lograte 即 §3.6 定盘 → 转 8×8 收敛验证 + HARD_CEILING 终判 + Phase 1 收口。

### 续12 补10（lograte 失正确性 gate — 变换选择反转，2026-06-02）
- lograte resume 250 sweep（554 总）：T=8 rel 3.4% PASS，但 **T=16 卡在 rel 11.3% 不再改善**（震荡 3.66-4.35≈4.1，loss 低稳 0.005）→ **收敛到真值下方 ~11% = 系统性低估偏置，非欠训**。
- **机理**：lograte decode `V_total=expm1(net)×k`，高 k 时 **×k 放大** log 空间小误差（0.02 log 误差 ×16≈11% V 误差）。lograte 用重尾换来了高-k 放大偏置 → **FAIL 4×4 正确性 gate**。
- **反转结论**：micro-check 的"loss 最小"是假象（lograte loss 小只因 log 空间目标小，非拟合更好）。真裁判是 4×4 正确性 gate：**rate PASS(2.74%) / lograte FAIL(11.3%)**。rate（plan 默认）才是正确的，但其 8×8 重尾下的收敛性未确认（30 sweep loss 大但在降）。
- **决策点**：rate 重新成为首选（正确性已证），但需确认 rate 在 8×8 重尾下能收敛 / 或测第 4 变换 logv=log1p(V_total)（无 ÷k/×k，可能两全）/ 或降 T / combo cap。上抛用户定向。

### 续12 补11（独立复审推翻 rate 推荐 + 建 8×8 精确真值锚，2026-06-02）
- 用户要求"推荐→独立 agent 审核→再执行"。独立复审 **REJECT** 我的"rate+优化"推荐，抓到核心漏洞：**rate decode 也是 u×k（rl8.py:267），与 lograte 的 ×k 放大同病**——rate 过 4×4 只因 4×4 有界、**根本测不到重尾域**；"重尾是优化问题"是回避 §3.6 存在理由。且 grad-clip 会**加重**偏置（压低高分态=EVPI 命脉）。
- **采纳复审路径**（更严谨）：① 加 `logv=log1p(V_total)`（decode expm1，**无 ÷k/×k**，去掉放大器+压尾）；② **建 8×8 精确真值锚** `exact_vtot_anchor`——用训练同一组 8 固定 hands 精确后向归纳（非 MC，memo），算出 V_total(empty,k)={1:**11.0**, 2:**65.9**, 3:**110.6**}（k1→k2 暴涨证锚确实压尾域，T=3 精算 52s）。**第一个能在 8×8 重尾域验变换正确性的靶**。
- 正跑决定性对比：rate/lograte/logv 训到 8×8 T=3，比各自 V_total(empty,1/2/3) 对锚的 rel 误差 → 证据选变换（不再靠"哪个 loss 小"假象）。

### 续12 补12（★锚测反转：变换不是主因，覆盖率才是，2026-06-02）
- 8×8 T=3 锚测三变换全败且**同模式**：anchor V(empty,k)={11.0,65.9,110.6}，rate={16.5,30.9,43.3}/lograte={17.1,24.6,26.3}/logv={16.0,23.0,33.0}，rel 全 45-76%，k=2/3 **集体大幅低估**、k=1 高估。
- **诊断反转**：三变换同败 → **不是变换问题，是采样 buffer 覆盖率**。self-play(+ε,未训好的 net 贪心)不访问 combo-setup 高价值态 → FVI 的 max 看不到高续值 → 系统性低估。正是 §3.4 on-policy 漂移，被精确锚揪出。
- **下一步隔离实验**：用 anchor memo 枚举的**全 T=3 reachable 闭包**当训练集（满覆盖）重训三变换 → 若仍败=变换/优化问题；若达标=确诊覆盖率（需 §3.4 off-policy 补采 strong/seer 轨迹态）。变换裁决必须在满覆盖下才公平。

### 续12 补13（★隔离实验定论：logv>>rate，覆盖非唯一因，2026-06-02）
- 满覆盖(全 T=3 闭包 buf=5288)重训 vs 锚{11.0,65.9,110.6}：**rate**={15.9,28.5,37.6} rel{45%,57%,66%}（满覆盖仍大败）；**logv**={16.3,47.5,137.0} rel{48%,**28%**,**24%**}（k2/k3 远胜 rate）。
- **两个定论**：① 覆盖率**非唯一因**——rate 满覆盖仍 45-66% 败 → 重尾是 rate 真收敛问题（MSE 被高 V 态梯度主导）；② **logv（log1p(V_total) 无 ×k）尾部远胜**（复审 no-×k 假设证实）→ logv 是 §3.6 前锋。
- **遗留**：两变换 k=1 都 ~45% 高估（既非覆盖也非变换，疑 FVI/收敛 artifact，empty 单点被邻态平滑抬高 / 高 V 梯度淹没 k=1）；150 sweep 均未收敛；logv k=3 反超(137 vs 110)需更多 sweep 看是否回落。
- **下一步**：logv 拉长 sweep（+断点）看是否收敛到锚 + 查 k=1 高估根因（网宽？max 偏置？k=1 梯度权重？）。logv 若收敛达标→过 4×4 gate→定盘。新增 `buf_override` 参（train_modeT）支持满覆盖隔离测。

### 续12 补14（★§3.6 定盘规则重写[用户审计] + 高-k 自洽真裁判 instrument 落地 + logv T=50 训练 launched，2026-06-02）
- **用户审出旧规则真洞**：旧 NEXT_ACTION"训 T=3、比锚、选 rel 最小者"**区分不出 logv（无×k）vs rate/lograte（被×k）**——`exact_vtot_anchor` 只精算到 T=3，生产 horizon T=50 的 ×k 放大量比 T=3 大 ~17×，一个 T=3 锚最小的变换到 k=50 照样爆。
- **§3.6 定盘规则重写并预注册**（PLAN §3.6★ + 状态块 TRANSFORM/SELFCONS_GATE_HIGHK + LAST_USER_GATE，写后不许改）：(a) 机理优先 logv（log1p(V_total) 无÷k/×k）；(b) T=3 锚降级为"低级-bug 证伪器"；(c) **真裁判 = 高-k 自洽 V_net vs MC-rollout @ k≈50**（MC 真实玩 k 轮累加真分、不经 decode ×k = 对放大免疫的无偏 V^π 参照；标准 rollout policy eval / Bellman 自洽，arXiv:2504.02221）。PASS = 聚合中位 rel@k50<0.10 **且** rel@k50/rel@k25<2.0（无-×k 放大签名）；(d) 全候选败→USER_GATE 重设计（降 T / cap combo），不许默默 inconclusive。
- **instrument 落地 rl8.py**（新增）：`board_planes_fast8`（split-32bit-半 bit-unpack，避 signed-int64 bit63 溢出，10.7ms/24k boards）/ `_strong_states`+`_seer_states`+`offpolicy_states`（§3.4 strong+seer off-policy 态，seer=真未来短-strong-rollout 前瞻触及高-combo 尾部）/ `collect_probes`（empty+32 冻结中局态）/ `mc_rollout_value`（M episodes 并行、每轮 beam_hand python+net 批量评分、16-block-SE）/ `vnet_predict` / `selfcons_highk`（orchestration+JSON+verdict）。train_modeT 加 `offpolicy=(ns,nz)` 参（8×8 训练 buffer 混入 strong/seer 态 → 让自洽测【变换】非【覆盖】）。CLI 加 `train8`/`check8`（设 T_MAX=生产 horizon 50）。
- **smoke 全绿**：train8 4-sweep（off-policy 补采 210 态、buf 820、v8 13.9→109.7 健康爬升、31s）；check8 M=48/5 probes 跑通（empty k50: V_net=119 vs MC=2338±143 rel0.90=欠训预期；**放大签名 rel@k50/k25=1.06≈flat → logv 无×k 放大，签名检测器工作正常**）。
- **吞吐实测**：beam_hand B=12 = 1.72ms/call（582/s）→ 真 check(M=2000,33probes,k25+50) 外推 ~1h。
- **logv T=50 真训练 launched**（harness-tracked bg，3h chunk，offpolicy=(20,20)、hidden=128、lr2e-3、min_sweeps≥50、断点续 /tmp/rl8_8x8_logv_T50.pt）。plateau 后跑 check8 → 定盘 logv 或测 rate/lograte 或 USER_GATE。
- **踩坑**：首次 launch 误叠 `nohup &` + run_in_background（违 §0.5 XOR）→ kill 重launch 为单一 tracked job。

### 续12 补15（★用户重定向 STRENGTH-FIRST：logv 定盘、最强=实战比分，2026-06-02）
- 用户质疑"方法很奇怪——我们是要找最强策略"。对齐结论：方法主干(afterstate-V)对，但**§3.6 三变换自洽选秀过重**。
- **治理覆盖（PLAN §3.6★★，优先级高于 §3.6★）**：(1) **logv 直接定盘**（机理无×k + 補13 尾部 rel 24-28%≫rate 57-66%，证据已足，不办三方 PK）；(2) **"最强"真裁判 = Phase 3 实战比分**（rl-greedy-on-logv vs strong 同种子 CRN，看得分，TOST），不是 V 校准；(3) **高-k 自洽降级为 sanity sidecar**（跑一次、advisory，给目标②天花板可信度背书，不当 blocking gate）；(4) 逃生分支仍在（logv 发散→USER_GATE）。原 §3.6★ (a)-(d) 保留作预注册记录，仅 (c) 从"真裁判"降"sanity"。
- **执行重心转**：logv 训练继续（在跑）→ plateau 后**建 competence_gate**（rl vs strong 配对比分）= 最强 verdict → 顺手 check8 一次 sanity → HARD_CEILING → Phase 1 DONE → Phase 4 路由。
- 关键认识：策略"最强"只需 argmax 排序对（实战得分高），不需 V 绝对校准；補13 的 logv k=1 高估只伤目标②天花板精度、不伤目标①策略强度 → 记为局限不阻塞。

### 续12 补16（logv T=50 训练 MPS OOM → chunk forward 修复，2026-06-02）
- 首段 logv 训练 sweep 30 后 **MPS OOM 崩**（exit 1）：sweep 20 recollect 把 buffer 8267→15075 → eval-dst 586k→~1M+ → `y_total_from_trans` 单批 forward 的 conv 激活 (N,32,8,8) 要 11.5GiB → 爆。非逻辑 bug，是 full-batch forward 不随 buffer 长大缩放。
- **两修**：① `net_forward_chunked`（eval forward 分块 120k，激活峰值压到 chunk×32×8×8，与 buffer 大小解耦）；② 8×8 训练 `recollect_every=10**9` 关闭重采（off-policy strong+seer + 初始 self-play 已覆盖，固定 buffer 更稳，不再无界长大）。
- sweep-30 ckpt（loss 良态、值在爬）保住 → resume 续训，未丢进度。教训并入 rl8_phase1 教训3（MPS 坑）。

### 续12 补17（★logv T=50 训练发散→NaN：value blow-up，上稳定化镇定剂，2026-06-02）
- logv 续训（resume sweep30）**发散**：v8 116(sw0)→87386(sw30)→NaN(sw80)，撞 wall(120 sweep)收尾全 NaN。回看首段早已剧烈震荡(17→57→220→99)、从未真稳。
- **根因 = FVI value blow-up**：logv 的 `expm1` 解码 + max 自举 + T=50 五十层后向归纳成正反馈——网络稍高估→expm1 指数放大→max 总挑最高估续值→跨 sweep 滚成 1e5→NaN。**logv 躲了 ×k 偏置，却引入 expm1 爆炸不稳定**（heavy-tail 换形式咬回）。末尾 ckpt 被 NaN 污染→须冷启。
- **稳定化三件（补17，标准优化镇定，非 §3.6(d) 重设计）**：① `decode_vtot` logv `clamp(u,max=11)`→V_total≤5.9e4 物理天花板(seer T=50 ~2.7e4<此，不偏真值)，**结构上杜绝 NaN**；② `grad_clip=1.0`（train_modeT 加参）；③ lr 2e-3→5e-4。
- 已冷启 45-sweep 稳定性 probe（/tmp/rl8_logv_diag.pt）验是否有界且不顶到 cap。probe 过→冷启全程训练；probe 仍发散/顶 cap→才升级 §3.6(d) USER_GATE（降 T / 真 cap combo）。

### 续12 补18（★logv 稳定化后仍退化→§3.6(d) USER_GATE，2026-06-02）
- 45-sweep 稳定化 probe（lr5e-4+grad_clip1.0+clamp11）：**NaN 杜绝**（clamp 生效），**但 logv 退化**——值爬到 ~3-4.4e4 顶在 cap 附近，且 **v8≈v16 全程**（net 不再区分 horizon，塌成"预测一个巨大近常数"）。lr↓/clip 只延迟不阻止爬顶。
- **结论：logv 在 T=50 真失败**（expm1×max-自举把 net 推到天花板，与稳定化无关）→ 命中预注册 §3.6(d) USER_GATE。
- **关键再认识（strength-first 下）**：logv 当初是为【无 ×k 偏置=目标②天花板校准】选的；但目标②已降级 sanity。**rate(×k) 解码是线性、不会 expm1 爆炸、4×4 已证收敛**；其 ×k 偏置只伤 V 绝对校准（目标②），**不一定伤策略实战强度（目标①，用 rollout 得分判）**。→ 推荐换 rate + 同款稳定化，×k 偏置记为目标②局限。备选：降 T / cap combo（改游戏口径，需用户拍板）。已 USER_GATE 报用户。

### 续12 补19（★rate 稳定性 probe PASS → 全程训练 launched，2026-06-02）
- rate 45-sweep probe（lr5e-4+grad_clip1.0+clamp5000）：**健康，与 logv 退化相反**——v16≈2×v8 全程(13/27→170/339→833/1666=正确 horizon 区分，真价值函数非塌缩)、平滑单调爬升、远不顶 cap、per-round~104（strong ~121/round 同量级）。loss 大(~16k 在降)是 rate 目标绝对量级大的已知 conditioning(补8)，非发散。
- **rate probe PASS** → §3.6(d) 决议落地：**TRANSFORM=rate 定盘**。launched 全程 `train8 rate 50`（cold start，stabilized，3h chunk，断点续 /tmp/rl8_8x8_rate_T50.pt，offpolicy=(20,20)）。
- 下一步：plateau → 建 competence_gate（rl-greedy-on-rate-V vs strong 配对比分）= 最强 verdict → check8 sanity（量化 ×k 偏置=目标②局限）→ HARD_CEILING → Phase 1 DONE。

### 续12 补20（★★rate 也发散 → 确诊：无折扣长-horizon FVI 本质不稳，非变换问题，2026-06-03）
- 全程 rate T=50（144 sweep）：v8 healthy 到 sweep~40，之后 per-round u 失控爬升 sweep20 u14→sweep60 u135→sweep100 u1107→sweep140 **u3844**（真值应 ~120-150），顶到 clamp(5000)；loss 20k→**29M** 爆。**假 PLATEAU=True**（clamp 冻住值→变化率<0.5%骗过 plateau 检测，危险：自动管线会误判收敛）。
- **确诊（关键）**：logv(expm1 爆) + rate(线性爬顶) **两变换都在 T=50 发散** → **不是变换问题，是无折扣 mode-T 值迭代在长 horizon 本质不稳**。根因 = deadly triad（bootstrap + 函数逼近 + max 高估偏置，补rl4①已记 +3.9%/层）**跨 50 层无折扣复利** + combo 重尾 → 爆。变换只改"怎么爆"不改"爆不爆"。
- **与 4×4 同构**：dp4 续5 R2 BLOCKING 早发现"无折扣 SSP V* 发散→必须 γ-折扣(0.95)"。8×8 T=50 无折扣撞同一面墙。§0.2 当初禁 γ 是为对齐 channelB 无折扣求和——但 strength-first 下天花板=策略**实战 undiscounted 得分**(rollout)，非 V → γ 只当**训练稳定器**、不进 channelB 比较 → §0.2 可不破。
- **→ §3.6(d)/§0.2 USER_GATE**：变换路线穷尽。推荐 **γ-折扣训练(稳定器) + undiscounted T=50 得分判强**（principled：γ-收缩保证有界收敛，4×4 已验；与 strength-first 兼容）。备选：降 T(粗) / cap combo(改游戏)。已报用户。

### 续12 补21（γ 实现 + γ=0.95 probe：增长被压、有界趋势，上全程验 plateau，2026-06-03）
- 实现 GAMMA（module 全局，默认 1.0 保 4×4 gate；穿进 y_total_from_trans 训练目标 / _greedy_pick self-play / mc_rollout 选点；rollout totals 仍累加**原始** hand_score=无折扣判强口径）。GAMMA=1.0 回归 OK。8×8 train8/check8 自动设 0.95。
- γ=0.95 probe（60 sweep, 小 buffer）：v8 sweep60=261（vs 无折扣 rate 同期 1083，**~4× 更低、爬升更慢**）→ γ 在压增长，符合理论（γ 把无界 ε/(1−γ)→有限 20ε，有界不动点）。loss 仍大但未爆。
- 小注：v16/v8≈2.0（非 γ 收敛态的 ~1.66）→ net 尚未学到 k-曲率（早期正常，不影响策略 ranking）。60 sweep 太短看不到 plateau。
- → 上**全程 γ=0.95 rate 训练**（cold start）验是否 plateau 到有界 sane 值（v8 预期 ~800-1000）而非顶 clamp。这是 γ 路线的决定性测试。
