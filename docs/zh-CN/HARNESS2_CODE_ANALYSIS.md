# Harness² 源码深读：让 Agent 改进自己的执行框架，而不是训练模型权重

> 阅读对象：希望从论文概念一路追到函数、文件、实际执行与实验边界的工程读者。
>
> 分析日期：2026-10-22。代码固定在上游 `google-research/harness2` 的提交 `1cfe254e8d25b15f37048c3730134686abe60335`；本文源码链接均指向该提交，不随 `main` 漂移。论文为 [Harness²: Recursive Agent Harnessing for an Open World，arXiv:2610.14224v1][P1]。
>
> 这是独立中文技术分析，不是 Google 官方文档。本文区分「源码可确认」「论文报告」「本次实际验证」与「工程推论」，不把 toy demo 当成真实模型效果复现。

## 先用一分钟建立正确认识

Harness² 不是新的大模型，不是推理服务引擎，也不是一个训练权重的 RL 框架。它是一套 **Agent 外层的在线改进流程**：观察同一任务上的执行，修改可执行的 harness 文件，让新 harness 再做同一任务，再根据行为差异进行组合。任务结束后，把采用的 harness 和有证据的经验放入改进库，供未来任务的改进器使用。

最重要的五个结论：

1. **改的是模型周围的软件与文件，模型参数不变。** 提示词、技能、Python 脚本、子代理定义、运行时插件/钩子都可能改变，不只是改一句 system prompt。
2. **一次基本步骤有四次任务执行。** 原始 harness 执行一次，通用候选执行一次，领域候选执行一次，组合 harness 再执行一次。`k=1` 不等于只调用一次模型。
3. **当前任务的改进与跨任务的积累是两层不同循环。** 当前任务内可以加宽或加深探索；下一任务仍从共同的 base harness 开始，并不是直接继承上一任务的最终 harness。
4. **无 ground-truth feedback 不等于没有评测。** 改进/选择阶段不接收 benchmark 的标准答案或分数，外部 judge 在选择提交之后另行评分。
5. **这是研究实现，不是现成安全边界。** 官方明确指出 executor 网络默认不受限制，Codex sandbox 被禁用。不能直接在有生产密钥、重要文件或敏感数据的主机上运行真实 agent 实验。[S01][S03][S07][S15][S24]

如果把此前阅读的推理系统看成「让每一次模型生成更快、更省显存」，Harness² 关注的是另一个层次：**给定一个会生成 token 的模型，怎样安排它读什么、用什么工具、如何执行，才能更可靠地把任务做完。** 两层可以配合，但不是同一种优化。

## 目录

- [1. 项目身份、分析范围与代码地图](#1-项目身份分析范围与代码地图)
- [2. 四个对象：Model、Executor、Execution Harness、Improvement Harness](#2-四个对象modelexecutorexecution-harnessimprovement-harness)
- [3. 八类组件怎样真正影响运行](#3-八类组件怎样真正影响运行)
- [4. 从命令行到一次真实 rollout](#4-从命令行到一次真实-rollout)
- [5. Propose–Probe–Compose 的文件级实现](#5-proposeprobecompose-的文件级实现)
- [6. Parallel 与 Sequential：预算和依赖关系](#6-parallel-与-sequential预算和依赖关系)
- [7. 选择与跨任务经验：哪里递归，哪里不递归](#7-选择与跨任务经验哪里递归哪里不递归)
- [8. 本次实跑：账本输入如何变成正确产物](#8-本次实跑账本输入如何变成正确产物)
- [9. 评测结果应该如何读](#9-评测结果应该如何读)
- [10. 缓存、失败、恢复与复现陷阱](#10-缓存失败恢复与复现陷阱)
- [11. 如何复现、扩展与部署](#11-如何复现扩展与部署)
- [12. 最容易读错的十句话](#12-最容易读错的十句话)
- [13. 验证记录与源码阅读索引](#13-验证记录与源码阅读索引)

## 1. 项目身份、分析范围与代码地图

- 官方代码仓：[google-research/harness2][R1]。
- 论文作者单位包含 Google Cloud AI Research 与 Northwestern University；不应把该项目直接称为「Google DeepMind 的官方产品」。README 本身也声明不是受官方支持的 Google 产品。
- 本次分析快照：`1cfe254e8d25b15f37048c3730134686abe60335`，提交时间 2026-10-14 02:22:49 UTC，提交说明为 `Fix arXiv link`。
- `pyproject.toml` 包版本是 `0.1.0`，要求 Python `>=3.12,<3.14`。包内版本号不等于已经发布了同名 GitHub Release。[S02]
- 检查到的 `AGENTS.md` 位于 `third_party/jb_runner/codex_base_harness/`，属于被研究的 base harness 文件，不是项目根目录对本报告的编辑规范。本次只新增分析文档，不修改执行代码或该目录。

主要代码只有三个需要始终分开的轴：

| 轴 | 对应目录 | 负责什么 | 不负责什么 |
|---|---|---|---|
| 改进方法 | `harness2/core/` | 递归、证据、候选编辑、选择、经验库、预算 | 不认识某 benchmark 的任务目录细节 |
| 执行底座 substrate | `harness2/substrate/` | OpenCode / Codex 可编辑文件规则、安装与日志格式 | 不决定任务语义与评分标准 |
| Benchmark adapter | `harness2/benchmarks/` | 取任务、准备源文件、启动 runner、抽取产物、评分 | 不重新实现通用递归算法 |

入口 `harness2/cli/run_sweep.py` 组装这三个轴。`BenchmarkAdapter` 是它们之间的契约：提供 `tasks()`、`task_prompt()`、`execute()`、`trajectory()`、`deliverable_text()`、`base_harness()`、`validate_tree()` 和事后使用的 `score()`。`RunView` 只记录执行是否完成、是否有产物、路径和错误等，不携带分数。[S04]

这套拆分的价值是：同一个 `run_mode()` 可以用于法律工作、Office 文档或代码任务，也可以把 OpenCode 换成 Codex，而无需修改递归本身。但这不意味着接一个新 benchmark 只需写几行配置，可靠的材料隔离、产物抽取和执行适配仍是实际工程工作。

## 2. 四个对象：Model、Executor、Execution Harness、Improvement Harness

### 2.1 模型不是 harness

模型是冻结参数的生成能力。真实执行时，底座会反复调用模型，根据响应运行工具，再把工具结果送回模型。Harness² 没有在这个过程中计算梯度、优化权重、保存新的 checkpoint。

执行 harness 是「让这个模型如何做事」的一组文件和设置。例如：

- 先读需求再开工，还是先写一版草稿；
- 财务金额统一采用 Decimal，还是随意使用浮点；
- 遇到长 PDF 时加载哪个 skill；
- 是否把资料核查交给一个单独子代理；
- 运行工具后是否由 hook 强制触发某项检查。

同一模型配不同 harness，可以产生不同工具序列和最终产物。**优化对象是行为的外部配置，不是模型内部参数。**

### 2.2 Executor 与 Editor 可以是不同模型

任务执行器用 execution harness 完成原任务；编辑器/组合器则读取执行证据，修改这个 harness。选择器比较候选产物；反思器提炼可以迁移的经验。这些是逻辑角色，不代表一定部署四个独立模型。

代码用 `ChannelSpec` 描述改进侧模型通道，和任务执行 substrate 分开。`Proposer` 这个类名容易误导：它不仅服务 Propose，也承载 Compose，以及选择/反思会话。JobBench 默认把 proposer 和 selector 的 channel 设为同一个配置。[S04][S09][S18]

论文实验的默认 editor 是运行于 OpenCode 的 Claude Opus 4.8；executor 是 Gemini 3.5 Flash 或 GLM-5，配 OpenCode/Codex，形成四种组合。**这是论文实验设置，不是代码默认替用户选好的模型。** 发行版 `HARNESS2_SOLVER_MODEL` 默认空，improver 未另设时跟随 solver，judge 默认也为空；真正运行前要求配置。[P1][S25]

### 2.3 Improvement Harness 不是「另一个正在训练的神经网络」

可以把改进侧分成两部分：

1. 固定的阶段规则、提示词、执行流程与选择规则。
2. 会增长的 harness improvement library，包含被采用的完整 harness 文件树及经验解释。

当前任务内，这份库保持固定；当前任务结束后才写入新经验。下一任务的编辑器会在新的证据上下文里看到扩充后的库。因此改进经验会影响未来「怎么提出和组合修改」，形成 harnessing-level recursion。它不是每轮让程序任意重写自身递归算法，更不是无限自我改进保证。[S07][S14][S15]

## 3. 八类组件怎样真正影响运行

「八个组件」是概念分解，不是所有底座都恰好八个独立文件。

| 组件 | OpenCode 可编辑表示 | 作用与边界 |
|---|---|---|
| System prompt | `systemprompt.md` | 常驻执行策略与工作方式 |
| Guardrails | `guardrails.md` | 常驻行为约束；纯文字本身不是 OS 安全沙箱 |
| Memory | `LongTermMEMORY.md` | 注入 execution harness 的记忆内容；与外层 improvement library 不是同一份东西 |
| Tool guidance | `tool_descriptions/tool_guidance.md` | 教 agent 何时、怎样使用已有工具；不等于重写每个工具的实现 |
| Subagents | `sub_agents/<name>/AGENT.md` | 可委派角色；还受 adapter 是否允许 subagents 限制 |
| Skills | `skills/<name>/SKILL.md` 及技能脚本 | 可按需加载的工作流程；必须有描述和可触发入口 |
| Scripts | `scripts/<name>.py` | 真正执行的确定性辅助程序，可改变数据处理逻辑 |
| Plugins | `plugin/<name>.ts` | 在运行时工具循环内部介入，而不只给模型建议 |

OpenCode 的前四类进入 always-on 文本，长度上限为 24,000 **字符**，不是 24,000 tokens。Codex 则把常驻内容集中在 `AGENTS.md`，本实现上限为 8,000 字符；技能放在 `.agents/skills/`，运行时扩展采用 `.codex/hooks.json` 而非 OpenCode TypeScript plugin。[S05][S06]

源码中的 `SurfaceSpec` 把差异表示为数据：允许哪些路径、技能命名规则、是否要求 subagent 的 `mode`、hook 支持哪些事件等。`read_tree()` / `is_allowed()` 控制路径范围，`tree.validate()` 再检查文字预算、skill frontmatter、触发引用、适用脚本的语法与 hooks 结构。不同 surface 的编译规则不同，例如 OpenCode 的通用校验并不编译 skill 目录内所有 Python 文件，不能把 validator 通过视为完整代码审计。`autotrigger()` 会在允许的长度内补技能触发行，避免「文件写好了但模型根本不知道它存在」。[S10]

以 JobBench + OpenCode 为例，原始组件文件并不是原封不动全丢给模型：`render_opencode_tree()` 把常驻文件组合成 `AGENTS.md`，把技能、子代理和 plugin 转为 `.opencode/` 下底座认识的布局，复制 scripts。**能编辑的源格式与底座实际加载的运行格式之间，有一个 materialise/render 层。**[S19]

## 4. 从命令行到一次真实 rollout

以下是源码追踪，不表示本次启动了付费模型实验。假设用户已准备好 JobBench、隔离环境、模型配置与预算，选择一个领域中的一个任务：

```text
harness2 --bench jb --substrate oc --k 1 --mode parallel ...
  cli.run_sweep.main / _main
    cli._common.make_adapter → JobBenchAdapter
    preflight.require_ready
    Proposer(channel)
    schedule.run_sweep
      schedule.run_domain
        schedule.run_task
          _roll_bases                         ← 原始 harness 执行
          _run_one_mode
            recursion.run_mode
              _run_stage(INV), _run_stage(DOM) ← 编辑 harness
              _roll(INV), _roll(DOM)           ← 候选执行
              _run_stage(AGG)                 ← 组合 harness
              _roll(AGG)                      ← 组合结果执行
            selection.trivial_selection       ← k=1 直接采用该 final
            selection.reflect_adoption
            Library.add
          summary.json
```

入口先确定任务流，做 preflight，核对保存的运行身份，再创建改进器。`--dry-run` 在创建 Proposer 前返回，可以预览已有数据里的任务流而不调用模型。`--max-tasks` 是**每个 domain 的任务上限**，不是整场实验总上限；没有收窄 domain 时，`--max-tasks 1` 仍可能运行很多任务。[S03][S20]

### 4.1 一次 `_roll()` 最后怎样到达模型

以 JobBench + OpenCode 的真实路径为例：

```text
recursion._roll
  tree.write_tree → harnesses/<label>/
  JobBenchAdapter.execute(task_id, harness_dir, run_id)
    检查 complete marker 与产物指纹；不能复用则重新执行
    substrate.materialise → 临时 Install(staging_dir, fingerprint)
    substrate.execute → benchmark 注入的 launcher
      JobBenchAdapter._launch_opencode
        render_opencode_tree
        _run_runner(jb_opencode_runner.py)
          run_one → _stage → _attempt → _command
            bun ... opencode/src/index.ts run ... --model ... --format json
            OpenCode 的模型/工具循环
          读取产物与 JSONL trajectory
    完成 marker + output fingerprint
  RunView(ok, delivered, run_dir, error)
```

这里最容易混淆的是：`OpencodeSubstrate.execute()` 并不自己知道 JobBench 如何启动，它调用构造时注入的 benchmark launcher。`run_argv()` 只是 reference argv，真实 runner 构造自己的命令；不要看见 reference helper 就误认所有 benchmark 都从它启动。[S08][S18][S21]

执行返回的是一个目录，不是直接返回 benchmark 分数。adapter 再把目录中的文件抽取成文本、图片或状态证据，供下一阶段查看。`ok=True` 只表示该级执行对象存在/被认为完成；`delivered=True` 只表示存在目标产物，也都不代表内容正确。[S04]

### 4.2 真实任务的输入和输出有什么

以 JobBench 为例，输入包括 `TASK_INSTRUCTIONS.txt` 与 `task_folder/` 内材料；执行器在独立 workspace 中做事，最终结果进入 `model_output/<label>/`，轨迹进入相应 `model_traj/`。改进器看到的是任务、材料、轨迹、抽取后的产物以及 harness 文件，而不是只看到模型最后一句自然语言。[S18][S21]

其他 adapter 可有不同交付形态，例如网页运行/渲染证据、Office 文件、代码环境状态。正因为输入输出复杂，论文中的「agent 在工作」不能简化成 `answer = model(question)`。

## 5. Propose–Probe–Compose 的文件级实现

### 5.1 Propose：让 agent 编辑文件，而不是只吐一个建议

`recursion._run_stage()` 是最值得逐行读的函数：[S11]

1. `context.build()` 把证据组织为目录。
2. `context.mission()` 生成短任务指令，告诉编辑器先读任务与当前 harness，再检查各次执行。
3. `prop.call(..., harvest="context/harness")` 启动改进会话。
4. 会话在隔离的临时工作目录中，直接编辑 `context/harness/`。
5. 回收编辑后的目录，读取允许范围内的完整文件树。
6. 自动补技能触发、执行通用 validator、benchmark validator 与评分信息 tripwire。
7. 若编辑不合法，把具体错误反馈给编辑器，再试一次；若仍不成功，继承该阶段的 fallback harness 并记录失败。

`Proposer.call()` 会复制上下文到临时目录，调用 OpenCode 或 Claude CLI，读取回答和审计记录，并在清理工作区前 harvest 文件。这里的「隔离目录」主要是证据/会话管理设计，不能因此推导出操作系统级安全隔离。[S09]

**以当前执行代码为准的细节：** `tree.py` 顶部仍有「未输出项继承」的注释，但本版关键路径不是解析自然语言 patch，再把部分文件合并回 base；它收回的是已预填当前 harness、经编辑后的完整目录。删除一个允许文件也可能体现到最终快照。读架构时应优先看 `harvest → read_tree → validate`，不要用旧注释替代实际控制流。[S10][S11]

### 5.2 两种候选不是两个连续补丁

- `INV`：从任务执行里找跨领域可能复用的问题，如计划、工具调用、检查遗漏、上下文管理。
- `DOM`：针对领域和任务环境中的约束，如数值单位、材料处理、专业工作流程。

在一个 contrastive step 中，两者都从同一个当前 harness 出发，形成两个独立候选。不是 INV 改完后再让 DOM 接着改；后者会破坏比较所需要的共同起点。

### 5.3 Probe：不读标准答案，也要真正执行

候选 harness 要在同一个任务上运行。比较的对象是：

- harness 到底改了什么；
- 工具轨迹在哪里改变；
- 产物发生什么变化；
- 有没有引入新错误或退化。

这比只让模型说「我的 prompt 会更好」多了一层实际证据，但它仍然不是正确性证明。模型可能共同误解任务；一次执行中的随机差异也可能被误归因于 harness 修改。

### 5.4 Compose：不是机械拼接，也不是多数投票

组合器读取 base/candidate 的 harness、diff、trajectory、deliverable 和决策记录，再编辑出新 harness。它可以保留某一候选的局部变化、重写冲突项、放弃证据不足的修改，甚至维持当前 harness。

新组合完成后还必须再跑一次原任务，生成真正要交付/选择的 final。**Propose 的输出是候选执行配置；Compose 的输出仍是执行配置；真正的用户产物来自 Probe/最终执行。**[S11][S12]

### 5.5 证据目录是一等公民

典型改进上下文如下：[S13]

```text
context/
  task.md
  source/                       原任务材料
  harness/                      当前要编辑的执行配置
  rollouts/<label>.md            执行轨迹
  deliverables/<label>/          可读产物/图片/索引，依 adapter 而定
  pool/<label>/                 候选 harness、diff 和执行证据
  decisions/                    先前修改与组合理由
  library/INDEX.md               过去任务的可复用经验入口
```

证据以文件提供，mission 只给路径与阅读要求，避免把整个历史硬塞进一条 prompt。这里也不是「绝对无截断」：代码允许对单个 observation 做显式截断，默认常见上限为 50,000 字符；部分 benchmark 可能只提供产物名称。文件存在不代表模型一定完整读完，读取工具与上下文窗口依然有限。

`copy_source()` 会去掉已配置的答案路径，`assert_no_grader_verdicts()` 检查指定文件名/字符串，`RunView` 没有 score 字段。这些共同支持「改进不消费评分」的架构意图，但不能阻止 executor 从互联网另行找到公开 rubric，也不是通用 prompt-injection 防护。[S04][S13][S24]

## 6. Parallel 与 Sequential：预算和依赖关系

令 `k` 为正整数。对于**单个任务、单种模式**，计划预算为：

```text
任务执行 rollout：R = k 个 base + 2k 个候选 + k 个组合结果 = 4k
改进阶段 session：P = k 个 INV + k 个 DOM + k 个 Compose = 3k
```

`Ledger.assert_identity()` 检查数量及 label 不重复。它是逻辑阶段账本，不是实际 LLM 请求次数：一个 agent session 内可有多轮模型/工具交互；transport retry、无效编辑重试、选择、反思、事后 judge 都不该误算成「没有成本」。选择和反思被明确记在这份 ledger 之外。通用 `_roll()` 只用 label 记录 rollout，没有把真实 executor 账单金额自动回填到 ledger；其 USD 字段也不能直接当作全系统真实费用。[S11][S16]

### 6.1 Parallel：一次 population update，宽度为 k

```text
                 base harness H0
                       │
                 k 次 reference
                       │
        ┌──────────────┴──────────────┐
     INV1 ... INVk                  DOM1 ... DOMk
        │                             │
        └──────── 2k 次 probe ─────────┘
                       │
             完整候选池与执行证据
                       │
          AGG1 ... AGGk（独立组合）
                       │
              k 次 final rollout
                       │
                匿名选择一个产物
```

每个组合器都从 base tree 出发，读取完整候选池；不是 AGG2 在 AGG1 上继续优化。`k` 扩大探索宽度，当前实现不是做 k 代 population evolution。代码实际线程宽度还有上限（默认 `PHASE_WIDTH=2`），所以「parallel」描述方法结构，不保证 2k 个进程同时跑满。[S12][S25]

### 6.2 Sequential：一轮轮改当前 harness，历史继续累积

```text
k 次 H0 reference，初始化证据
H0 → (INV1 || DOM1) → 两次 probe → Compose c1 → rollout
H1 → (INV2 || DOM2) → 两次 probe → Compose c2 → rollout
...
H(k-1) → (INVk || DOMk) → 两次 probe → Compose ck → rollout
选择范围：c1、c2、...、ck 的现有产物
```

下一轮继承上一轮 composed harness，且保留 base 与历史 candidate/rollout/decision 证据。第 t 轮的 INV 和 DOM 仍然是从同一当前 harness 出发的并发兄弟分支。初始 k 次 base 不会每轮再从头生成 k 次，因此总预算仍是 4k，而不是 4k²。[S12]

另一个容易忽略的实现：**sequential 最后可以选 c1，不一定交付 ck。** 选择池是所有 k 个 composed checkpoints；这允许避免后期退化，但也说明「深度为 k」不意味着最后一次一定更好。[S14]

### 6.3 同时跑两种模式，不能把共享 base 重复算作真实支出

同一 `run_task()` 同时请求 agg/seq 时，k 次 base 在两种模式之间共享，物理执行数为 `k + 3k + 3k = 7k`；每种模式自己的 ledger 仍显示 4k，便于公平描述各自算法预算。CLI 默认只跑 parallel，只有明确要求两种模式时才有这个共享。[S07][S20]

本次 k=2、两种模式、两个任务的 demo 正好是 `7 × 2 × 2 = 28` 次本地执行。改进器调用数为每任务每模式 `3k + 1 selector + 1 reflection = 8`，总计 32。它们是 deterministic editor calls，不是 32 次付费 LLM API 调用。

## 7. 选择与跨任务经验：哪里递归，哪里不递归

### 7.1 选择器只选择已有交付，不再写一份「融合答案」

`selection_pool()` 明确只放入 `agg1..aggk` 或 `c1..ck`。base、INV、DOM 的原始产物都不在正式选择池里；即使 base 看起来最好，selector 也不能直接选它。Compose 可以保留 base 内容，但那是以该 harness 重新执行生成的 final。[S14]

选择时先丢弃未完成/无可读产物的候选，再将候选映射为匿名 ID，给选择器任务需求与源材料。选择器不知道来源 harness，也不拿 benchmark score 排序。它返回某个 ID，框架映射回对应 run；不允许现场综合出新产物。`k=1` 则绕过选择会话，直接采用唯一合格 final，不能把它描述成经过多候选 judge 认证。

个别任务的交付是环境状态而非普通文件，adapter 可以声明 `state_deliverable`，在没有文件产物时提供自身轨迹作为选择证据。这个特例也提醒我们：「有可用证据」与「任务成功」是不同概念。[S14]

### 7.2 Reflection 必须看过程，而不只看最终选择

`reflect_adoption()` 构造更完整的证据包：原始 harness、采用的 harness、base 与候选执行、diff、各阶段决策、匿名 ID 映射。`_distil()` 要求经验至少包含三项：

- `cue`：什么情形适合使用这条经验；
- `lesson`：该怎样改变 harness/工作方法；
- `evidence`：哪些具体执行差异支持它。

还可保留 `confidence` 和逐项 edits。没有完整的 cue/lesson/evidence，或反思会话失败、返回 `NO_ENTRY` 时，不写入经验库。**程序检查的是这些字段和泄漏标记是否存在，不是形式化证明经验的因果真实性。**「证据支持」在很大程度上仍依赖反思模型的判断质量。[S15][S30]

### 7.3 下一个任务到底继承了什么

持久化结构大致为：

```text
library/<benchmark>/<budget-and-substrate-and-seed>/<mode>/<domain>/
  INDEX.md
  harnesses/<previous-task>/
    harness/                    被采用的完整可执行文件树
    EXPERIENCE.md               适用条件、经验、证据与采用理由
```

下一任务开始时，`schedule.run_task()` 再次调用 `adapter.base_harness()`。与此同时，改进器上下文带上之前任务的 library，允许它主动查看并迁移其中的组件。实际关系是：

```text
上一任务 final harness ─┐
上一任务反思经验 ──────┴→ library → 下一任务的 editor / composer
共同 base harness H0 ─────────────────→ 下一任务的初始 executor
```

`run_domain()` 按顺序处理同一领域任务，并用 `lib.retain(task_ids[:i-1])` 只保留当前任务之前的条目，防止恢复运行时误读「未来任务」经验。不同 mode/domain/benchmark/substrate/budget/seed 的库分开；不同 domain 可以并行。这里 substrate 指 oc/cx，不是模型 ID；模型切换由 CLI 身份检查拒绝静默复用，应另建运行根目录。[S03][S07][S15][S20]

默认实现没有在所有历史任务间跑一个向量检索器。它把当前 domain 的文件库复制到 context，给出 INDEX，改进 agent 自己选择读取哪些条目。因此这是**按领域隔离 + agent 文件浏览式复用**，不能想当然写成 embedding/RAG top-k 检索。

### 7.4 与几个相近概念的区别

| 概念 | 主要改变对象 | 与 Harness² 的关键区别 |
|---|---|---|
| 普通反思/自我纠错 | 当前回答或当前步骤 | Harness² 还修改可执行 harness，并再运行原任务 |
| 多次采样 + best-of-N | 多个答案 | Harness² 的候选先是多个执行配置，Compose 再生成配置并执行 |
| 外部经验记忆 | 给固定 agent 增加上下文 | 本项目经验也包含可执行脚本/skills 等，供改进器迁移 |
| 离线 harness 搜索 | 用训练/验证任务挑部署配置 | 本项目对正在遇到的任务做 task-local refinement，不要求该任务的标签反馈 |
| SFT / RL | 模型权重 | 本项目冻结模型，主要持久变化在文件库 |

上表是机制层面的对照，不是对所有同类工作的性能排名。论文比较组中的 **Meta-Harness** 是另一个方法名称，不要与本文泛称的 improvement harness 混为一谈。

## 8. 本次实跑：账本输入如何变成正确产物

本节不是凭空编的示例。我们在固定源码快照上审查后运行了官方 deterministic demo，执行真实 Python 脚本与真实调度/证据/选择/经验库流程，但用硬编码 `DemoProposer` 取代模型。[S27][S28]

### 8.1 第一个任务的原始输入

任务要求：把 `ledger.csv` 的所有带符号金额求和，输出保留两位小数的 `Total:`。

```csv
amount
100.25
-20.10
5.00
```

base 的 `scripts/summarize.py` 使用 Decimal，但对金额加了 `amount > 0` 过滤。结果为 `Total: 105.25`，把 -20.10 漏掉了。这个例子说明：**采用精确数值类型不等于业务规则正确**，问题是过滤条件，不是浮点误差。

### 8.2 Propose 到底改了哪里

demo editor 对 harness 中的脚本执行确定性替换：移除正数过滤。概念上变化如下：

```diff
- 只对 amount > 0 的金额求和
+ 对所有带符号 amount 求和
```

INV、DOM 以及各次组合在这个 toy 中执行相同的预写修改，因而不能用来证明 scope 分工的价值，也不能证明真实模型能自动找到问题。

### 8.3 Probe 与 final 的实际输出

| 任务 | 输入金额 | base 输出 | 候选/组合后输出 |
|---|---|---:|---:|
| ledger-1 | 100.25，-20.10，5.00 | 105.25 | 85.15 |
| ledger-2 | 35.50，-10.25，-5.00 | 35.50 | 20.25 |

每次 `DemoAdapter.execute()` 真正启动一个 Python 子进程，运行该次 harness 下的 `scripts/summarize.py`，写入 `answer.txt` 与包含 action/observation 的 `trajectory.md`。这不是只改 JSON 中的成功标记。[S27]

第一任务反思后，library 保存「有借贷/正负金额时保留符号」的 cue/lesson/evidence 和脚本文件树；第二任务的改进 context 中能看到 `ledger-1`。第二任务的 base 仍输出 35.50，这也直接验证了它**没有自动以第一任务修好的 harness 启动**。

但 demo 的 editor 并不靠读取 library 才学会修复，它无条件执行预写替换；selector 也直接从 CSV 算出期望和来找候选。因此本次能证明的是**状态传递和流程契约**，不是学习算法的效果、因果迁移或专业 benchmark 性能。

### 8.4 实际验证结果

- 运行环境：Python 3.12.14。
- 上游 `tests/test_workflow.py`：5 个测试，全部通过。
- `k=2`，parallel 与 sequential，两个任务：完成 2、失败 0。
- 本地执行计数：28；deterministic editor calls：32。
- 每个任务每种模式的 ledger：`R=8`、`P=6`。
- 再次运行同一 demo 目录：两个任务均 `skipped`，新增执行数 0、editor calls 0。
- 未安装 benchmark 依赖，未调用外部模型，未运行真实 benchmark 或外部 judge，未验证论文分数。

5 个回归测试覆盖了预算/恢复、完整证据与 library 前缀、无效编辑后反馈重试及成本累积、无有效经验时不入库、失败选择不被 judge。通过这些 focused tests，不代表所有底座集成、网络隔离、图像抽取、文件格式或并发情形已经全面验证。[S29]

## 9. 评测结果应该如何读

### 9.1 论文结果，不是本次复现结果

下面摘取论文 Table 1/2 的均值，用于理解量级。每格依次为「base → Harness² k=1 → parallel k=5」。最后一项由原表 k=1 均值加 `+ Recursion` 增量计算，**不是另行测得**；原表还报告三次运行的标准差，这里为简洁省略。[P1]

| Executor / 底座 | WorkBuddy Avg | JobBench | LAB Partial |
|---|---:|---:|---:|
| Gemini 3.5 Flash / OpenCode | 72.4 → 77.5 → 80.8 | 31.5 → 36.0 → 39.0 | 79.9 → 81.9 → 84.5 |
| Gemini 3.5 Flash / Codex | 74.5 → 77.5 → 80.4 | 33.9 → 35.3 → 39.5 | 71.6 → 75.6 → 78.5 |
| GLM-5 / OpenCode | 64.2 → 75.7 → 79.6 | 20.6 → 24.2 → 28.4 | 59.6 → 71.4 → 78.1 |
| GLM-5 / Codex | 64.4 → 75.3 → 78.3 | 23.6 → 24.8 → 27.9 | 58.1 → 65.9 → 70.7 |

这些数字支持「在论文测的四种配置上，相比各自 base 有收益」，不支持「所有情况下都比所有方法好」。例如 Gemini/Codex 的 LAB，Meta-Harness 是 76.5，而 Harness² k=1 为 75.6。

测试口径也不能混用：WorkBuddy 的 Code/Web/Office 测试任务为 40/35/25，Avg 是三个领域等权宏平均；JobBench 为 34 个测试任务；LAB 为 100 个 held-out 任务。不是用发布仓库能下载的全部任务数作为论文分母。各主表为三个打乱任务顺序的运行均值。[P1]（Appendix B）

### 9.2 有收益为什么仍不能直接说「便宜了」

论文 `#Steps` 指 executor 会话内的 model turns，不是整个 Propose–Probe–Compose 系统总调用数或总 token。即使最终 harness 让某次任务执行步数下降，它前面仍花了多次 reference/probe、editor/compose、selection 的预算。完整成本比较应同时报告：

- 任务执行与改进侧的实际输入/输出 token；
- 重试和失败带来的额外调用；
- selector、reflector、外部 judge 是否计入；
- wall time、并发数、机器开销和 API 单价；
- 最终交付是否达到同等质量。

因此本文不根据 `R=4k` 或 `#Steps` 推算美元、毫秒或加速比。论文的总 token 研究与单个 executor 的 steps 是不同口径，不能互相替代。[P1][S16]

### 9.3 消融可以支持什么，不能支持什么

论文 Table 3 在 Gemini/OpenCode、k=1 下报告 WorkBuddy/JobBench：full 为 77.5/36.0，去 library 为 74.4/34.2，去 scope 分工为 74.6/32.3，去 composition 为 75.9/34.1。它支持这些设计在该设定中的经验价值，但不是各组件可相加、放之四海而皆准的因果贡献。[P1]

进一步阅读时应注意实验改变了什么：无 scoping 仍保持两个 generic candidates；无 composition 改成从两候选中选择。已优化 harness 上删除 skills/scripts 的实验也不同于从一开始禁用该组件后重新优化。允许看未来任务经验的离线重跑，更不能称为严格在线系统的实时增益。

### 9.4 改进阶段无评分反馈，外部评测仍然存在

默认 selected-run 评分路径由 `core.judge` 处理，只评分成功完成 selection 的 run；另有 `harness2-judge --bases`，会直接调用 `adapter.score()` 对记录的 reference runs 做事后基线评分，不要求它们经过 final selection。两条评分路径都不把分数回流到在线改进。`harness2-judge` 与 `harness2-report` 是运行结束后另行执行的 CLI；评分器失败会留下 ungraded，而不是自动记成 0。[S04][S22][S23][S31]

因此「无标签在线改进」是信息访问约束，不是「没有测量器」「不需要 benchmark rubric」或「已经证明结果正确」。尤其公共 benchmark 的答案/判分标准可能在线可得，实验必须报告真实网络隔离条件。

## 10. 缓存、失败、恢复与复现陷阱

### 10.1 防止目录存在就被当成完成

`runmark.py` 用 `.adapt_complete` 记录完成与 harness 指纹。JobBench 还检查产物指纹；仅有输出文件或一段报错 trajectory 不代表一次正常 rollout。外层 runner 的 wall-timeout 不写完成标记，后续可重跑。但 OpenCode 内层超时若已产生产物，runner 会先依据非空 output 判断交付，仍可能返回成功并被写入完成标记。因此 marker 不等于「没有超时」或「产物完整」。[S17][S18][S21]

summary 层 `_resume()` 会核对所需模式、selection 是否成功、采用的执行目录与输出是否仍在、harness 是否存在且指纹一致。上游单测实际覆盖了采用的 harness 被改动、产物被删除之后不能静默 resume 的情形。[S07][S29]

但仍要谨慎理解缓存边界：`fingerprint(None)` 把 base rollout 记为字面值 `base`，不是动态 hash 当前 base 文件；CLI 配置身份检查也不能替代完整的源码、依赖和数据版本记录。**更换上游代码/基础 harness 后应使用新的运行根目录**，不要因为模型 ID 一样就复用旧缓存。[S17]

### 10.2 改进失败与选择失败不同

| 失败位置 | 主要处理 | 分析时的含义 |
|---|---|---|
| 构建证据失败 | stage 使用 fallback tree 并记 error | 未真正完成该次改进 |
| 编辑文件不合法 | 给错误反馈再尝试；失败则继承当前 tree | 阶段数仍可满足预算，不代表编辑成功 |
| rollout 抛错/无目录 | 生成失败 RunView | 不应该拿失败结果当有效候选 |
| selector 失败 | 记录 fallback 候选，但 `sel.ok` 仍为 false | 不算成功采用，不入库，不被正常 judge |
| reflector 失败/无完整经验 | 不写新经验条目 | 当前采用可能有效，但没有跨任务学习条目 |
| 外部 judge 失败 | ungraded | 不是 0 分，也不代表通过 |

另一个字段陷阱是 `kept_base`：在 sequential 后续轮次中，它表示与该轮 fallback/current harness 相比没有变化，不必然表示退回最初 H0。

特别是 `ModeResult.ok` 与所有 stage 是否都成功改善不是一回事；应检查 `fell_back`、stage errors、selection flags 和完整分母。`R=4k/P=3k` 只是预算结构的校验，不是任务质量证明。[S11][S14][S15][S16][S22]

### 10.3 汇总分母可能让表面分数变好

`core.report.summarise()` 对有评分的任务取平均，未评分项计入 `n_ungraded` 后排除。失败/未完成任务不能悄悄从实验报告里消失：至少同时列出计划任务总数、成功采用数、成功评分数、失败原因与覆盖率。否则「仅成功评分任务平均分」可能掩盖系统可靠性下降。[S23]

### 10.4 seed 只控制任务顺序

`--seed` 决定任务流排序与命名空间，不控制模型抽样、服务端随机性或所有工具环境。由于 library 按流积累，顺序会影响后续任务，这也是三次 shuffled stream 要报告标准差的原因。固定 seed 不等于完整可复现。[S03][S07]

### 10.5 最实质的污染风险来自网络

官方实验指南直接披露：验证发布版时，某次 JobBench Codex rollout 在没有被要求的情况下下载了自己任务的 rubric。仓库本身没有限制 executor 网络，且 Codex 使用禁用 sandbox 的启动设置。[S24][S06]

因此本报告中的路径过滤、grader marker、独立会话目录，都**不能**被宣传为「已经杜绝泄漏」。公平复现实验必须在执行环境层设置 egress 策略，并把政策、被拦截目标、允许的任务所需网络范围一起记录。代码里的行为 guardrails 不等于这个环境安全边界。

## 11. 如何复现、扩展与部署

### 11.1 推荐的第一次阅读顺序

1. `demo.py`：先看懂什么是 task、harness、rollout、deliverable。
2. `core/spec.py`：理解 method / benchmark / substrate 的接口边界。
3. `core/recursion.py`：重点读 `_run_stage()`、`run_parallel()`、`run_sequential()`。
4. `core/selection.py` 与 `core/library.py`：确认选择池和跨任务状态。
5. `core/schedule.py`：理解任务流、共享 base、恢复与命名空间。
6. `benchmarks/jobbench.py` 与真实 runner：再向下追到外部 agent 执行。
7. `core/context.py`、`tree.py`、`proposer.py`：最后补证据契约、校验和会话细节。

不要一上来只读 1,000 多行 prompts，把方法误解为一个巨型提示词；也不要只看架构图就跳过 run 与 resume 的真实路径。

### 11.2 先跑本地 deterministic demo

在已 checkout 本文固定 SHA 的仓库根目录、Python 3.12 环境中，以下流程不要求安装模型 SDK 或调用模型。输出写到新临时目录；执行前仍应像本次一样审查所运行源码。

```bash
REPO="$PWD"
test ! -e .env || { echo "请在没有 .env 的干净源码副本中运行此 toy 验证。"; exit 1; }
PYTHON="$(command -v python3)"
OUT="$(mktemp -d)"
mkdir -p "$OUT/home" "$OUT/tmp"

env -i PATH=/usr/bin:/bin HOME="$OUT/home" TMPDIR="$OUT/tmp" \
  PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 LC_ALL=C.UTF-8 \
  "$PYTHON" -B -m unittest discover -s tests -p 'test_*.py' -v

env -i PATH=/usr/bin:/bin HOME="$OUT/home" TMPDIR="$OUT/tmp" \
  PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 LC_ALL=C.UTF-8 \
  "$PYTHON" -B -m harness2.cli.demo --output "$OUT/demo" --k 2 --mode both

cat "$OUT/demo/demo.json"
```

环境清空与临时 HOME 可避免把模型凭证环境变量继承给 toy 子进程，但它们不是网络防火墙。demo 的安全性还来自已审查的调用路径只执行本地确定性脚本；不要把同样命令模板直接用于任意新下载的 agent 代码。

要验证恢复，只需再运行同一条 demo 命令，保持 `--output` 不变。之后可检查 `runs/demo-oc-agg2/<task>/summary.json`、对应 selection、harnesses 与 `library/k2/`。注意：即使仅运行 sequential，scheduler 的共享 summary 也放在 `agg` 命名空间；这是路径约定，不是跑错了模式。[S20][S28]

### 11.3 真正复现论文前的最小检查单

- 固定本仓 SHA、外部依赖 SHA、Python/Node/Bun/Codex 版本与数据 revision。
- 记录实际测试任务集合、split、domain、顺序 seed，不拿全部公开数据量冒充论文分母。
- 记录 executor、improver、selector、judge 的准确 model ID 与服务配置。
- 用独立无敏感文件的环境，最小权限凭证与网络策略；不要把现成云主机的全部环境变量/账号暴露给 agent。
- 先 `--dry-run`，再极小的单领域单任务运行；确认调用成本后再扩大。
- 给不同配置新的 `HARNESS2_RUNS_DIR`；保存全部失败和 ungraded，不只保存成功案例。
- 明确选择/反思/重试/评测是否计入成本，分别统计 token、wall time 与任务质量。

JobBench 的数据 revision 在下载时解析，不是仅由本仓 SHA 固定；setup 会在数据目录记录 `.harness2-download.json`。真实 benchmark 有额外系统依赖，LAB 涉及文档工具，WorkBuddy 涉及 Docker；本次没有安装和运行它们。[S24][S26]

### 11.4 接入自己的任务：先实现 adapter，不先改递归

最小原型应实现 `BenchmarkAdapter` 的输入、执行、轨迹、产物、base harness 和校验接口。先用确定性 mock/本地脚本证明这些契约可靠，再换真实 executor。`score()` 可服务离线验证，但不要把它悄悄接回在线改进，改变了研究问题而仍声称是同一设定。[S04]

对于一个内部报表 agent，合理试验可以是：固定一组不含敏感数据的历史材料，比较 base 与不同 harness 下的文件正确性、生成过程、失败率和成本。不要从 toy 算术成功直接跳到财务、医疗或法律高风险自动决策。

### 11.5 工程判断：适合什么，不适合什么

**较适合探索：** 单任务价值较高、能容忍多次执行、工具与工作流程是主要瓶颈、可以获得丰富过程证据、后续会出现相近领域任务的场景。

**应谨慎：** 极低时延聊天；每次执行都有不可逆外部动作；只能承受一次调用；无隔离环境；无法保留/审查经验中的敏感事实；任务质量几乎无法从现有材料自查的场景。

后一点尤其重要：Probe 会重做任务，若任务包含真实发送邮件、购买、删库或生产部署，重复执行不再只是研究预算问题。生产设计应先把这些副作用改造成可回放的沙箱/模拟接口，执行权限与审核由外部受控系统掌握。这是基于源码执行结构的工程建议，不是本仓已经实现的能力。

## 12. 最容易读错的十句话

| 容易误读 | 更准确的说法 |
|---|---|
| 「它会训练出更强模型」 | 冻结模型；改变 harness 文件与改进经验 |
| 「只是自动改 prompt」 | 还可改 skills、scripts、subagents、plugin/hooks |
| 「递归就是把上一题 final 接到下一题」 | 每题从共同 base 开始；经验影响下一题的改进器 |
| 「k 是模型调用次数」 | k 是方法预算；一次 session 内可有多轮调用 |
| 「parallel 是 k 代进化」 | 当前实现是一轮 population update，宽度 k |
| 「sequential 必须选最后一轮」 | 从 c1..ck 的 composed 产物中选一个 |
| 「selector 会在所有结果中选最好」 | 正式池仅 composed finals，且没有真值分数 |
| 「无 ground truth 就完全不用 judge」 | 在线改进不消费评分；事后评测仍有 judge |
| 「seed 一样就完全可复现」 | seed 只控制任务顺序；模型/数据/环境另需固定 |
| 「demo 通过证明能自我进化」 | demo 使用预写编辑器，证明流程而非智能改进效果 |

## 13. 验证记录与源码阅读索引

### 本次完成与未完成

| 项目 | 状态 |
|---|---|
| 上游仓库与固定 SHA 核验 | 已完成 |
| CLI → scheduler → recursion → adapter → runner 静态追踪 | 已完成 |
| 官方 workflow 单测 | 5/5 通过 |
| 双模式 k=2 deterministic demo | 2 任务完成，0 失败 |
| 同目录 resume | 2 任务 skipped，0 新执行，0 editor calls |
| 真实模型/benchmark/judge | 未运行 |
| 论文分数与成本复现 | 未完成，不作复现声明 |
| OpenCode/Codex 安全隔离与生产适用性认证 | 未做，也不能由 toy 测试推出 |

本报告只增加文档，不修改上游算法；没有把完整运行输出、临时路径或模型凭证提交到仓库。另附 [机器可读验证摘要](validation-summary.json)，便于与将来的运行对照。

### 源码索引

| 想回答的问题 | 建议先读 |
|---|---|
| 命令从哪里进入？ | [CLI `_main()`][S03] |
| 各模块如何对接？ | [协议与 dataclass][S04] |
| 八类组件映射是什么？ | [OpenCode surface][S05]、[Codex surface][S06] |
| 每题怎么调度/恢复？ | [schedule][S07] |
| 底座为何需要 benchmark launcher？ | [Substrate protocol][S08] |
| 编辑器怎样拿回改后的文件？ | [Proposer.call][S09]、[`_run_stage`][S11] |
| 合法 harness 如何校验？ | [tree validators][S10] |
| 两种递归真正差在哪？ | [parallel/sequential 实现][S12] |
| 证据怎样组织和过滤？ | [context][S13] |
| 为什么 seq 可选 earlier checkpoint？ | [selection_pool][S14] |
| 下一个任务怎么使用经验？ | [reflection][S30]、[Library][S15] |
| 4k/3k 到底算什么？ | [Ledger][S16] |
| 什么叫可恢复的完成？ | [runmark][S17] |
| 一次真实 JobBench 如何跑？ | [adapter][S18]、[render][S19]、[runner][S21] |
| 评分是否进入改进循环？ | [judge][S22]、[report][S23] |
| 为什么必须重视污染风险？ | [官方实验指南][S24] |
| 本次 toy 做了什么？ | [demo][S27]、[demo CLI][S28]、[tests][S29] |

### 来源与归属

论文：Ruiyao Xu、Yanfei Chen、Zhongying CuiZhu、Bhavana Dalvi Mishra、Yifei Ming、Han Yu、Rujun Han、Chen-Yu Lee、Tomas Pfister，*Harness²: Recursive Agent Harnessing for an Open World*，arXiv:2610.14224v1，2026。[论文登记页](https://arxiv.org/abs/2610.14224v1) 的许可链接为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本文对其机制与实验数据进行了中文概述、筛选与重新组织，并明确标注计算值；不暗示作者认可本分析。

代码：Google LLC，Apache-2.0；第三方代码和 benchmark 另有各自许可，见上游 `LICENSE`、`third_party/README.md` 与依赖说明。下面所有 S 开头的引用均为同一固定 commit 的源码/文档，不是未来 main 的保证。

[P1]: https://arxiv.org/html/2610.14224v1 "论文全文：方法、主表及附录"
[R1]: https://github.com/google-research/harness2 "官方仓库"
[S01]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/README.md "项目概览与安全声明"
[S02]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/pyproject.toml#L1-L44 "包版本、Python 约束与 CLI"
[S03]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/cli/run_sweep.py#L120-L239 "主入口、preflight 与运行身份"
[S04]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/core/spec.py#L33-L260 "ChannelSpec、BenchSpec、RunView 与协议"
[S05]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/substrate/opencode.py#L42-L88 "OpenCode editable surface"
[S06]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/substrate/codex.py#L39-L198 "Codex surface、安装与执行参考"
[S07]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/core/schedule.py#L350-L733 "恢复、任务执行、选择反思与 domain 流"
[S08]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/substrate/base.py#L26-L80 "Install、RunRequest 与底座协议"
[S09]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/core/proposer.py#L90-L208 "改进会话、重试、harvest、审计"
[S10]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/core/tree.py#L15-L399 "文件树、路径范围、自动触发与校验"
[S11]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/core/recursion.py#L108-L300 "接受编辑、stage 与 rollout"
[S12]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/core/recursion.py#L343-L578 "parallel 与 sequential"
[S13]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/core/context.py#L105-L546 "证据树、材料复制、泄漏 tripwire 与 mission"
[S14]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/core/selection.py#L71-L399 "候选池、匿名选择与 k=1"
[S15]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/core/library.py#L43-L181 "经验有效性、保留前缀、写入与索引"
[S16]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/core/ledger.py#L15-L132 "逻辑预算与实际 attempts"
[S17]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/core/runmark.py#L15-L115 "完成标记与指纹"
[S18]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/benchmarks/jobbench.py#L502-L850 "JobBench 配置、执行、runner 启动"
[S19]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/benchmarks/jobbench.py#L422-L499 "OpenCode 格式渲染"
[S20]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/cli/_common.py#L30-L139 "默认模式、路径约定与 adapter 构造"
[S21]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/benchmarks/jb_opencode_runner.py#L117-L271 "实际 OpenCode runner"
[S22]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/core/judge.py#L33-L155 "事后评分与 ungraded"
[S23]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/core/report.py#L15-L165 "汇总分母与警告"
[S24]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/docs/EXPERIMENTS.md#L1-L122 "预算、任务选择、污染与复现"
[S25]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/config.py#L99-L189 "模型变量、依赖与并发默认值"
[S26]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/docs/SETUP.md "真实运行环境要求"
[S27]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/demo.py#L30-L163 "确定性账本任务、执行器与预写 editor"
[S28]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/cli/demo.py#L27-L58 "demo 命令入口与计数"
[S29]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/tests/test_workflow.py#L31-L225 "五个 workflow 回归测试"
[S30]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/core/selection.py#L402-L505 "adoption reflection 与经验提炼"

[S31]: https://github.com/google-research/harness2/blob/1cfe254e8d25b15f37048c3730134686abe60335/harness2/cli/judge.py#L108-L190 "事后 base 评分与 --bases 分支"
