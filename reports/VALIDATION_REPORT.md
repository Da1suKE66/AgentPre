# AgentPre：Articraft 微波炉 + Franka 远端验证报告

- 验证日期：2026-07-14--15（Asia/Shanghai）
- 验证主机：`lsh-stable32314`
- 源码目录：`/workspace/liluchen/AgentPre`
- 缓存根目录：`/cache/liluchen/agentpre`

## 1. 结论与提交边界

本报告必须区分两代证据，不能把旧版任务成功门槛等同于运动连续性：

- **旧版 124 帧结果**：真实 Articraft 与仓库 fixture 各执行过一次
  `kinematic` 和一次 `physics_assisted`。四次 clean run 的 shell、Git clean、IK、
  位姿/门角/抓取漂移、碰撞、静态关节范围及有限值门槛均通过；但旧门槛没有检查
  帧间关节步长、速度、加速度或 jerk，因此这些结果只能证明旧版位姿与任务门槛，
  **不能证明轨迹连续、可执行或满足 URDF 动态限制**。
- **最终 1336 帧 kinematic 结果**：真实 Articraft 六阶段轨迹已通过；nominal
  `t=0` 到首帧也纳入速度、加速度和 jerk 审计。最大单关节步长为
  `0.020829987 rad`、最大速度限制比为 `0.478850270`，旧 UI 89→90 对应位置的
  `Δq L2` 已降至 `0.026979270 rad`。完整指标见 §5.2。
- **最终 1336 帧 physics 结果**：完整 cold-start Newton `physics_assisted` run
  `articraft_continuity_physics_20260714_04` 已以 shell exit=0、
  `success=true / run_status=success` 完成，20/20 acceptance gates 全部通过；碰撞、
  关节越界与非有限值均为 0。完整动态指标与边界见 §5.6。

> [!IMPORTANT]
> 最终通过结论只绑定 §5.2 的 1336 帧 kinematic `_06` 和 §5.6 的完整 cold-start
> physics `_04`。早期 `_02`、`_03` 以及 repo 外断点 A/B 都保留为失败与根因证据，
> 不能单独作为最终通过结果。

旧版四次 124 帧仿真严格绑定源码提交：

```text
a430dea08b04b6aa701dcb3d7498b39d923d0fb0
```

仿真完成后发现并修复了 daemon 停止时 EXIT trap 读取已离开作用域局部变量的
问题。该修复提交为：

```text
93e3b4797651defe2e6905bf867e14d3ee618731
```

`a430dea..93e3b47` 只修改 `scripts/sync_daemon.sh` 和
`tests/test_sync_daemon.py`，没有修改资产、配置或仿真代码。这两个提交和后文旧
产物哈希只绑定旧版 124 帧历史快照，不能作为最终 1336 帧连续性修复的提交或产物
证明。最终报告提交不能在正文中自包含自身 Git hash；最终 GitHub HEAD 由交付时的
`ls-remote` 结果给出。

## 2. 环境与 CPU-only 边界

```text
Conda 环境     /cache/liluchen/agentpre/envs/agentpre-conda
运行产物根目录 /cache/liluchen/agentpre/outputs
Python         3.11.15
NumPy          2.4.6
Newton         1.3.0
Warp           1.14.0
设备           cpu
线程           OMP/OPENBLAS/MKL/NUMEXPR = 1
Git remote     ssh://git@ssh.github.com:443/Da1suKE66/AgentPre.git
```

`scripts/env.sh` 设置 `CUDA_VISIBLE_DEVICES=""`。Warp 启动时会打印没有 CUDA
driver 的探测信息，但列出的唯一运行设备为 `cpu`；旧版四次验证没有申请或占用
GPU。

## 3. 真实 Articraft 资产证据链

### 3.1 固定来源

| 字段 | 值 |
|---|---|
| Record / revision | `rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707` / `rev_000001` |
| Articraft repo / commit | `https://github.com/mattzh72/articraft` / `59eb5e0ed72a734111012b43f881423b15d4931d` |
| Articraft license | Apache-2.0 |
| Data repo / commit | `https://github.com/mattzh72/articraft-data` / `0cdcaa49f5571e9b4df04476c7f09587ee3ab7bd` |
| Data license | CC-BY-4.0 |
| Runtime manifest | `/cache/liluchen/agentpre/assets/articraft/rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707.manifest.json` |
| Manifest SHA-256 | `baf4220b98f845cb622f7dde72914c7326a8d5e4223a0afd21dbe96c44885863` |

官方 strict compile 为 `success`、full compile/full validation：failures=0、signal
warnings=0、notes=2。唯一几何 overlap 是 `cabinet/hinge_pin` 与
`door/hinge_barrel` 的有理由铰链嵌套；两条 signal 都是非阻断的
`NOTE_ALLOWED_OVERLAP`。源 compile report 为 2105 bytes，SHA-256：

```text
479e7c2492bc9597020b81af506b7e680610456cfe24b57afa7b00b8ddcc6e5b
```

### 3.2 确定性惯性后处理

原始 Articraft URDF 缺少可供动态仿真的完整惯性。本项目以结构化规范进行一次
确定性注入，并在 materialization 前后校验来源、link 集合、物理可行性、XML
值、sidecar 与哈希：

| 证据 | SHA-256 | 大小 |
|---|---|---:|
| 惯性注入前 URDF | `03f6aa1366ddbf0b740f1e051bfb0b8f673c9cf7bc293ad37f6ac60289beff36` | — |
| `inertials.json` | `b6049d1f65dabd7ad5d9305f53362f7f7e4a4880407df0bfa0c96f2d69276057` | 3179 bytes |
| completion sidecar | `e68de7679048484ed5b841e6224c90b5dfcf8e4c2e638189ec7b21b978da0f73` | 704 bytes |
| 惯性注入后 URDF | `4c4676c1df02a9525bf15baca51786b54fe1d503e1797b38b566291786ee3917` | 22456 bytes |

注入 link 为 `cabinet`、`door`、`selector_knob_0`、`selector_knob_1`、
`turntable`。质量按 `max(300 kg/m³ × collision AABB 体积, 0.02 kg)`，COM
取联合 collision AABB 中心，惯量按均匀实心长方体计算。这些是明确标注的
simulation proxy inertials，不是制造商实测质量，也不是 CAD 体积分结果。

### 3.3 Runtime 文件完整性

最终 URDF 有 5 links、4 joints、8 次 mesh 引用，无缺失文件；run-time inspector
为 `ok=true`、errors=0、warnings=0。三个唯一 mesh 与现场哈希如下：

| Mesh | 引用次数 | 大小 | SHA-256 |
|---|---:|---:|---|
| `selector_knob.obj` | 4 | 614836 | `87b577ab32335d3c825f2ce7b823f60916f0f54846378142b5a0e9b9d892a23d` |
| `turntable_clip_ring.obj` | 2 | 30915 | `773d0b35a2a627d58ab558794e5e232dbb154b9914fd85ba75f8ccf45b006b6d` |
| `turntable_outer_rim.obj` | 2 | 30915 | `a938c1960c3f4a23e16827a9920441f1d9e3e0cf80dd3188eda0555c5889d937` |

URDF 加三个唯一 mesh 共 4 个 runtime 文件、699122 bytes。旧版 Articraft 两次 run
都在后端初始化前校验 expected/observed URDF SHA，均为 `4c4676…3917`。配置的
`pull_grip_center` frame 被直接选中，`used_geometry_fallback=false`。

## 4. Franka 与 fixture 资产身份

Franka 来源为 Isaac Sim extension
`isaacsim.asset.importer.urdf-2.3.10+106.4.0`，license 为 Apache-2.0。
稳定缓存 URDF 与 bootstrap 文件逐字节一致：

```text
URDF SHA-256  ad9f5298a4d1a375cf16824b0de4f0d1c7cc446597964b80aa639ca830e998a1
Manifest      0a454b358e00ba9ee5419ea73b7a57eebf8ff0273b93578bd80b45066165b437
```

URDF 有 22 次 mesh 引用、20 个唯一 mesh；20/20 文件现场哈希均与 manifest
一致，总计 10633539 bytes。旧版四次 run 的 expected/observed Franka URDF SHA
全部一致。

仓库 fixture 只作为回归资产，不替代真实 Articraft 结果。其 URDF SHA-256 为：

```text
d6ba39f326d52a02efe6c4292accc8503e32c3a19a5462a90e564cddf52177a1
```

## 5. 旧版问题诊断与新版 1336 帧验证

### 5.1 旧版四次 clean run（124 帧；历史证据）

旧版所有 run 使用 seed `20260714` 和五阶段 124 帧轨迹：pregrasp 12、approach
12、close 8、actuate 80、retreat 12。transcript 与 `resolved_config.json` 均确认
源码为 `a430dea…0fb0`、运行前/运行时 Git clean。

| 资产 / 模式 | 输出目录 | shell exit | IK | TCP 中位误差：位置 / 姿态 | 最终门角 / 误差 | 最大 grasp 漂移：位置 / 姿态 | 碰撞 / 越界 / 非有限值 | 历史结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Articraft / kinematic | `articraft_kinematic_0001` | 0 | 124/124 | 0.000611533 m / 0.083526° | 65.000000° / 0° | 0.003256837 m / 0.176567° | 0 / 0 / 无 | 旧门槛 PASS；连续性未验证 |
| Articraft / physics | `articraft_physics_0002` | 0 | 参考 124/124 | 0.001315410 m / 0.171639° | 65.728413° / 0.728413° | 0.006273500 m / 0.756948° | 0 / 0 / 无 | 旧门槛 PASS；连续性未验证 |
| Fixture / kinematic | `fixture_kinematic_0001` | 0 | 124/124 | 0.000614416 m / 0.084453° | 65.000000° / 0° | 0.003463282 m / 0.183908° | 0 / 0 / 无 | 旧门槛 PASS；连续性未验证 |
| Fixture / physics | `fixture_physics_0001` | 0 | 参考 124/124 | 0.001692495 m / 0.151787° | 65.377328° / 0.377328° | 0.006273053 m / 0.797747° | 0 / 0 / 无 | 旧门槛 PASS；连续性未验证 |

旧版共同门槛为：IK success rate >= 0.95；TCP 位置/姿态误差中位数分别
<= 0.02 m / 10°；最终门角误差 <= 3°；最大抓取位置/姿态漂移
<= 0.015 m / 7.5°；碰撞帧比例、静态关节越界和非有限值均为 0。这些门槛没有
速度、加速度、jerk 或帧间连续性检查。Fixture physics 的最大 TCP 位置误差为
0.021014188 m，但 2 cm gate 明确检查中位数 0.001692495 m；本报告不声称所有帧
位置误差都低于 2 cm。

对应旧版 shell transcript：

| Run | Started UTC | Finished UTC | 耗时 | Exit |
|---|---|---|---:|---:|
| Articraft kinematic | 2026-07-14T07:29:06Z | 2026-07-14T07:29:19Z | 13 s | 0 |
| Articraft physics | 2026-07-14T07:57:37Z | 2026-07-14T08:05:21Z | 7 min 44 s | 0 |
| Fixture kinematic | 2026-07-14T07:30:18Z | 2026-07-14T07:30:28Z | 10 s | 0 |
| Fixture physics | 2026-07-14T07:30:43Z | 2026-07-14T07:31:58Z | 1 min 15 s | 0 |

#### 旧 UI 89→90 跳变

旧 UI 的 89→90 帧对应 0-based rollout 的 88→89 行。该步测得：

```text
Δq L2                         2.017 rad
最大单关节 |Δq|               1.416 rad
按 60 Hz 隐含的最大关节速度    84.96 rad/s
URDF 对应速度上限              2.61 rad/s
隐含速度 / URDF 上限           32.6x
```

因此这不是正常插值误差，也不能由“旧 acceptance 已通过”来掩盖。Franka 手臂是
7DoF 冗余机构；旧 IK 在相近末端位姿之间切换到了另一组关节解，即发生 IK 分支/
零空间构型切换。旧求解链只有位姿目标、静态关节范围和有限的 warm start，没有把
与上一帧的关节距离作为稳定目标，也没有硬性的速度、加速度、jerk 时序约束和对应
acceptance gate。跳变已经存在于 IK 关节命令中，不是物理引擎在该帧凭空移动了
全部关节。

### 5.2 最终 Articraft kinematic（1336 帧）

正式产物 `articraft_continuity_kinematic_20260714_06` 使用 seed `20260714`、
`dt=0.0166666667 s` 和六阶段 1336 帧轨迹：pregrasp 224、approach 72、close
96、actuate 800、release 48、retreat 96。release 保持目标门角、handle 与 TCP
不动，只按 quintic timing 打开夹爪；retreat 才撤离。因此 grasp drift 在 close、
actuate、release 共 944 帧上审计。close 从 48 延长到 96 帧不是放宽 finger gate，
而是降低真实接触速度；measured finger jerk 仍必须通过 30 m/s³ 门槛。

相较失败的 808 帧配置，最终配置保持抓取位置偏置不变，围绕 gripper local
closing axis 增加 `+15°` roll：

```text
orientation_wxyz = [0.5609855268, -0.4304593346,
                    -0.5609855268, -0.4304593346]
```

这不是放宽门槛，而是改变冗余 7DoF 的可行构型，使开门后段 q4 不再贴住控制/
硬限位。`metrics.json` 为 `run_status=success`、`success=true`，1336/1336 个 IK
waypoint 均通过独立 FK 验收：

| 指标 | 最终 1336 帧结果 |
|---|---:|
| IK success | 1336/1336（1.0） |
| TCP 位置误差：中位数 / 最大值 | 0.000481065 / 0.001317722 m |
| TCP 姿态误差：中位数 / 最大值 | 0.06798453° / 0.17339280° |
| 全程最大单关节 `|Δq|` | 0.020829987 rad（arrival frame 83，joint 7） |
| 全程最大 `Δq L2` | 0.027021299 rad（arrival frame 87） |
| 最大关节速度 | 1.249799204 rad/s（frame 83，joint 7） |
| 最大速度限制比 | 0.478850270（frame 83，joint 7） |
| 最大加速度 | 5.543589570 rad/s²（frame 1192，joint 5） |
| 最大 jerk | 338.6921862 rad/s³（frame 1192，joint 5） |
| q4 最小值 / frame | -2.962478399 rad / 1254 |
| q4 到 URDF hard lower 的最小余量 | 0.109321601 rad |
| physics reserve 全局最小余量 | 0.095238111 rad（frame 1333，joint 2 lower） |
| 最终门角 / 误差 | 65° / 0° |
| 最大 grasp 位置 / 姿态漂移 | 0.000571081 m / 0.07414484° |
| 碰撞帧 / 关节越界 / 非有限值 | 0 / 0 / 无 |
| UI 89→90（rollout rows 88→89）`Δq L2` | 0.026979270 rad |
| UI 89→90 最大单关节 `|Δq|` | 0.020550936 rad（joint 7） |

速度、加速度和 jerk 的 9352 个预期值（1336×7）全部有限。最大速度比
`0.478850270 < 1`、最大加速度 `5.543589570 < 7.5 rad/s²`、最大 jerk
`338.6921862 < 450 rad/s³`；位姿、门角、抓取漂移、碰撞、关节范围及有限值 gate
也全部通过。physics preflight 独立审计真实 nominal 初始状态和 1336 帧 reference
共 1337 个样本；最小硬限位余量为 `0.095238111 rad`，control-bound touch、initial
reserve violation 和 trajectory reserve violation 均为 0，满足 `0.05 rad` 要求。

旧 UI 的 89→90 对应 0-based rollout rows 88→89。最终同两行七个关节的差为：

```text
Δq = [ 0.001207866, -0.000628583, -0.003217869,  0.003610849,
       0.014626384, -0.008146286, -0.020550936 ] rad
Δq L2                         0.026979270 rad
最大单关节 |Δq|               0.020550936 rad（joint 7）
```

相同 UI 帧号在旧版位于 actuate，在新版位于 pregrasp，故不代表相同任务进度；
保留它是为了直接复核原始投诉。旧故障位置的 `Δq L2` 从约 `2.017 rad` 降至
`0.026979270 rad`，最大单关节步长从约 `1.416 rad` 降至 `0.020550936 rad`。
全局任意相邻帧 `Δq L2` 也不超过 `0.027021299 rad`，旧 IK 分支/零空间切换在
完整 1336 帧轨迹中未重现。

非静止目标继续由 Newton LM/analytic IK 顺序求解。每帧以先前有限且已投影的
关节状态 warm start，并用 temporal joint-reference objective 约束上一帧 `q`。
对 close/release 等数值相同的 Cartesian target，后端不再重复运行带 nominal
posture 项的 LM，而是让同一硬运动投影器先合规制动，再在速度和加速度归零后精确
保持 float32-realized `q`，因此不会沿 Franka 零空间逐帧漂移。

六阶段均采用 `s(t)=10t^3-15t^4+6t^5`；存储帧是右端点，nominal `t=0` 到首帧
也进入速度、加速度和 jerk 审计。有限 IK candidate 被投影到 float32 可表示的
控制位置、URDF 速度、`7.5 rad/s²` 加速度和 `450 rad/s³` jerk 单步交集。最终
正式 run 的 raw candidate 已全部位于交集内：motion/velocity projection frame、
raw velocity violation 与各类 trigger count 均为 0，`max_abs_q_correction_rad=0`。

### 5.3 808 帧 physics 失败复验与根因

第一次包含完整连续性、fixed-grasp transaction 和 measured dynamics 门槛的长程
run 为 `articraft_continuity_physics_20260714_02`。它真实完成 808 帧 Newton
physics，但 shell exit=3、`run_status=acceptance_failed`，因此不能计为通过。

通过项包括：最终门角 `64.77588°`（误差约 `0.2241°`）；collision frame=0；
TCP 中位位置/姿态误差 `0.0088686 m / 0.99098°`；最大 grasp 位置/姿态漂移
`0.009418 m / 1.057°`；capture pose/twist gate、captured anchor coincidence、
fixed-joint enable/disable transaction/readback 以及全部 finger measured-dynamics gate
均通过。

失败项是 q4 在 actuation 后段饱和。参考命令从 frame 637 起贴在控制下边界
`-3.0517998 rad`（URDF hard lower `-3.0718 rad` 加 0.02 rad control margin）；
抓取负载把 measured q4 推到 hard lower 附近，产生 39 个 measured joint-limit
violation frames，最大 raw overshoot `2.717e-5 rad`。权威 Newton `joint_qd` 的最大
arm acceleration/jerk 分别为 `22.077 rad/s²` 和 `2096.443 rad/s³`，峰值位于
frame 669 的 q4，均超过 `7.5 / 450` 门槛。

另一个由相邻 reconstructed `joint_q` 差分得到的 q3 峰值为
`8.716 rad/s² / 769.981 rad/s³`。frame 59 附近 reconstructed q3 短暂换到等价的
零坐标表示，而 Newton `joint_qd` 仍连续，因此该差分 acceleration/jerk 只保留为
diagnostic，不再作为 physics acceptance；hard joint limit、position-difference
velocity ratio 以及 measured `joint_qd` 的 velocity/acceleration/jerk gate 均未
取消或放宽。

针对 `_02` 限位根因的下一步修复采用两项 fail-closed 设计：抓取姿态增加 +15° closing-axis roll 并把
actuation 延长到 800 帧，令 q4 参考硬限位余量增至 `0.109321601 rad`；同时新增
`0.05 rad` tracking-reserve preflight，审计 float32-realized 初始 nominal state 与
完整 reference。任一样本硬限位余量不足或触及 control endpoint，都会在进入昂贵
physics rollout 前以 `ik_unreachable / ik_motion_limits` 失败。

### 5.4 1288 帧 physics `_03`：限位修复通过，剩余切换冲击失败

产物 `articraft_continuity_physics_20260714_03` 完成 1288 帧，shell exit=3、
`run_status=acceptance_failed`。它证明 +15° roll 与 0.05 rad reserve 已消除上一轮
q4 限位失败，但也暴露了两个更窄的约束切换问题，不能计为最终通过。

通过项：最终门角 `64.887542°`（误差 `0.112458°`）；collision frame=0；measured
joint-limit violation=0、raw overshoot=0；TCP 中位/最大位置误差
`0.009347001 / 0.009482820 m`，中位/最大姿态误差
`1.062794° / 1.086859°`；最大 grasp 位置/姿态漂移
`0.009340590 m / 1.067772°`；reference reserve 的 1289 个样本全部通过，最小硬
限位余量 `0.095238111 rad`、control touch=0。capture 在 frame 344 以
`0.009356741 m / 1.061244°` pose error 和
`0.002300260 m/s / 0.236493°/s` relative twist 通过；anchor post-capture error
仅 `2.44e-9 m / 3.19e-6°`，enable/release write/readback 均为一次且 verified。

失败 gate 只有三项：

| Gate | 实测 | 门槛 | Frame / DoF | 根因 |
|---|---:|---:|---|---|
| finger measured jerk | 44.963385 m/s³ | 30 | 336 / finger1 | close 时单侧手指接触把手并在一帧内减速 |
| arm measured acceleration | 9.267461 rad/s² | 7.5 | 1192 / q4 | fixed joint disable 后 PD 目标与受约束平衡位置不一致 |
| arm measured jerk | 695.957901 rad/s³ | 450 | 1193 / q4 | 同一 release 卸载瞬态 |

frame 1191 的 measured/command q4 分别为 `-2.973300457 / -2.960705280 rad`，
约束解除时存在 `-0.012595177 rad` 的弹性偏差；frame 1192 q4 `joint_qd` 从
`-0.000740053` 跳到 `0.153717622 rad/s`。finger1 在 frame 334→335 从
`-0.021720946` 降至 `-0.008703861 m/s`，而 finger2 仍贴合
`-0.018319735 m/s` 指令，说明 44.96 峰值是单侧接触冲击，不是 quintic 指令自身
不连续。最终修复因此必须同时降低 close 接触速度，并做 fixed-joint release 的
bumpless PD target transfer；不能放宽 measured gate。

### 5.5 两项修复的断点物理 A/B

在再次执行完整长程仿真前，两个 repo 外诊断从 `_03` 保存的 Newton post-step
body pose 与 measured q/qd 恢复状态；body velocity 由 measured q/qd 经 `eval_fk`
重建，body pose 再写回保存值。fresh collision pipeline/solver 没有旧 run 的 warm-start
历史，因此这些结果只用于隔离故障机制，不能替代 §5.6 的完整 cold-start run。

close=96 诊断从 frame 295（approach 末端）恢复，执行最终 `_06` 的 frames
296--400，并走生产 activation gate、measured anchor capture、write/readback 与 contact
evidence 路径。首续帧最大 body 位置、joint q、joint qd 恢复误差分别为
`5.616e-6 m / 6.706e-6 rad / 0.001495 rad/s`，door error=0；结果为：

| close=96 measured finger 指标 | 实测 | 门槛 | 结论 |
|---|---:|---:|---|
| 最大速度 / URDF 比例 | 0.026971994 m/s / 0.134860 | <= 1.0 | PASS |
| 最大加速度 | 0.353329488 m/s²（frame 374, finger1） | <= 1.5 | PASS |
| 最大 jerk | 22.177132689 m/s³（frame 375, finger1） | <= 30 | PASS |

frame 392 activation pose error 为 `0.009403377 m / 1.063957°`，relative anchor
twist 为 `0.001840685 m/s / 0.141386°/s`，均通过原门槛；capture 后误差仅
`1.76e-10 m / 2.41e-6°`，anchor/enable readback verified，collision frame=0。
这证明延长 close 降低的是接触速度，finger dynamic gate 没有被放宽。

release A/B 从 `_03` 的 pre-release frame 1143 恢复，并在相同 1288 帧源命令和
约束 schedule 上对比旧切换与生产 bumpless helper。恢复后的首续帧最大 body 位置、
arm q、arm qd 对原 run 误差分别为 `2.31e-6 m`、`8.62e-6 rad` 和
`0.001305 rad/s`。结果为：

| measured arm 指标 | 旧 `_03` | bumpless | 门槛 | 结论 |
|---|---:|---:|---:|---|
| 最大加速度 | 9.267461 | 5.500327 rad/s² | 7.5 | FAIL → PASS |
| 最大 jerk | 695.957901 | 335.828503 rad/s³ | 450 | FAIL → PASS |

旧 disable 首帧 q4 为 `qd=+0.153718 rad/s, a=+9.267461 rad/s²`；bumpless 后为
`qd=-0.071528 rad/s, a=-4.24728 rad/s²`，下一帧 q4 jerk 从
`-695.957901` 降到 `+316.65318 rad/s³`。constraint schedule/readback 一致、
collision frame=0；生产 effective applied target 的 position/reserve 与
velocity/acceleration/jerk preflight 也全部通过。

### 5.6 最终 1336 帧 physics `_04`：完整 cold-start 通过

输出目录为
`/cache/liluchen/agentpre/outputs/articraft_continuity_physics_20260714_04`。
不中断的 Newton CPU run 以 shell exit=0 结束；`run.log` 最终事件为
`physics_rollout_completed(frame_count=1336, collision_frame_count=0)` 和
`run_completed(status=success, acceptance_passed=true)`。`metrics.json` 为
`success=true / run_status=success`，20 个 acceptance gate 全部 PASS，失败 gate
为空；`rollout.jsonl` 恰好 1336 行。

| 最终 physics 指标 | 实测 | 门槛 | 结论 |
|---|---:|---:|---|
| 最终门角 / 误差 | 64.922942476° / 0.077057524° | 误差 <= 3° | PASS |
| collision frame / joint-limit violation | 0 / 0 | 0 / 0 | PASS |
| TCP 位置误差 median / max | 0.009398131 / 0.009807159 m | median <= 0.02 m | PASS |
| TCP 姿态误差 median / max | 1.066716° / 1.145677° | median <= 10° | PASS |
| 最大抓取位置 / 姿态漂移 | 0.009394854 m / 1.078523° | <= 0.015 m / 7.5° | PASS |
| arm measured 速度限制比 | 0.499364060（frame 81, q7） | <= 1 | PASS |
| arm measured 加速度 | 5.513281743 rad/s²（frame 1192, q5） | <= 7.5 | PASS |
| arm measured jerk | 443.403909817 rad/s³（frame 82, q7） | <= 450 | PASS |
| finger measured 速度限制比 | 0.269573797（frame 1216, finger2） | <= 1 | PASS |
| finger measured 加速度 | 0.359840327 m/s²（frame 374, finger1） | <= 1.5 | PASS |
| finger measured jerk | 22.454151235 m/s³（frame 375, finger1） | <= 30 | PASS |

NaN/Inf 均为 false，IK 为 1336/1336。门 coordinate/DOF 的 q、qd、target 和
generalized-force runtime write count 继续全部为 0。frame 392 的 activation pose
error 为 `0.009411084 m / 1.065475°`，relative anchor twist 为
`0.001567146 m/s / 0.209987°/s`；capture 后 anchor error 为
`9.38e-10 m / 2.41e-6°`。anchor 写入/readback 以及 constraint 初始禁用、激活启用、
release 禁用均 verified。

release equilibrium 在 frame 1191 捕获，约束在 1192--1239 的 unload 中保持启用，
在 frame 1240 禁用，并于 1240--1271 在约束关闭状态下接回 planned retreat。
NPZ 独立复核确认 `applied_arm_joint_target_q` 和 `qd` 均为 float32，约束布尔序列
严格等于 `[392,1240)`；frame 1239→1240 的 applied arm target position 七个分量
逐 bit 相同，最大步长为 0。disable 前最大 target 速度仅
`1.108646e-4 rad/s`，disable 帧为 0。applied target 的 hard position、0.05 rad
reserve/control endpoint、velocity、acceleration 与 jerk preflight 全部 PASS；最小
硬限位余量 `0.095238111 rad`，control-bound touch=0。

Newton `joint_qd` 独立差分复现了表中 arm/finger acceleration 与 jerk 峰值。
相邻 reconstructed `joint_q` 的 acceleration/jerk 仍为 diagnostic-only（最大
`6.870922 rad/s² / 627.627401 rad/s³`），原因和 acceptance 边界见 §5.3 与 §6.2；
它没有替代或关闭 hard limit、position-difference velocity 与 Newton `joint_qd`
动态门禁。

## 6. Physics-assisted 语义与审计

### 6.1 旧版 124 帧历史快照

本节只描述旧版两次 physics run 的控制语义，不能被引用为最终 1336 帧物理验证。
旧版两次 run 均为 Newton 1.3.0、CPU、XPBD、64 solver iterations、每帧
32 substeps。状态在每帧完成 step 后采样：
`state_sample_timing=post_step_end_of_frame`。

- Franka 是动态刚体，运行时验证 dynamic flags 与正有限 inverse mass；控制为
  Newton joint position/velocity PD target（arm `ke=650, kd=100`；finger
  `ke=300, kd=40`），不是 torque-PD。
- robot target writer 只 indexed-scatter 写命名机器人 coordinates/DOFs 0..8；
  door coordinate/DOF 9 被排除。robot body state、generalized force 与 object body
  均不被该控制路径写入。
- 门只配置 `ke=0, kd=0.8` 的被动速度阻尼。door reference 只用于诊断，从不施加；
  门的 q、qd、target、generalized force runtime write count 全部为 0，初末 target
  q/qd 保持 0。
- 门零写入证据是 `static_indexed_control_path_guarantee`，不是全局运行时写拦截器。
- fixed-loop grasp 在 frame 32 激活、frame 112 释放，恰好覆盖 80 帧 actuate。
  激活前 gate：Articraft 为 6.251 mm / 0.749°，fixture 为 6.272 mm / 0.755°，
  都低于 15 mm / 7.5°；`remote_latch_allowed=false`。
- fixed joint 明确设置 `grasp_parent_child_collision_filtered=false`，没有屏蔽
  parent-child 碰撞；两次 run 均无 forbidden contact pair。
- 碰撞 margin 为 3 mm，证据范围严格为 `cross_asset_robot_object`。kinematic
  使用命名 URDF SAP/OBB，physics 使用 Newton contact evidence；不包含 Franka
  self-collision 证明。
- physics 根目录以实测动态状态及其 command/reference/target 对照为主；完整
  kinematic reference 隔离在 `kinematic_reference/`，reference
  acceptance=true、exit code=0。

因此，旧版 `physics_assisted` 的准确表述是“动态 Franka joint-target PD + 动态门
+ 有 gate 的 fixed-loop grasp assistance”，不是纯接触抓取、无辅助开门或力矩
控制；该语义说明本身不等于新版物理结果。

### 6.2 最终 1336 帧控制与验收语义

最终 checked-in 配置仍使用 Newton 1.3.0、CPU、XPBD 和 64 solver iterations，
但每帧为 48 substeps，arm `ke/kd=650/200`，finger `ke/kd=300/40`。六阶段边界为：

```text
pregrasp  0..223        approach 224..295       close   296..391
actuate   392..1191     release  1192..1239     retreat 1240..1335
```

fixed-loop 在 frame 392 启用，覆盖 `[392,1240)`，并在首个 retreat frame 1240
禁用。启用前同时检查 planned-vs-measured handle→TCP pose 与 parent/child anchor
relative twist；通过后从 measured hand/handle pose 捕获重合 parent anchor。child
anchor finalize、parent capture、initial disable、activation enable、retreat disable
均要求单次 transaction 和 state readback。release 阶段约束仍启用，门、handle、
TCP 保持 goal pose，只打开 fingers。frame 1191 捕获约束仍启用时的 measured
float32 arm equilibrium；1192--1239 将 applied PD target 以端点精确的 quintic
从 planned hold 卸载到该 equilibrium。frame 1240 禁用约束时 target 与上一帧
bit-exact 相同，随后 1240--1271 再平滑接回 planned retreat。因此解除约束与控制
目标切换不再在同一帧叠加冲击。planned reference 与实际 applied q/qd 均独立写入
rollout/NPZ；applied 全轨迹在写入 Newton 前重新通过 hard position、0.05 rad
tracking reserve/control endpoint、velocity、acceleration 和 jerk fail-closed 审计。

door reference 继续只用于诊断；door q/qd/targets/generalized force 不进入 indexed
robot target writer。physics acceptance 的 arm/finger velocity、acceleration 和 jerk
权威来源是 Newton post-step name-resolved `joint_qd`，分别使用
`7.5 rad/s² / 450 rad/s³` 与 `1.5 m/s² / 30 m/s³` 门槛。相邻 reconstructed
`joint_q` 差分得到的 acceleration/jerk 仍输出但明确标为 diagnostic-only；它没有
替代或关闭 hard joint-limit、position-difference velocity ratio 或 measured `joint_qd`
动态 gate。这里的 peak 定义是 60 Hz frame endpoint 样本，不声称覆盖 48 个
substep 内先冲高再回落的未采样瞬态。

在物理初始化前，tracking-reserve preflight 对 float32-realized nominal 初始 arm
state 和全部 1336 个 reference frames 做 fail-closed 审计。硬限位余量严格小于
`0.05 rad` 或触及由 URDF bounds、`0.02 rad` margin 和 float32 inward endpoint
共同定义的 control bound，都会以 `ik_unreachable / ik_motion_limits` 退出，且
不会调用 Newton physics simulator。

## 7. 测试与静态检查

### 7.1 旧提交历史记录

在远端 Linux、真实 `flock` 环境中，旧提交 `93e3b47` 的验证为：

```text
python -m unittest discover -s tests -v     140/140 PASS
python -m py_compile src/*.py scripts/*.py tests/*.py     PASS
bash -n scripts/*.sh                       PASS
git diff --check                           PASS
```

覆盖内容包括：配置/资产哈希 fail-closed、URDF inspection、affordance、FK/IK、
碰撞、metrics、physics 写入审计、Articraft 惯性事务/中断恢复/materialization、
同步 isolated-index 竞态和 daemon 生命周期。这组 140/140 记录只属于旧提交。

### 7.2 最终连续性修复快照

最终源码在本地 macOS 和远端 Linux 的完整回归分别为：

```text
local  python3 -m unittest discover -s tests   196 PASS, 4 skipped
remote python  -m unittest discover -s tests   196 PASS
python3 -m py_compile src/*.py scripts/*.py tests/*.py   PASS
bash -n scripts/*.sh                           PASS
git diff --check                               PASS
```

新增覆盖包括：UI 跳变回归、temporal IK objective、stationary target hold、
float32 velocity/acceleration/jerk projector、physics command terminal hold、measured
arm/finger `joint_qd` gates、capture pose/twist/readback、release transaction、door
zero-write、bumpless release applied-target 审计/序列化、position-difference
diagnostic-only 口径，以及 nominal 初始状态 + 完整
reference 的 hard-reserve/control-endpoint fail-early 审计。

## 8. 旧版 124 帧关键运行产物 SHA-256（历史）

### 8.1 Articraft kinematic

```text
metrics.json               a363eec6dcf1fcc3b1bdf3818c1fe44991993746db8588425d34163b579a5760
resolved_config.json       43a7ef9be8428fd02922a805c7335c2e75414355e727849627a34677b5fabdc2
asset_inspection.json      c1b8dcb26d41f56b1c6f7df6105aef9ad2b44d1ce8cb939f3fb5d8416e8cc94c
affordance_candidates.json 275552f2f86810a0a698897ec5f3ea828350b3050fdd787b252a85f34a695e95
collision_report.json      ae85a22db1341f6c95c4ec8543b8b79e417127f0623d961d7f4ad943e7274562
trajectory.npz             0d51281f2316269ab19bf62e2670891407583805c46c5fb6fc35ede49616671b
rollout.jsonl              0d00931472243459a5541695c8d2645158a292e2e5bf9d78b2bdec0e15a384c4
run.log                    72d925e0f3ded125ac2da08dbf58e933eba3fe7cbfb8d5bc392d2428f291cdd5
```

### 8.2 Articraft physics

```text
metrics.json                          e025dcea43292fe55593254fd200f4da1f11d5da196a6137c81ee26bd134eecc
resolved_config.json                  f359e80ac2219b0099c3f8199f054e1fb959cb03a13588bd20a44d186ca88ea7
asset_inspection.json                 c1b8dcb26d41f56b1c6f7df6105aef9ad2b44d1ce8cb939f3fb5d8416e8cc94c
affordance_candidates.json            275552f2f86810a0a698897ec5f3ea828350b3050fdd787b252a85f34a695e95
collision_report.json                 673f921d53707b27cda8109af61218e438845472b62d16c1d4219b45f1493637
trajectory.npz                        197a8e2f3e75692afefc31b3b858dd5541c860f9d7cb3addcb77d5baae6935aa
rollout.jsonl                         65c904c3f69cf489ce3d2234e2e4766406ad67a64f18207fcfc6e7336ad9af17
run.log                               f04c57e057d23f05bf096f8797418e965b2488ba80820886e4f923edf18e4946
kinematic_reference/metrics.json      a363eec6dcf1fcc3b1bdf3818c1fe44991993746db8588425d34163b579a5760
kinematic_reference/run.log           72d925e0f3ded125ac2da08dbf58e933eba3fe7cbfb8d5bc392d2428f291cdd5
```

### 8.3 Fixture kinematic

```text
metrics.json               7820083c94eb2f6bbf0b43023858be5b42e0b73de0c15ce2659ca4148d110984
resolved_config.json       d993af9cf423166b5e679a4172b77857e6b78778f38b5f2a9bc70f56bf4089ed
asset_inspection.json      5357189e8df80e48c62218933a1f60888dc8b0877ed2043ca0a15a286e2b8b83
affordance_candidates.json 1c0da9a155059eaf43e664bef2fe981ed7ea2b32e3c4d785d5d3f0f6e52fb4b7
collision_report.json      2d298d2e9c6d2fc7f479057657e36a76e879a80e8de00ecd68174715b70fcf42
trajectory.npz             404e765a0ab698339fce94f5581fdccc4f9bfcdda43439f1b6a684d5a83ae80e
rollout.jsonl              ef3b4ee6aff35c3e3f19083faf5c97752d35a3354b0d81a862f3b09b7efd90d7
run.log                    9dd5da309a95e54c19541015769792a11bb9526849170aac78b6a8ab3a12fc43
```

### 8.4 Fixture physics

```text
metrics.json                          848cf1b1d4b385052a72adbd3263a222661ee83b1732ff7fa1aba290eb19a19d
resolved_config.json                  fee5538c9383fefc423f4e72a001261ac9cc61f575cc070fb5c0ea493473244e
asset_inspection.json                 5357189e8df80e48c62218933a1f60888dc8b0877ed2043ca0a15a286e2b8b83
affordance_candidates.json            1c0da9a155059eaf43e664bef2fe981ed7ea2b32e3c4d785d5d3f0f6e52fb4b7
collision_report.json                 ab183f6148c31b7f50a9b5265fd406eafa5a3dc6e057e2fa8867e9d34781d88a
trajectory.npz                        460035f54905bd3e8fb670bc8830147bd24d59794f17d0a9b9add1644705b149
rollout.jsonl                         265c5468f7e67ef8dd804c6b495c56bf3e248b8d5715c74ba491c11222a19249
run.log                               f04c57e057d23f05bf096f8797418e965b2488ba80820886e4f923edf18e4946
kinematic_reference/metrics.json      7820083c94eb2f6bbf0b43023858be5b42e0b73de0c15ce2659ca4148d110984
kinematic_reference/run.log           9dd5da309a95e54c19541015769792a11bb9526849170aac78b6a8ab3a12fc43
```

旧版两次 physics 的 reference metrics/run.log 与各自独立 kinematic run 字节级
一致。§8.1--§8.4 哈希均只属于旧版 124 帧产物。

### 8.5 最终 1336 帧 Articraft kinematic

输出目录：`articraft_continuity_kinematic_20260714_06`

```text
metrics.json               bc3dbfa8403ec6b8796afc3ca5dbaf946c748a7d226ad4d503cbcbde52b1bc19
resolved_config.json       2dad53ec67c9897d9dbc3d6788f96bb03fad5d9351bc846e3ce1091bca687b8e
asset_inspection.json      c1b8dcb26d41f56b1c6f7df6105aef9ad2b44d1ce8cb939f3fb5d8416e8cc94c
affordance_candidates.json f1d8d083bb1fb345dc6160849ca8c21c7bb13e7ad012930db7b2f4f82282f2dc
collision_report.json      6f51d24ecf304762a40fb2779abc28790ba6256948b06b9d2c3ca7e08ed8b760
trajectory.npz             ec9bacc4e5adbb10b088a783a36592d2e3abdfaae691f73f1e64f986d64066ed
rollout.jsonl              5b42c4ac7ca0b2b0fe95145f6982c362f2959c532ddd780189b3a9a846a970dd
run.log                    c8c8e36876f21235ccf04827819bf9d76984c4c5ca9dfe790d5be59ea04b9b42
```

### 8.6 Repo 外断点物理诊断

```text
close96_activation_resume_diag.json 2f15e474632618ec4ac91b8ec8906ef0aff31ba7a63268fecf75d9a48bc3a83c
close96_activation_resume_diag.npz  ff5638b137e711e7ee289becb065bccb39974c983c6f47a6f427186401283f0a
resume_release_bumpless_diag.json    f9f4a8db0ec18a85ed4e82b72ded5d8bf33a5254a9cfdc10c0182e6b30499eaf
resume_release_bumpless_diag.npz     3afd01fb9a4358bc825a616e09901e925188b1a44eb6f18f6278f498cbcaac66
```

这些文件位于 `/cache/liluchen/agentpre/outputs`，不进入 Git；准确性边界见 §5.5。

### 8.7 最终 1336 帧 Articraft physics

输出目录：`articraft_continuity_physics_20260714_04`

```text
metrics.json               727110052fbc8bc66445fe1ebd28a14b3a6072d6705beed3e3d66ff3b2eb0b3f
resolved_config.json       8e027efcf0a69d474fad57c6603b71b5fbbefc34be05f43e75144c599acbdeb8
asset_inspection.json      c1b8dcb26d41f56b1c6f7df6105aef9ad2b44d1ce8cb939f3fb5d8416e8cc94c
affordance_candidates.json f1d8d083bb1fb345dc6160849ca8c21c7bb13e7ad012930db7b2f4f82282f2dc
collision_report.json      a37b474343be441c1d2209a4e5d587d14d957bec59ca6fc38e59914b7acffda9
trajectory.npz             dcb4b6b0d060855352bccd05ebb6cfa90f14cf18aad927239fa75fe1792c10cd
rollout.jsonl              80a4c766e2072d9bb5035b12afdf481ff9b4c431f51485fab27f3ced2cadbdb9
run.log                    9a457ad5c880b297e3681d1f81866e7f78e6c18ddbb14f5026f4e5ac6e67d93d
```

## 9. GitHub 同步 daemon（历史状态快照）

### 9.1 竞态与安全策略

`sync_to_github.sh` 使用 `flock` 与 alternate index，只允许源码、配置、测试、报告
等白名单路径；不会提交 `/cache/liluchen` 中的环境、资产 materialization、NPZ、
JSONL 或日志。提交前扫描常见凭据模式。脚本还记录普通 Git index 的字节哈希；
若交互用户在 isolated commit 期间改变 staging，脚本以 exit 6 fail-closed，不执行
`git read-tree HEAD`，保留用户 staging 且不 push。

### 9.2 两周期短测

修复提交 `93e3b47` 上，以 3 秒测试间隔运行 PID `557252`：

| 周期 | Started UTC | Finished UTC | Exit | Git 结果 |
|---|---|---|---:|---|
| 1 | 2026-07-14T08:23:54Z | 2026-07-14T08:23:58Z | 0 | Everything up-to-date |
| 2 | 2026-07-14T08:24:01Z | 2026-07-14T08:24:05Z | 0 | Everything up-to-date |

停止后 PID 文件被移除，heartbeat 为：

```text
pid=557252 state=stopped timestamp_utc=2026-07-14T08:24:07Z timestamp_epoch=1784017447 last_sync_exit_code=0 next_sync_epoch=0 interval_seconds=3
```

### 9.3 正式服务

当时记录的生产 daemon 状态快照为：

```text
pid=557544
interval_seconds=1800
state=sleeping
last_sync_exit_code=0
heartbeat_utc=2026-07-14T08:27:25Z
next_sync_utc=2026-07-14T08:57:25Z
log=/cache/liluchen/agentpre/logs/github-sync-daemon.log
```

首次生产同步为 `08:27:21Z -> 08:27:25Z`、exit 0、
`Everything up-to-date`。只读取该 PID 的四个命名环境变量，结果为：

```text
CUDA_VISIBLE_DEVICES=
NVIDIA_VISIBLE_DEVICES=void
HIP_VISIBLE_DEVICES=
ROCR_VISIBLE_DEVICES=
```

installer 精确移除了旧 AgentPre cron marker + matching command pair，未修改其他
crontab 项；核验为 `legacy_marker_count=0`、`matching_command_count=0`。

## 10. 复现命令

```bash
cd /workspace/liluchen/AgentPre
source scripts/env.sh

# 资产编译、确定性惯性注入与缓存 materialization
bash scripts/build_articraft_asset.sh

# 单元测试
python -m unittest discover -s tests -v

# 真实 Articraft
python -m src.run --config configs/articraft_microwave_franka.json --mode kinematic
python -m src.run --config configs/articraft_microwave_franka.json --mode physics_assisted

# Fixture regression
python -m src.run --config configs/microwave_franka.json --mode kinematic
python -m src.run --config configs/microwave_franka.json --mode physics_assisted

# Daemon
bash scripts/install_sync_daemon.sh
bash scripts/sync_daemon.sh status
```

## 11. 不计为成功的范围

- 旧版四次 124 帧运行只通过旧位姿/任务门槛；因未检查帧间步长、速度、加速度和
  jerk，不计为运动连续性或动态可执行性成功证据。
- 早期 `_02` 与 `_03` physics run 都是失败诊断，不能替代最终 1336 帧 `_04`
  cold-start 通过结果；repo 外断点 A/B 也只证明局部根因修复。
- 真实资产是官方 Articraft geometry/kinematics 加 AgentPre 后处理的确定性 proxy
  inertials；不是原厂质量参数或未经修改的原始 URDF。
- `kinematic` 是规划、FK/IK、候选、碰撞和指标基线，不是物理 rollout。
- `physics_assisted` 明确依赖 fixed-loop grasp assistance；不证明纯接触抓取或
  无辅助开门。
- 门零写入是命名、索引化控制代码路径保证，不是全局写入拦截器。
- 碰撞只覆盖机器人—物体跨资产对，不证明 Franka self-collision 或完整环境碰撞。
- 本次仅验证 CPU，不作 GPU 性能或 GPU 正确性声明。
- 控制流程是固定 seed、固定六阶段（pregrasp、approach、close、actuate、release、
  retreat）、单资产/单配置的确定性基线，无 LLM 决策，不证明跨资产泛化。
- Fixture 结果只用于 regression，不能替代最终真实 Articraft 1336 帧 physics
  结果。
