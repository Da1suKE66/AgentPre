# AgentPre：Articraft 微波炉 + Franka 远端验证报告

- 验证日期：2026-07-14（Asia/Shanghai）
- 验证主机：`lsh-stable32314`
- 源码目录：`/workspace/liluchen/AgentPre`
- 缓存根目录：`/cache/liluchen/agentpre`

## 1. 结论与提交边界

AgentPre 已在指定远端主机上完成真实 Articraft 微波炉与 Franka Panda 的确定性
开门基线。真实 Articraft 与仓库 fixture 各执行一次 `kinematic` 和一次
`physics_assisted`，四次 clean run 均满足：

- shell transcript `exit_code=0`；
- `resolved_config.resolved_runtime.git_dirty=false`；
- `metrics.success=true`、`run_status=success`；
- 最终日志事件为 `run_completed` 且 `acceptance_passed=true`；
- 两次 kinematic 及两次 physics 的 kinematic reference 均为 124/124 个 IK
  waypoint 成功，全部 acceptance gates 通过；
- 0 个碰撞帧、0 次关节越界、无 NaN/Inf。

四次仿真严格绑定源码提交：

```text
a430dea08b04b6aa701dcb3d7498b39d923d0fb0
```

仿真完成后发现并修复了 daemon 停止时 EXIT trap 读取已离开作用域局部变量的
问题。该修复提交为：

```text
93e3b4797651defe2e6905bf867e14d3ee618731
```

`a430dea..93e3b47` 只修改 `scripts/sync_daemon.sh` 和
`tests/test_sync_daemon.py`，没有修改资产、配置或仿真代码。最终报告提交不能在
正文中自包含自身 Git hash；最终 GitHub HEAD 由交付时的 `ls-remote` 结果给出。

## 2. 环境与 CPU-only 边界

```text
Conda 环境     /cache/liluchen/agentpre/envs/agentpre-conda
运行产物       /cache/liluchen/agentpre/outputs/clean_a430dea
Python         3.11.15
NumPy          2.4.6
Newton         1.3.0
Warp           1.14.0
设备           cpu
线程           OMP/OPENBLAS/MKL/NUMEXPR = 1
Git remote     ssh://git@ssh.github.com:443/Da1suKE66/AgentPre.git
```

`scripts/env.sh` 设置 `CUDA_VISIBLE_DEVICES=""`。Warp 启动时会打印没有 CUDA
driver 的探测信息，但列出的唯一运行设备为 `cpu`；四次验证没有申请或占用 GPU。

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

URDF 加三个唯一 mesh 共 4 个 runtime 文件、699122 bytes。Articraft 两次 run
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
一致，总计 10633539 bytes。四次 run 的 expected/observed Franka URDF SHA
全部一致。

仓库 fixture 只作为回归资产，不替代真实 Articraft 结果。其 URDF SHA-256 为：

```text
d6ba39f326d52a02efe6c4292accc8503e32c3a19a5462a90e564cddf52177a1
```

## 5. 四次 clean run

所有 run 使用 seed `20260714` 和五阶段 124 帧轨迹：pregrasp 12、approach 12、
close 8、actuate 80、retreat 12。transcript 与 `resolved_config.json` 均确认源码
为 `a430dea…0fb0`、运行前/运行时 Git clean。

| 资产 / 模式 | 输出目录 | shell exit | IK | TCP 中位误差：位置 / 姿态 | 最终门角 / 误差 | 最大 grasp 漂移：位置 / 姿态 | 碰撞 / 越界 / 非有限值 | 结果 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Articraft / kinematic | `articraft_kinematic_0001` | 0 | 124/124 | 0.000611533 m / 0.083526° | 65.000000° / 0° | 0.003256837 m / 0.176567° | 0 / 0 / 无 | PASS |
| Articraft / physics | `articraft_physics_0002` | 0 | 参考 124/124 | 0.001315410 m / 0.171639° | 65.728413° / 0.728413° | 0.006273500 m / 0.756948° | 0 / 0 / 无 | PASS |
| Fixture / kinematic | `fixture_kinematic_0001` | 0 | 124/124 | 0.000614416 m / 0.084453° | 65.000000° / 0° | 0.003463282 m / 0.183908° | 0 / 0 / 无 | PASS |
| Fixture / physics | `fixture_physics_0001` | 0 | 参考 124/124 | 0.001692495 m / 0.151787° | 65.377328° / 0.377328° | 0.006273053 m / 0.797747° | 0 / 0 / 无 | PASS |

共同门槛：IK success rate >= 0.95；TCP 位置/姿态误差中位数分别 <= 0.02 m / 10°；
最终门角误差 <= 3°；最大抓取位置/姿态漂移 <= 0.015 m / 7.5°；碰撞帧比例、
关节越界和非有限值均必须为 0。Fixture physics 的最大 TCP 位置误差为
0.021014188 m，但 2 cm gate 明确检查中位数 0.001692495 m；本报告不声称所有帧
位置误差都低于 2 cm。

对应 shell transcript：

| Run | Started UTC | Finished UTC | 耗时 | Exit |
|---|---|---|---:|---:|
| Articraft kinematic | 2026-07-14T07:29:06Z | 2026-07-14T07:29:19Z | 13 s | 0 |
| Articraft physics | 2026-07-14T07:57:37Z | 2026-07-14T08:05:21Z | 7 min 44 s | 0 |
| Fixture kinematic | 2026-07-14T07:30:18Z | 2026-07-14T07:30:28Z | 10 s | 0 |
| Fixture physics | 2026-07-14T07:30:43Z | 2026-07-14T07:31:58Z | 1 min 15 s | 0 |

## 6. Physics-assisted 语义与审计

两次 physics run 均为 Newton 1.3.0、CPU、XPBD、64 solver iterations、每帧
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

因此 `physics_assisted` 的准确表述是“动态 Franka joint-target PD + 动态门 +
有 gate 的 fixed-loop grasp assistance”，不是纯接触抓取、无辅助开门或力矩控制。

## 7. 测试与静态检查

在远端 Linux、真实 `flock` 环境中，提交 `93e3b47` 的最终验证为：

```text
python -m unittest discover -s tests -v     140/140 PASS
python -m py_compile src/*.py scripts/*.py tests/*.py     PASS
bash -n scripts/*.sh                       PASS
git diff --check                           PASS
```

覆盖内容包括：配置/资产哈希 fail-closed、URDF inspection、affordance、FK/IK、
碰撞、metrics、physics 写入审计、Articraft 惯性事务/中断恢复/materialization、
同步 isolated-index 竞态和 daemon 生命周期。

## 8. 关键运行产物 SHA-256

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

两次 physics 的 reference metrics/run.log 与各自独立 kinematic run 字节级一致。

## 9. GitHub 同步 daemon

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

生产 daemon 已安装并保持运行：

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

- 真实资产是官方 Articraft geometry/kinematics 加 AgentPre 后处理的确定性 proxy
  inertials；不是原厂质量参数或未经修改的原始 URDF。
- `kinematic` 是规划、FK/IK、候选、碰撞和指标基线，不是物理 rollout。
- `physics_assisted` 明确依赖 fixed-loop grasp assistance；不证明纯接触抓取或
  无辅助开门。
- 门零写入是命名、索引化控制代码路径保证，不是全局写入拦截器。
- 碰撞只覆盖机器人—物体跨资产对，不证明 Franka self-collision 或完整环境碰撞。
- 本次仅验证 CPU，不作 GPU 性能或 GPU 正确性声明。
- 控制流程是固定 seed、固定五阶段、单资产/单配置的确定性基线，无 LLM 决策，
  不证明跨资产泛化。
- Fixture 结果只用于 regression，不能替代真实 Articraft 两次结果。
