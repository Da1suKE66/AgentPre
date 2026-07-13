# AgentPre 远端验证报告

验证日期：2026-07-14（Asia/Shanghai）  
验证主机：`lsh-stable32314`  
验证源码提交：`2191a67d1363356f3e9c5c915aa539ac044caf57`

## 1. 结论

AgentPre 的确定性微波炉开门流水线已在指定远端主机上完成复验。要求的
`kinematic` 命令和附加的 `physics_assisted` 命令均以退出码 0 完成，所有
配置化 acceptance gates 均通过。验证运行只使用 CPU，没有申请或打断主机上
已占用的 GPU。

这里的能力边界必须明确：

- 仓库内微波炉是 `deterministic_test_fixture_not_articraft` 测试夹具，不是
  Articraft 导出物。
- `kinematic` 使用 Newton 1.3 的解析 LM IK，以及项目自己的确定性 FK、抓取
  候选、碰撞和指标流水线。
- `physics_assisted` 使用运动学 Franka body driver，Newton 只负责推进动态门和
  固定抓取约束；它不是力矩控制或全动态机器人验证。
- 当前碰撞证据范围严格为 `cross_asset_robot_object`，不声称 Franka
  self-collision 已验证。
- 控制流程是配置驱动、无 LLM 决策的确定性基线。

## 2. 可复现环境

```text
源码目录       /workspace/liluchen/AgentPre
缓存根目录     /cache/liluchen/agentpre
Conda 环境     /cache/liluchen/agentpre/envs/agentpre-conda
运行产物       /cache/liluchen/agentpre/outputs
Python         3.11.15
Newton         1.3.0
Warp           1.14.0
设备           cpu
线程           OMP/OPENBLAS/MKL/NUMEXPR = 1
Git remote     ssh://git@ssh.github.com:443/Da1suKE66/AgentPre.git
```

`CUDA_VISIBLE_DEVICES` 在 `scripts/env.sh` 中设为空。Warp 启动时会打印
“CUDA driver not available”，但其设备列表只有 `cpu`；这是预期的 CPU-only
运行边界，不是失败或 GPU 回退。

Franka bootstrap 验证解析了 URDF 的全部 mesh 引用：20/20 个文件存在并写入
SHA-256 manifest。URDF 哈希为
`ad9f5298a4d1a375cf16824b0de4f0d1c7cc446597964b80aa639ca830e998a1`。

## 3. 复现命令与测试

```bash
cd /workspace/liluchen/AgentPre
source scripts/env.sh
python scripts/fetch_assets.py
python -m unittest discover -s tests -p 'test_*.py'
python -m src.run --config configs/microwave_franka.json --mode kinematic
python -m src.run --config configs/microwave_franka.json --mode physics_assisted
```

最终提交上的全量测试为 92/92 通过；同时通过：

- `python -m py_compile src/*.py scripts/fetch_assets.py`
- `bash -n scripts/*.sh`
- `git diff --check`

## 4. Kinematic 验证

产物目录：
`/cache/liluchen/agentpre/outputs/kinematic_seed_20260714_0004`

| 指标 | 实测值 | 门槛 | 结果 |
|---|---:|---:|---|
| IK 成功率 | 124/124 = 100% | >= 95% | PASS |
| TCP 位置误差中位数 | 0.000614416 m | <= 0.02 m | PASS |
| TCP 姿态误差中位数 | 0.083598 deg | <= 10 deg | PASS |
| TCP 最大位置误差 | 0.002997900 m | 诊断值 | — |
| TCP 最大姿态误差 | 0.165683 deg | 诊断值 | — |
| 最终门角 | 65.000000 deg | 目标 65 deg | PASS |
| 最终门角误差 | 0.000000 deg | <= 3 deg | PASS |
| 最大 handle-TCP 位置漂移 | 0.002832822 m | <= 0.015 m | PASS |
| 最大 handle-TCP 姿态漂移 | 0.163044 deg | <= 7.5 deg | PASS |
| 碰撞帧 | 0/124 | 0 | PASS |
| 关节越界 | 0 | 0 | PASS |
| NaN / Inf | 0 / 0 | 必须为 0 / 0 | PASS |

## 5. Physics-assisted 验证

产物目录：
`/cache/liluchen/agentpre/outputs/physics_assisted_seed_20260714_0004`

| 指标 | 实测值 | 门槛 | 结果 |
|---|---:|---:|---|
| Kinematic 参考 | 124/124 IK，acceptance 通过 | 必须通过 | PASS |
| TCP 位置误差中位数 | 0.000614358 m | <= 0.02 m | PASS |
| TCP 姿态误差中位数 | 0.083602 deg | <= 10 deg | PASS |
| 最终动态门角 | 67.240637 deg | 目标 65 deg | PASS |
| 最终门角误差 | 2.240637 deg | <= 3 deg | PASS |
| 抓取启用位置误差 | 0.000360609 m | <= 0.015 m | PASS |
| 抓取启用姿态误差 | 0.077497 deg | <= 7.5 deg | PASS |
| 最大 handle-TCP 位置漂移 | 0.000894489 m | <= 0.015 m | PASS |
| 最大 handle-TCP 姿态漂移 | 0.092139 deg | <= 7.5 deg | PASS |
| 碰撞帧 | 0/124 | 0 | PASS |
| 关节越界 | 0 | 0 | PASS |
| NaN / Inf | 0 / 0 | 必须为 0 / 0 | PASS |

物理语义与审计证据：

- backend：Newton XPBD，64 solver iterations，32 substeps/frame，CPU。
- Franka 11 个命名 body 由 indexed scatter 写入 `body_q/body_qd`；这些 body 的
  kinematic flag、零逆质量和零逆惯量均在运行时验证。
- object body indices `[11, 12, 13]` 与 robot body indices 不相交，且不被 driver
  写入。
- 门角由动态 body state 经 Newton `eval_ik` 重建；door reference 仅用于诊断，
  从不施加。
- 门的 `q`、`qd`、target、generalized force 运行时写入计数均为 0；证据类型是
  `static_indexed_control_path_guarantee`。这不是运行时全局写拦截器的观测声明。
- robot joint force backend 为 `none`，记录的 external robot joint force command
  全为 0。
- fixed-grasp anchors 来自计划的 handle-frame-to-TCP 关系，运行时不重写 anchor；
  frame 32 启用前通过 15 mm / 7.5 deg gate，因此不允许 remote latch。
- 碰撞按 3 mm 配置 margin 对 Newton signed effective-surface clearance 判定，证据
  范围仅为机器人—物体跨资产对。

Kinematic 参考的所有 trajectory/rollout 文件均隔离在 physics 产物的
`kinematic_reference/` 子目录，未与根目录的实测 physics 产物混用。

## 6. 关键产物 SHA-256

### Kinematic

| 文件 | SHA-256 |
|---|---|
| `metrics.json` | `27e099939b0b109a173936178b46b4465bc803ded188a3c5d6970ca1da0f9405` |
| `resolved_config.json` | `fc104868863366e9208a3776080e6ab61c6285f992a48092028932673100d92e` |
| `trajectory.npz` | `b07f6cfea46956ca1f687ca8c3aa6c2d0d04bd8f6058eab16c218198ccf0223a` |
| `rollout.jsonl` | `32808e10d808b0c17f9bb27751c3b53e12388e481a57b1a1cdee460453312e4c` |
| `collision_report.json` | `ac40de60cdefe6238bbf15b81b32dd306650ad1c54ab6ff13d835e8a207995aa` |

### Physics-assisted

| 文件 | SHA-256 |
|---|---|
| `metrics.json` | `97940ce47cedbc72e1dd77dd62f216818eb1d2627415d10e89f56152d9370e58` |
| `resolved_config.json` | `c84a35397332415f5752b1e0c94272a23e64aff67dcb0e34ad262122a539c225` |
| `trajectory.npz` | `5d94250431c05f233a5425ec371e5bdab02e71fe01afb43ed240af8013bd6c23` |
| `rollout.jsonl` | `fe16cccf3e60c34d46ce044d29cfc83d044918cd2c7cc75d3e4e862177cfa9f4` |
| `collision_report.json` | `303ca2977084e4b8874e00cf44aafc4dccbe9a217b43df9ede6f6ca45a35ff5a` |

两个 `resolved_config.json` 都记录了验证源码提交
`2191a67d1363356f3e9c5c915aa539ac044caf57`。

## 7. GitHub 与定时同步

GitHub 标准 SSH 22 端口在该环境不可用，但
`ssh://git@ssh.github.com:443/Da1suKE66/AgentPre.git` 已完成 push 和
`ls-remote` 双向验证。同步脚本使用隔离 Git index、路径白名单、凭据模式扫描和
`flock`，不会把 Conda 环境、缓存、运行日志、JSONL 或 NPZ 推入仓库。

已安装的 crontab：

```cron
# AgentPre periodic GitHub sync
*/30 * * * * AGENTPRE_ROOT=/workspace/liluchen/AgentPre AGENTPRE_CACHE_ROOT=/cache/liluchen/agentpre /workspace/liluchen/AgentPre/scripts/sync_to_github.sh >> /cache/liluchen/agentpre/logs/github-sync.log 2>&1
```

## 8. 不计为成功的范围

本报告不把以下内容包装成已验证结果：

- Articraft 真实导出资产；当前只有明确标注的 deterministic fixture。
- 全动态或力矩控制 Franka；最终 schema 只接受 `kinematic_body_driver`。
- Franka self-collision；没有匹配的 SRDF/disabled-pair policy，粗粒度包围盒会产生
  不可信的假阳性。
- GPU 运行；本次验证主动隐藏 CUDA，仅确认 CPU 基线。

