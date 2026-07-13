# AgentPre: deterministic microwave-door opening with Franka

AgentPre compiles an articulated microwave task into a deterministic five-phase
trajectory and solves it with Newton 1.3's analytic Levenberg-Marquardt IK.  It
also has a CPU-only `physics_assisted` mode: the Franka links are prescribed by
a named-link kinematic body driver while Newton advances the dynamic microwave
door and the grasp constraint.  This mode deliberately does **not** claim a
torque-controlled or fully dynamic robot.  The first-stage controller is not an
LLM; a future agent can call this deterministic compiler/runtime as a tool.

The checked-in microwave is a small deterministic URDF fixture used for pipeline
verification.  It is clearly marked `deterministic_test_fixture_not_articraft`
and must not be presented as an Articraft export.  Replace the object URDF and
affordance paths in the config when a real Articraft export is available; no code
or simulator indices need to change.

## Repository layout

```text
assets/       microwave fixture, meshes, and affordances.json
configs/      all asset names, poses, controls, solver settings, and gates
src/          inspection, FK, grasp candidates, IK, collision, metrics, CLI
tests/        dependency-light unit tests
outputs/      documentation only; real rollouts live under /cache/liluchen
reports/      checked-in validation reports
scripts/      environment, asset bootstrap, run, and GitHub synchronization
```

All project-facing quaternions use `wxyz`.  The Newton adapter converts to and
from Newton/Warp's `xyzw` convention only at the backend boundary.  Links and
joints are resolved by their URDF labels (exact match or a unique qualified
suffix), never by hard-coded numeric indices.

## Remote setup

The intended host layout is:

```text
/workspace/liluchen/AgentPre        # small source checkout
/cache/liluchen/agentpre/envs       # Conda environment
/cache/liluchen/agentpre/assets     # Franka asset tree
/cache/liluchen/agentpre/outputs    # run artifacts
/cache/liluchen/agentpre/*-cache    # pip, Conda, Warp, Newton caches
```

On `lsh-stable32314`:

```bash
cd /workspace/liluchen/AgentPre
bash scripts/setup_env.sh
source scripts/env.sh
```

`scripts/env.sh` hides CUDA and fixes numerical thread counts to one.  The
checked-in baseline therefore does not reserve or interrupt the host's occupied
GPU.  Dependencies are pinned in `requirements.lock`; the validated runtime is
Python 3.11, `newton==1.3.0`, `warp-lang==1.14.0`, and CPU.

The Franka asset is copied into the AgentPre cache from the configured existing
Apache-2.0 `franka_description` tree.  `scripts/fetch_assets.py` parses the URDF,
validates every referenced mesh, and records every referenced file in a SHA-256
manifest.  It fails closed if the configured source is unavailable or the tree
is incomplete.

## Inspect and run

Inspect the articulated object before running:

```bash
python -m src.asset_inspector assets/microwave/microwave.urdf \
  --door-joint door_hinge \
  --door-link microwave_door \
  --handle-link handle
```

Run the required deterministic mode:

```bash
python -m src.run \
  --config configs/microwave_franka.json \
  --mode kinematic
```

Or use the wrapper:

```bash
bash scripts/run_example.sh kinematic
```

Run the measured Newton rollout with:

```bash
python -m src.run \
  --config configs/microwave_franka.json \
  --mode physics_assisted
```

In `physics_assisted`, the kinematic result supplies the prescribed Franka link
poses.  Robot bodies are marked kinematic with zero inverse mass and inertia;
the driver writes only the explicitly resolved robot `body_q/body_qd` indices.
It writes no robot generalized force.  Newton advances the articulated object,
and the door coordinate is reconstructed from the measured Newton state with
inverse kinematics.

A pre-authored fixed loop constraint, whose anchors encode the planned
handle-frame-to-TCP relation, is enabled after `close` and released at
`retreat`.  Immediately before activation, the measured relation must pass a
15 mm / 7.5 degree gate; a remote or discontinuous latch is rejected.  The door
reference trajectory is diagnostic only.  The implementation excludes the door
coordinate and DOF from its indexed driver and performs no runtime write to the
door position, velocity, target, or generalized force.  This is a static,
indexed-control-path guarantee recorded in the artifacts, not a claim that a
runtime write interceptor observed every possible external mutation.

Use `--output-dir /absolute/path` to select a particular run directory.  Without
it, the CLI creates a run below the config's `output.root`.  Completed rollouts
use `success` or `acceptance_failed`; structural pipeline failures use `failed`
and explicit codes such as `asset_invalid`, `frame_missing`, `ik_unreachable`,
`joint_limit_violation`, `collision`, or `numerical_instability`.  The CLI exit
codes distinguish success, acceptance failure, and pipeline failure.

Each completed rollout writes:

- `resolved_config.json` and `asset_inspection.json`
- `affordance_candidates.json` and `collision_report.json`
- `rollout.jsonl`, one auditable row per phase sample when enabled in config
- `trajectory.npz`, containing planned and realized arrays
- `metrics.json`, including every acceptance gate
- `run.log` (JSONL events)

Physics-assisted kinematic reference artifacts are isolated below
`kinematic_reference/`; they are never mixed with the measured physics
trajectory at the run root.  An early structural failure guarantees a structured
failure status but may occur before downstream rollout artifacts exist.

The fixed seed is `20260714`.  The five phases are `pregrasp`, `approach`,
`close`, `actuate`, and `retreat`.  During actuation, the door angle is sampled
uniformly, object FK gives the handle pose, and a fixed handle-to-gripper
transform generates the Cartesian target.  Sequential IK warm-starts each
waypoint from the previous solution and independently validates the result with
fresh FK, finite-value checks, and URDF joint limits.

## Affordances and collision policy

`affordances.json` stores a named handle frame with position, `wxyz` orientation,
gripper closing axis, approach axis, and recommended width.  If the requested
frame is absent, the resolver extracts the configured handle link's geometry,
generates deterministic AABB/PCA candidates, and retains all rejection reasons.
Candidates pass configured reachability and collision checks before selection.

The deterministic kinematic collision backend uses sweep-and-prune over world
AABBs followed by a 15-axis OBB separating-axis test.  The physics mode records
Newton contact evidence and compares signed effective-surface clearance against
the configured 3 mm margin.  A pair at or below that margin is forbidden unless
both named links appear in `collision.allowed_contact_links`; listing one link
does not suppress its contacts with the rest of the scene.  The margin is
applied once and is retained with the per-frame collision evidence.

The current audited collision scope is exactly
`cross_asset_robot_object`.  It does not claim robot self-collision clearance:
the cached Franka URDF has no matching SRDF/disabled-pair policy, and using whole
mesh AABB envelopes for self-collision would create untrustworthy false
positives.  Adding self-collision acceptance requires a matching named
disable-collision matrix plus a mesh-level narrow phase.

## Acceptance and tests

Acceptance thresholds live only in the JSON config.  The checked-in gates require
at least 95% IK success, median position error below 2 cm, median orientation
error below 10 degrees, no NaN/Inf, no joint-limit violations, no disallowed
collision frames, final door angle within 3 degrees, and small handle-to-gripper
drift throughout `close + actuate`.

Run the complete dependency-light test suite with:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The authoritative remote validation evidence is recorded under `reports/` and
points to the corresponding cached run directory rather than committing bulky
runtime artifacts.

## Periodic GitHub synchronization

The sync script builds its commit from an isolated Git index and stages only
source, configs, tests, documentation, and reports.  It refuses to run over
pre-existing staged work, rejects non-allowlisted paths and credential-like
content, uses `flock` to prevent overlapping jobs, and leaves environments,
cached assets, logs, NPZ files, and rollouts outside Git:

```bash
bash scripts/sync_to_github.sh
AGENTPRE_SYNC_INTERVAL_MINUTES=30 bash scripts/install_sync_cron.sh
```

The default remote is
`ssh://git@ssh.github.com:443/Da1suKE66/AgentPre.git`, which works on hosts where
GitHub's standard SSH port is blocked.
