# AgentPre: deterministic microwave-door opening with Franka

AgentPre compiles an articulated microwave task into a deterministic six-phase
trajectory and solves it with Newton 1.3's analytic Levenberg-Marquardt IK.  It
also has a CPU-only `physics_assisted` mode: dynamic Franka bodies track the IK
trajectory through Newton XPBD joint position/velocity PD targets while Newton
advances the dynamic microwave door and the grasp constraint.  No robot body
pose or generalized force is prescribed at runtime.  The first-stage controller
is not an LLM; a future agent can call this deterministic compiler/runtime as a
tool.

The checked-in microwave URDF is a small deterministic fixture used for fast
pipeline verification.  It is clearly marked
`deterministic_test_fixture_not_articraft` and must not be presented as an
Articraft export.  A separate `configs/articraft_microwave_franka.json` targets
the real official Articraft Data record
`rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707`.  Its generated URDF and
meshes stay in `/cache/liluchen/agentpre/assets`; only the compact explicit
handle affordance and source/license metadata are checked into Git.

## Repository layout

```text
assets/       fixture plus compact Articraft affordance/source metadata
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
GPU.  Dependencies are pinned in `requirements.lock`; the target runtime is
Python 3.11, `newton==1.3.0`, `warp-lang==1.14.0`, and CPU.  Consult a validation
report tied to the same Git commit before treating a particular run as proven.

The Franka asset is copied into the AgentPre cache from the configured existing
Apache-2.0 `franka_description` tree.  `scripts/fetch_assets.py` parses the URDF,
validates every referenced mesh, and records every referenced file in a SHA-256
manifest.  It fails closed if the configured source is unavailable or the tree
is incomplete.

### Real Articraft microwave

The external source checkouts and build environment are expected outside the
repository at the following target-host paths; they are not created merely by
cloning AgentPre:

```text
/cache/liluchen/articraft           # harness, pinned source commit
/cache/liluchen/articraft-data      # sparse record checkout, pinned data commit
/cache/liluchen/articraft-env       # independent Articraft Python environment
```

With both pinned external checkouts present, install their independent
Python/uv environment under `/cache/liluchen`:

```bash
bash scripts/setup_articraft_env.sh
```

Then the build wrapper verifies both checkout commits, recompiles the record
offline with validation and strict geometry QC, applies the checked-in inertial
specification, runs AgentPre's asset inspector on the generated tree, and only
then atomically materializes the accepted runtime files:

```bash
bash scripts/build_articraft_asset.sh
```

The checked-in
`assets/articraft/rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707/inertials.json`
has reviewed `state: ready` values for all five actual record links (`cabinet`,
`door`, `turntable`, `selector_knob_0`, and `selector_knob_1`).  The values are
deterministic simulation proxies derived from the pristine compiled URDF's
per-link union collision AABB: COM is the AABB center, mass is AABB volume at an
effective density of 300 kg/m^3 with a 0.02 kg floor, and inertia is the
uniform-solid-box diagonal tensor about that center.  The specification records
the source URDF SHA-256 as a structured field and records the reviewed AABBs,
formula, and resulting values.  Injection rejects any pristine compiler output
whose bytes do not match that SHA-256.  These are deliberately identified as
collision-envelope proxies, not
manufacturer-measured masses or CAD volume integrals; no value is inferred at
runtime or hard-coded in Python.

For a finalized specification, the equivalent postprocessing and
materialization steps for an already-compiled record are:

```bash
python scripts/apply_articraft_inertials.py \
  --urdf /cache/liluchen/articraft-data/cache/record_materialization/rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707/model.urdf \
  --spec assets/articraft/rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707/inertials.json

python scripts/materialize_articraft_asset.py \
  --source-root /cache/liluchen/articraft-data/cache/record_materialization/rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707 \
  --inertial-spec assets/articraft/rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707/inertials.json \
  --inertial-sidecar /cache/liluchen/articraft-data/cache/record_materialization/rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707/agentpre_inertial_completion.json
```

The injector requires the specification link set to match the pristine URDF's
link and missing-inertial sets exactly.  A second run is idempotent only when
all existing inertials match every JSON value.  Unknown/extra links, partially
processed URDFs, non-positive mass, non-finite values, and non-positive-definite
inertia matrices, physically unrealizable rigid-body tensors, and source-URDF
hash drift are rejected before a write.  A per-URDF advisory lock covers the
whole read/validate/commit transaction, and each file is atomically replaced in
recoverable sidecar-first order: the stable
`agentpre_inertial_completion.json` records the specification path/hash,
pre/post URDF hashes, and injected link list, then the URDF is replaced.  An
interruption before the URDF replacement can be retried from the still-pristine
compiler output.

The materializer independently reloads the strict specification, requires its
record identity and data commit to match the requested asset, verifies every
URDF inertial against it and the sidecar, inventories the runtime URDF/assets,
and records the specification plus sidecar content and SHA-256 in
the immutable manifest.  The sidecar and volatile upstream `compile_report.json`
are provenance only and are not copied into the runtime asset.  Re-verifying
identical assets does not rewrite the persisted manifest.  These helpers do not
run the Articraft compiler or prove task-level acceptance by themselves.
The sidecar's absolute paths and therefore its hash are stable for the fixed
target layout documented above; changing root/cache overrides intentionally
creates different provenance.
Provenance, pinned commits, licenses, hinge geometry, and the authored
`pull_grip_center` frame are recorded under
`assets/articraft/rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707/`.

## Inspect and run

Inspect the articulated object before running:

```bash
python -m src.asset_inspector assets/microwave/microwave.urdf \
  --door-joint door_hinge \
  --door-link microwave_door \
  --handle-link handle
```

Inspect the materialized Articraft record with its real names:

```bash
python -m src.asset_inspector \
  /cache/liluchen/agentpre/assets/articraft/rec_microwave_oven_5e86f3429e954dcd9ab6c9d3a94db707/model.urdf \
  --door-joint door_hinge \
  --door-link door \
  --handle-link door
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

The wrapper's optional second argument selects another config, for example:

```bash
bash scripts/run_example.sh kinematic configs/articraft_microwave_franka.json
bash scripts/run_example.sh physics_assisted configs/articraft_microwave_franka.json
```

Those Articraft commands require the external record to have been compiled,
materialized at the configured cache path, and accepted by `asset_inspector`.
The Git checkout contains its config and compact metadata, not the generated
`model.urdf` or meshes, so cloning the repository alone is not enough to run
that config.

Run the measured Newton rollout with:

```bash
python -m src.run \
  --config configs/microwave_franka.json \
  --mode physics_assisted
```

In `physics_assisted`, the kinematic result supplies name-resolved robot joint
targets.  Franka bodies retain finite positive inverse mass and dynamic flags.
An indexed Warp kernel writes position and finite-difference velocity targets
only for the configured seven arm and two finger coordinates/DOFs; Newton XPBD
applies the configured stiffness and damping.  The measured robot coordinates,
door coordinate, TCP, and handle pose are reconstructed from the evolved body
state with Newton inverse kinematics.  A configurable 0.02 rad IK control margin
keeps the reference away from hard URDF joint limits.  Before the expensive
physics rollout starts, an independent 0.05 rad arm tracking-reserve audit
checks the complete reference against the original URDF limits and rejects any
sample that touches a realized float32 control boundary or lacks the configured
hard-limit clearance.  Measured physics acceptance still checks the original
limits with the configured numerical tolerance.
Both checked-in configs use a 1/60 s control interval, 48 physics substeps,
64 solver iterations, arm stiffness/damping 650/200, and finger
stiffness/damping 300/40.

A fixed loop grasp aid is enabled after `close`, remains active through
`actuate` and `release`, and is disabled before the first `retreat` frame.
For the checked-in schedule it activates at frame 392, remains active on
`[392, 1240)`, and is disabled at frame 1240.
Immediately before activation, the measured relation must pass a planned
15 mm / 7.5 degree pose gate and a 0.01 m/s / 5 degree/s relative anchor-twist
gate; a remote, moving, or discontinuous latch is rejected.  Only after those
gates pass does the runtime capture a coincident parent anchor from the measured
hand/handle poses.  The finalized child anchor, captured parent anchor, and all
initial-disable, activation-enable, and first-retreat-disable joint-enabled
transactions are each written once and read back.  The captured parent anchor
is also read back and checked for post-capture coincidence.  The
door reference trajectory is diagnostic only.  The implementation excludes the door
coordinate and DOF from the indexed target writer and performs no runtime write
to the door position, velocity, position/velocity target, or generalized force.
At model construction, configuration requires zero door position stiffness and
zero target velocity; a positive velocity damping gain provides passive hinge
damping without a position target.  Initial/final door target values are checked
for exact equality.  These are static indexed-control-path guarantees recorded
in the artifacts, not a claim that a runtime write interceptor observed every
possible external mutation.

Use `--output-dir /absolute/path` to select a particular run directory.  The
explicit path must be absent or empty; the CLI refuses to mix a new run with
existing evidence.  Without it, the CLI creates a run below the config's
`output.root`.  Completed rollouts
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
trajectory at the run root.  The physics result fails closed unless that
reference independently passes its configured acceptance gates.  An early
structural failure guarantees a structured failure status but may occur before
downstream rollout artifacts exist.

The fixed seed is `20260714`.  The six phases are `pregrasp`, `approach`,
`close`, `actuate`, `release`, and `retreat`.  During actuation, the door angle
is sampled with quintic smoothstep timing, object FK gives the handle pose, and
a fixed handle-to-gripper transform generates the Cartesian target.  Release
holds the goal door, handle, and TCP poses fixed while opening the fingers with
the same quintic timing; only after that does retreat move the open gripper away
from the handle.  The checked-in schedules use 224, 72, 96, 800, 48, and 96
samples respectively, for 1336 stored right-endpoint frames: `pregrasp`
0--223, `approach` 224--295, `close` 296--391, `actuate` 392--1191,
`release` 1192--1239, and `retreat` 1240--1335.  The longer close phase lowers
the finger contact speed so a one-sided contact stop remains inside the measured
finger-jerk gate.  The grasp offset keeps its
position unchanged and adds a +15 degree roll about the gripper's local closing
axis: `wxyz = [0.5609855268, -0.4304593346, -0.5609855268, -0.4304593346]`.
This keeps the late-actuation q4 reference away from its hard
lower limit without weakening the IK or physics gates.  Sequential IK
warm-starts each waypoint from the
previous solution and independently validates the result with fresh FK,
finite-value checks, and URDF joint limits.

Every stored frame is the right endpoint of one control interval; the explicit
time-zero state is the configured nominal arm pose with an open gripper, zero
joint velocity, and zero joint acceleration.  Each task phase uses a quintic
smoothstep so its continuous velocity and acceleration vanish at both ends.
Ordered IK solves add a previous-state objective and then project the Newton
candidate onto the intersection of the control-position range, each joint's
URDF velocity limit, the configured arm acceleration limit, and the configured
arm jerk limit.  Projection is resolved on the actual float32 grid used by
Newton; the same quantized state is used for FK, output, and the next frame.
Numerically identical Cartesian hold targets (including sign-equivalent unit
quaternions) skip a second LM update, so the nominal-posture objective cannot
drift through Franka's null space during `close` or `release`; the same hard
motion projector first dissipates any residual velocity/acceleration and then
holds the realized float32 joint state exactly.
Independent one-pose grasp-candidate checks do not enable the temporal
continuity objective.

Acceptance includes nominal-to-frame-zero finite differences and hard gates for
per-joint URDF velocity utilization, arm acceleration, and arm jerk.  Physics
command preflight additionally checks prismatic finger acceleration/jerk in SI
units, float32 target velocities, and two virtual terminal holds that return the
command to zero velocity and then zero acceleration.  The measured physics
result is separately gated using Newton's post-step, name-resolved arm and
finger `joint_qd`: both groups must remain within their URDF velocity limits;
the checked-in arm acceleration/jerk limits are 7.5 rad/s² and 450 rad/s³,
and the finger limits are 1.5 m/s² and 30 m/s³.  Dynamic
overshoot therefore cannot be hidden by smooth endpoint positions.
Acceleration and jerk reconstructed from adjacent `joint_q` samples remain in
the artifacts as diagnostics, but are not physics acceptance gates because
Newton inverse-coordinate reconstruction can change an equivalent generalized
coordinate representation while the authoritative `joint_qd` stays continuous.
Raw-to-projected IK motion diagnostics and measured arm/finger velocities are
retained in the rollout and trajectory artifacts.

The fixed grasp remains active through `release`.  At frame 1191 the runtime
captures the measured float32 arm equilibrium.  Over frames 1192--1239 the
applied PD target is unloaded from the planned hold to that equilibrium with an
endpoint-exact quintic blend.  The constraint is disabled at frame 1240, whose
applied target is bit-identical to the preceding release endpoint; frames
1240--1271 then blend back into the planned retreat.  These applied targets and
their finite-difference velocities are serialized alongside the unmodified
planned IK reference and are re-audited, before use, against hard position
limits, the 0.05 rad tracking reserve/control endpoints, velocity, acceleration,
and jerk limits.

## Affordances and collision policy

`affordances.json` stores a named handle frame with position, `wxyz` orientation,
gripper closing axis, approach axis, and recommended width.  If the requested
frame is absent, the resolver extracts the configured handle link's geometry,
generates deterministic AABB/PCA candidates, and retains all rejection reasons.
Candidates pass configured reachability and collision checks before selection.
The Articraft record intentionally uses an authored frame on the `door` link,
because `pull_grip` is named geometry inside that link rather than an independent
URDF link; falling back to AABB/PCA would incorrectly summarize the whole door.

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
drift throughout `close + actuate + release`.  Physics mode additionally
requires the full reference-reserve audit, measured arm/finger velocity,
acceleration, and jerk gates, and successful fixed-joint capture/release
transactions with their readbacks.

Run the complete unittest suite in the configured project environment with:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

`reports/` contains commit-specific validation snapshots and may describe an
older controller or asset state.  A report is evidence only for the exact Git
commit and cached run directory that it names; it must not be used to infer that
newer unreported changes or the external Articraft config have passed.

## Periodic GitHub synchronization

The sync script builds its commit from an isolated Git index and stages only
the explicit project allowlist (`assets/`, `configs/`, `scripts/`, `src/`,
`tests/`, `outputs/`, `reports/`, and the named top-level project files).
`.gitignore` keeps `outputs/README.md` while excluding runtime output.  The
script refuses to run over
pre-existing staged work, rejects non-allowlisted paths and credential-like
content, uses `flock` to prevent overlapping jobs, and leaves environments,
cached assets, logs, NPZ files, and rollouts outside Git.  The first two commands
below are mutating: the one-shot command commits and pushes, while installation
starts the loop and performs its first sync immediately.

```bash
bash scripts/sync_to_github.sh       # one commit/push attempt
bash scripts/install_sync_daemon.sh  # start loop; first attempt is immediate
bash scripts/sync_daemon.sh status   # read current PID and heartbeat
bash scripts/sync_daemon.sh stop     # stop without force-killing
```

`sync_daemon.sh` is a user-owned, single-instance loop and does not depend on a
system `cron` or `systemd` service.  Its PID, lock, heartbeat, and log live under
`/cache/liluchen/agentpre`; it hides CUDA/ROCm and caps numerical-library thread
counts even though the sync operation is Git-only.  The default interval is 30
minutes.  Repeated `start` and `stop` calls are safe; `status` is read-only and
exits 0 while running or 3 while stopped.  Before starting, the installer uses
`crontab` when available to remove only the exact legacy AgentPre marker and its
matching following sync command; isolated markers and all unrelated entries are
preserved.  Absence of the
`crontab` command does not block the user daemon.
`scripts/install_sync_cron.sh` is a legacy alternative for a host with a
confirmed active cron service: stop the user daemon first, and never install
both schedulers concurrently.

The default remote is
`ssh://git@ssh.github.com:443/Da1suKE66/AgentPre.git`, using GitHub's SSH endpoint
on port 443 for hosts where the standard SSH port is blocked.
