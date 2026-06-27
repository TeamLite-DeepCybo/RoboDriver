# RoboDriver Collection Interaction Handoff

This document is the integration contract between the dual-Lite teleop FSM and
the RoboDriver/LeRobot data recorder.

本文档是双 Lite 主从遥操 FSM 与 RoboDriver/LeRobot 数据落盘端之间的交互契约。

## TL;DR / 快速结论

- FSM entrypoint: `src/bar_ros2/ops/lite/scripts/bilateral_fsm_ready.sh`
- Recorder trigger topics: `/to_robodriver/start_collect`,
  `/to_robodriver/finish_collect`, `/to_robodriver/affirm_to_collect`
- Message type: `std_msgs/msg/Bool`
- QoS expected by recorder: RELIABLE + TRANSIENT_LOCAL compatible
- Recorder should treat messages as state changes, not as counted pulses
- Segment is opened by `start_collect=true`
- Segment is closed/flushed by `finish_collect=true`
- Closed segment is kept only after `affirm_to_collect=true`
- Closed segment is discarded after `affirm_to_collect=false`
- If a segment is finished but no affirmation arrives, treat it as pending or
  operator-interrupted

核心规则：

- FSM 入口：`src/bar_ros2/ops/lite/scripts/bilateral_fsm_ready.sh`
- recorder 触发话题：`/to_robodriver/start_collect`、
  `/to_robodriver/finish_collect`、`/to_robodriver/affirm_to_collect`
- 消息类型：`std_msgs/msg/Bool`
- recorder 订阅 QoS 建议兼容 RELIABLE + TRANSIENT_LOCAL
- recorder 应把消息当作状态变化处理，不要按消息数量计数
- `start_collect=true` 打开一个 episode/segment
- `finish_collect=true` 关闭并 flush 当前 episode/segment
- `affirm_to_collect=true` 表示保留刚关闭的 episode/segment
- `affirm_to_collect=false` 表示丢弃刚关闭的 episode/segment
- 如果已经 finish 但没有收到 affirmation，应按 pending 或人工中断处理

## Process Ownership / 进程分工

| Side | Responsibility |
|---|---|
| Teleop FSM | drives controller state transitions and publishes recorder commands |
| Controller stacks | launched by `ros2 launch bar_bringup_lite dual_bilateral.launch.py start_bridge:=true` |
| RoboDriver recorder | subscribes to `/to_robodriver/*`, records ROS topics, saves/discards episodes |

| 侧别 | 职责 |
|---|---|
| Teleop FSM | 驱动 controller 状态跳转，并发布 recorder 控制命令 |
| Controller stacks | 通过 `ros2 launch bar_bringup_lite dual_bilateral.launch.py start_bridge:=true` 启动 |
| RoboDriver recorder | 订阅 `/to_robodriver/*`，记录 ROS 话题，并保存/丢弃 episode |

## Topic Contract / 话题契约

All messages are `std_msgs/msg/Bool`.

所有消息类型均为 `std_msgs/msg/Bool`。

| Topic | `true` meaning | `false` meaning | Recorder action |
|---|---|---|---|
| `/to_robodriver/start_collect` | open a segment | clear start latch | create/open a new episode if not already collecting |
| `/to_robodriver/finish_collect` | close a segment | clear finish latch | close and flush active episode if collecting |
| `/to_robodriver/affirm_to_collect` | keep latest closed segment | discard latest closed segment | commit or delete the pending episode |

| 话题 | `true` 含义 | `false` 含义 | recorder 动作 |
|---|---|---|---|
| `/to_robodriver/start_collect` | 打开一段采集 | 清除 start latch | 如果当前未采集，则创建/打开新 episode |
| `/to_robodriver/finish_collect` | 关闭一段采集 | 清除 finish latch | 如果当前正在采集，则关闭并 flush 当前 episode |
| `/to_robodriver/affirm_to_collect` | 保留刚关闭的数据段 | 丢弃刚关闭的数据段 | commit 或删除 pending episode |

The FSM publishes with RELIABLE + TRANSIENT_LOCAL QoS and repeats each state
edge three times. This improves delivery during node discovery, but RoboDriver
must still deduplicate by state transition.

FSM 使用 RELIABLE + TRANSIENT_LOCAL QoS 发布，并对每个状态边沿重复发布三次。
这样可以提高节点发现期间的交付概率，但 RoboDriver 仍必须按状态变化去重。

## Recommended Recorder State Machine / 推荐 recorder 状态机

Use an internal recorder state machine similar to:

```text
IDLE
  on start_collect=true:
    open episode
    state = COLLECTING

COLLECTING
  on finish_collect=true:
    close and flush episode
    state = WAITING_AFFIRM

WAITING_AFFIRM
  on affirm_to_collect=true:
    keep episode
    state = IDLE
  on affirm_to_collect=false:
    discard episode
    state = IDLE
```

Late or stale messages should be harmless:

- Ignore `finish_collect=true` while `IDLE`.
- Ignore duplicate `start_collect=true` while `COLLECTING`.
- Ignore duplicate `finish_collect=true` while `WAITING_AFFIRM`.
- Ignore `affirm_to_collect=*` if there is no pending closed episode.

推荐 RoboDriver 内部使用类似状态机：

```text
IDLE
  收到 start_collect=true:
    打开 episode
    state = COLLECTING

COLLECTING
  收到 finish_collect=true:
    关闭并 flush episode
    state = WAITING_AFFIRM

WAITING_AFFIRM
  收到 affirm_to_collect=true:
    保留 episode
    state = IDLE
  收到 affirm_to_collect=false:
    丢弃 episode
    state = IDLE
```

晚到或陈旧消息应无害化处理：

- `IDLE` 状态收到 `finish_collect=true`，忽略。
- `COLLECTING` 状态重复收到 `start_collect=true`，忽略。
- `WAITING_AFFIRM` 状态重复收到 `finish_collect=true`，忽略。
- 没有 pending closed episode 时收到 `affirm_to_collect=*`，忽略。

## Segment Timeline / 单段时序

The controller reset and standby phase happens before recording starts:

```text
whatever robot mode
-> master/slave: zero_torque_controller
-> master/slave: damping_controller
-> master/slave: standby_controller
-> wait until both standby_controller/state report fresh is_finished=true
-> wait for operator **[Enter]**
-> countdown
```

Recorder-visible timeline:

```text
teleop starts:
  master: zero_torque_controller
  slave : remote_policy_controller
  publish /to_robodriver/finish_collect = false
  publish /to_robodriver/start_collect  = true

operator finishes segment with **[Enter]**:
  publish /to_robodriver/start_collect  = false
  publish /to_robodriver/finish_collect = true

operator confirms quality with **[Y/y]** or **[N/n]**:
  publish /to_robodriver/affirm_to_collect = true   for **[Y/y]**
  publish /to_robodriver/affirm_to_collect = false  for **[N/n]**
```

The affirmation is deliberately after `finish_collect=true`, so the recorder
can close and flush before deciding whether to keep or discard the segment.

人工确认刻意放在 `finish_collect=true` 之后，方便 recorder 先关闭/flush 当前
segment，再根据人工判断保留或丢弃。

## Operator UX Contract / 操作员交互约束

The FSM terminal prompts default to Chinese. English prompts are available with
`--language en` or `BILATERAL_FSM_LANGUAGE=en`.

FSM 终端提示默认中文。如需英文提示，可使用 `--language en` 或
`BILATERAL_FSM_LANGUAGE=en`。

Keyboard gates:

- Start teleop: empty **[Enter]** line only
- Finish segment: empty **[Enter]** line only
- Keep segment: exact single-character **[Y/y]**
- Discard segment: exact single-character **[N/n]**
- Undefined input reprints the same prompt

键盘 gate：

- 开始遥操：只接受空行 **[Enter]**
- 结束本段：只接受空行 **[Enter]**
- 保存本段：只接受精确单字符 **[Y/y]**
- 丢弃本段：只接受精确单字符 **[N/n]**
- 未定义输入会重新打印同一个输入提示

The FSM displays an operator-facing saved counter in bold green. It increments
only after **[Y/y]** and does not increment after **[N/n]**. This counter is for
operator feedback only; RoboDriver should use its own committed episode count
as the source of truth.

FSM 会用绿色加粗显示面向操作员的成功保存段数。该计数只在 **[Y/y]** 后累计，
**[N/n]** 后不累计。这个计数仅用于操作员反馈；RoboDriver 应以自身已 commit 的
episode 数量作为真实数据源。

## Exit And Interruption / 退出与中断

When `--max-cycles` is reached, or when the operator presses **[Ctrl+C]**, the
FSM performs:

```text
if currently collecting:
  publish /to_robodriver/start_collect=false
  publish /to_robodriver/finish_collect=true

master/slave -> damping_controller
sleep 5 seconds
master/slave -> zero_torque_controller
process exits
```

If **[Ctrl+C]** happens after `finish_collect=true` but before **[Y/y]** or
**[N/n]**, the FSM does not publish `affirm_to_collect`. RoboDriver should leave
that episode in a pending/interrupted state or apply its own cleanup policy.

如果 **[Ctrl+C]** 发生在 `finish_collect=true` 之后、**[Y/y]** 或 **[N/n]**
之前，FSM 不会发布 `affirm_to_collect`。RoboDriver 应将该 episode 保持为
pending/interrupted，或按自身策略清理。

## Late Subscriber Notes / 晚启动订阅者注意事项

Because the topics are TRANSIENT_LOCAL, a late RoboDriver subscriber may receive
the last latched values immediately. Recommended handling:

- If `start_collect=true` is received while `IDLE`, open a segment. This allows
  RoboDriver to attach to an already-started collection window.
- If `finish_collect=true` is received while `IDLE`, ignore it as stale.
- If `affirm_to_collect=*` is received without a pending closed segment, ignore
  it as stale.
- Persist enough local recorder state to avoid committing or deleting the wrong
  episode after a process restart.

由于话题使用 TRANSIENT_LOCAL，晚启动的 RoboDriver 订阅者可能会立即收到最近的
latched 值。推荐处理方式：

- `IDLE` 状态收到 `start_collect=true` 时打开 segment，允许 recorder 接入一个
  已经开始的采集窗口。
- `IDLE` 状态收到 `finish_collect=true` 时视为陈旧消息并忽略。
- 没有 pending closed segment 时收到 `affirm_to_collect=*`，视为陈旧消息并忽略。
- recorder 应保存足够的本地状态，避免进程重启后 commit 或 delete 错误的 episode。

## Minimal Recorder Pseudocode / recorder 伪代码

```python
state = 'IDLE'
pending_episode = None

def on_start_collect(value: bool):
    global state
    if value and state == 'IDLE':
        open_episode()
        state = 'COLLECTING'

def on_finish_collect(value: bool):
    global state, pending_episode
    if value and state == 'COLLECTING':
        pending_episode = close_and_flush_episode()
        state = 'WAITING_AFFIRM'

def on_affirm_to_collect(value: bool):
    global state, pending_episode
    if state != 'WAITING_AFFIRM' or pending_episode is None:
        return
    if value:
        commit_episode(pending_episode)
    else:
        discard_episode(pending_episode)
    pending_episode = None
    state = 'IDLE'
```

## Test Checklist / 测试清单

Static checks:

```bash
cd ~/Desktop/bar_ws
python3 -m py_compile src/bar_ros2/ops/lite/scripts/bilateral_fsm_loop.py
bash -n src/bar_ros2/ops/lite/scripts/bilateral_fsm_ready.sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash
src/bar_ros2/ops/lite/scripts/bilateral_fsm_ready.sh --help
```

Observe recorder commands:

```bash
ros2 topic echo /to_robodriver/start_collect std_msgs/msg/Bool
ros2 topic echo /to_robodriver/finish_collect std_msgs/msg/Bool
ros2 topic echo /to_robodriver/affirm_to_collect std_msgs/msg/Bool
```

Run one segment:

```bash
src/bar_ros2/ops/lite/scripts/verify_dual_controllers.sh
src/bar_ros2/ops/lite/scripts/bilateral_fsm_ready.sh --max-cycles 1 --start-delay 0.5
```

Run with English plain-text prompts:

```bash
src/bar_ros2/ops/lite/scripts/bilateral_fsm_ready.sh \
  --max-cycles 1 \
  --start-delay 0.5 \
  --language en \
  --no-color
```

Expected recorder events:

```text
START **[Enter]**:
  start_collect=true
  finish_collect=false

FINISH **[Enter]**:
  start_collect=false
  finish_collect=true

**[Y/y]**:
  affirm_to_collect=true
  recorder commits the pending episode

**[N/n]**:
  affirm_to_collect=false
  recorder discards the pending episode

normal max-cycles exit:
  master/slave damping_controller for 5s, then zero_torque_controller
```
