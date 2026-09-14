# H3 native long-form delivery

Owner: /root/consumer_delivery_finish. Worktree ~/cozy/.worktrees/packages/h3-long-form-native-assembly,
branch feat/h3-long-form-native-assembly, fetched base f62101b.
Tracker: se-057; root owns actual paid CLI/capacity and the 2/4/8-shot qualification.

The implementation restores the reviewed streaming assembler inside the H3 package,
using Runtime MediaDecoder and Outputs. It will return a playable video and a native
prefix Tree, preserve per-shot native bytes, bind prefix reuse to exact shot/model/code
identity, and keep failures resumable without memoizing inference.

The old assembler's frame/sample clock, gain-ride and wrong-master controls are reused.
One-shot prefixes are admitted. A helper runs assembly inside the current admitted
attempt, while assemble_video remains an ordinary CPU callable job.

Runtime owners separately supply the Model-bearing segment job defaults and native
Tree-member typed capability. No new package, workflow DSL, storage authority or local
H3 GPU invocation is introduced. The primary's concurrent fidelity changes remain untouched.
