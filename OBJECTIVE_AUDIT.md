# Objective audit

The attachment defines a broad model assessment and reverse-engineering workbench, an agent interface, native workflows, a real-model demonstration, visualization, repair and defensible research positioning. The implementation language is Rust.

| Requirement | Authoritative evidence | Current scope |
| --- | --- | --- |
| Accept model files and inspect weights | `scan`, safetensors validation and full element traversal in `src/main.rs`; `examples/qwen-demo.json` | Full supported-element numerical checks and exact reference differences. General semantic vulnerability discovery remains incomplete. |
| Localize a problem | Changed index and nonfinite count in `examples/qwen-demo.json`; raw tile comparisons | One deliberately seeded NaN localized in a real Qwen query projection. A naturally occurring security vulnerability has not been established. |
| Native workflows | CLI assessment, graph, slice, tile, experiment, repair, evaluate and runtime-compare commands | Working Rust workflows with measured checkpoint and runtime evidence. |
| Agent connection | Stdio MCP implementation and tools/list in `src/main.rs`; `src/trace_report.rs` | Assessment, graph metadata, bounded flattened slices, matrix tiles and filtered recorded activation observations. |
| Small real-model proof of concept | Qwen2.5 1.5B Instruct, 1,543,714,304 stored parameters, 338 tensors | Model download digest, numerical experiment and full-model runtime comparison are preserved in `examples`. |
| Demonstrate local impact and repair | `examples/qwen-runtime.json`, repaired checkpoint manifest in the generated run | Candidate nonfinite logits on three prompt evaluations; repaired checkpoint, generated tokens and logit hashes match the reference. |
| Visualize the network and nodes | `src/atlas.rs`, `web/atlas.js`, `src/traced_qwen.rs`, graph and tile APIs | Hierarchical all-tensor inventory, validated Qwen2 module topology, paged actual weight connections and recorded numerical execution overlays. Individual neuron semantics and general causal interpretation remain incomplete. |
| Show patch process | Before/after findings, selected-tensor restoration, manifest, file hashes and runtime comparison | Exact restoration from a supplied compatible trusted checkpoint. Learning an original semantic repair without a reference remains research work. |
| Correct conceptual gaps | `RESEARCH.md`, `DEMO.md` | Explicit serving-runtime isolation boundary, seeded-fault disclosure, research references and limits. |
| Establish novelty | Prior-work comparison in `RESEARCH.md` | Integrated engineering workflow demonstrated. Algorithmic novelty and a new security vulnerability class remain unproven. |
| Prepare a presentation | `DEMO.md`, committed evidence snapshots, loopback viewer | Reproducible commands and a presentation sequence. |
| Use Rust and maintain the repository | Cargo sources, tests and Git history | Rust implementation with a private repository named LLMIUM. |

The seeded numerical-fault demonstration supports availability and restoration observations under the recorded settings. It does not prove cross-user data access or identify a universal malicious weight. Those broader claims require a concrete serving environment, a reproducible isolation failure and additional evidence. A four-token continuation experiment also does not establish reasoning-loop diagnosis or general model quality improvement.
