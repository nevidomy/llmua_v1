# Experiments

`runs.json` holds the study's last 140 runs as plain commands: one entry per
run with its `argv` (runner `model_runner.py` or `sft_runner.py`, `{ROOT}`
stands for the data root) and its final metrics. `legacy_run_id` is the run
number the study's own notes used. `replay.py` runs selected entries;
`appendix_c.py` defines the earlier sweeps behind Appendix C and the 80× probes
(used by `python main_runner.py NAME`).

    python experiments/replay.py --list
    python experiments/replay.py --group paper1_final_R2 --dry-run
    python experiments/replay.py --id 92 --id 104

Entries run in `id` order, one subprocess at a time; a run whose
`ckpt_final.pt` exists is skipped unless `--force`. Replay stops on a non-zero
exit, and on exit 42 (paused).

| group | paper |
|---|---|
| `p1_cooldown_branching`, `p2_cooldown_branching`, `p3_cooldown_branching`, `p4_cooldown_branching`, `target_ladder` | Tables 2, 3, 5 |
| `p1_basesweep*` | D.1, D.3 |
| `p1_re_adamw_arm*`, `p1_adamw_batch` | section 5.2, Table 4, D.3 |
| `paper1_R1_masked_small_batch_muon` | R1 |
| `paper1_final_R2` | R2, R2b, and #139 |
| `sft_probe` | Appendix M |
| `pre_5090_compare`, `P4_batch_size_sweep` | not in the paper's tables |

The target model ran as a 40x trunk (#92), a 20x branch (#93), and then an
extension of the trunk to 80x (#104). `sft_probe` starts from those target
checkpoints, so run `target_ladder` first. Runs #66, #92 and #141 were paused and
resumed: `runs.json` holds the original launch command, and durations in their logs
cover only the part after the resume.

Notes kept in `runs.json`: #92 was paused and resumed, so its `argv` is the
original launch (without the resume checkpoint) and log durations cover only
the resumed part; #139 was launched without the frozen recipe flags.
