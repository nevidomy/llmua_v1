"""Named run sets for the sweeps of Appendix C that are not in runs.json.

Each function returns (runid, [RunConfig, ...]) like the ones in
main_runner.py and carries the section title of the author's lab notes
(note.txt) in its docstring. Grids are the ones listed in those notes; the
launch commands come from the git history of main_runner.py. Runs that the
notes mark as "reused" (seed 47 of a seed sweep taken from an LR sweep) are
included again: they are the same config and the same run name.

    python main_runner.py --list
    python main_runner.py p2_lr_grid p2_base_5_seeds

Early P1 sets (up to the initialization fix) use enable_init, exactly as they
were launched. Names follow the old launcher, so they differ from the
names in runs.json.
"""

from main_runner import (RunConfig, adamw_only, cooldown_branching, lr_sweep,
                         no_qk_norm, p1_best_sweep)


# ------------------------------------------------------------------ P1
def initial_floor_sweep(learning_rate=1e-3, seeds=range(47, 47 + 8)):
    """Base (noise level) sweep for cosine scheduler"""
    return "initial_floor_sweep", [
        RunConfig(
            run=f"Phase0_initial_Sweep_seed_{seed}", model="P1", seed=seed,
            lr=learning_rate,
            enable_init=True, enable_swiglu=True, enable_rms_norm=True,
            enable_qk_norm=True, use_muon=False, ckpt_final=True)
        for seed in seeds
    ]


def initial_floor_sweep_wsd(learning_rate=1e-3, seeds=range(47, 47 + 8)):
    """Base (noise level) sweep for WSD scheduler"""
    return "initial_floor_sweep_wsd", [
        RunConfig(
            run=(f"Phase0_initial_Sweep_wsd_seed_{seed}"
                 if learning_rate == 1e-3 else
                 f"Phase0_initial_Sweep_wsd_lr_{learning_rate}_seed_{seed}"),
            model="P1", seed=seed, schedule="wsd", lr=learning_rate,
            enable_init=True, enable_swiglu=True, enable_rms_norm=True,
            enable_qk_norm=True, use_muon=False, ckpt_final=True)
        for seed in seeds
    ]


def phase0_lr_sweep(schedule="cosine",
                    learning_rates=(2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2, 6.4e-2)):
    """Learning Rate sweep (cosine, seed 47) / Learning Rate sweep (WSD, seed 47)

    1e-3 is the seed-47 run of the base sweep."""
    return f"Phase0_P1_lr_{schedule}", [
        RunConfig(
            run=f"Phase0_P1_lr_{schedule}_{lr}_s47", model="P1", seed=47,
            schedule=schedule, lr=lr,
            enable_init=True, enable_swiglu=True, enable_rms_norm=True,
            enable_qk_norm=True, use_muon=False, ckpt_final=True)
        for lr in learning_rates
    ]


def p1_lr_sweep_cosine():
    """Learning Rate sweep (cosine, seed 47)"""
    return phase0_lr_sweep("cosine")


def p1_lr_sweep_wsd():
    """Learning Rate sweep (WSD, seed 47)"""
    return phase0_lr_sweep("wsd", (2e-3, 4e-3, 6e-3, 8e-3, 1e-2, 1.2e-2,
                                   1.4e-2, 1.6e-2, 3.2e-2))


def p1_seed_noise_wsd():
    """Seed noise sweep for WSD @ lr=8e-3 (seed 47 is the lr-sweep run)"""
    return initial_floor_sweep_wsd(8e-3, seeds=range(48, 47 + 8))


def muon_arch_sweep_wsd(model="P1", adamw_lrs=(8e-3,), muon_lrs=(0.01,),
                        seeds=(47,)):
    return f"{model}_muon_arch_sweep_wsd", [
        RunConfig(
            run=f"{model}_adamw_lr_{adamw_lr}_muon_lr_{muon_lr}_seed_{seed}",
            model=model, seed=seed, schedule="wsd",
            lr=adamw_lr, muon_lr=muon_lr,
            enable_init=True, enable_swiglu=True, enable_rms_norm=True,
            enable_qk_norm=True, use_muon=True, ckpt_final=True)
        for seed in seeds
        for adamw_lr in adamw_lrs
        for muon_lr in muon_lrs
    ]


def p1_muon_lr_sweep_pre_fix():
    """Muon LR sweep · WSD · adamw lr=8e-3 · seed 47"""
    return muon_arch_sweep_wsd(
        muon_lrs=(0.006, 0.008, 0.01, 0.02, 0.03, 0.06, 0.09, 0.12))


def arch_sweep_wsd(model="P1", learning_rates=(4e-3, 8e-3, 1.6e-2),
                   seeds=(47,), selected_features=None):
    """Architecture ablation × LR · WSD · seed 47

    Base config plus one-flag ablations, expanded over seeds × lrs."""
    base = dict(model=model, schedule="wsd",
                enable_init=True, enable_swiglu=True, enable_rms_norm=True,
                enable_qk_norm=True, use_muon=False, ckpt_final=True)
    ablations = {
        "no_init":     {"enable_init": False},
        "no_swiglu":   {"enable_swiglu": False},
        "no_rms_norm": {"enable_rms_norm": False},
        "no_qk_norm":  {"enable_qk_norm": False},
    }
    if selected_features is not None:
        ablations = {k: v for k, v in ablations.items()
                     if k in selected_features}
    return f"{model}_arch_sweep_wsd", [
        RunConfig(run=f"{model}_arch_sweep_wsd_lr_{lr}_seed_{seed}_{tag}",
                  seed=seed, lr=lr, **{**base, **override})
        for tag, override in ablations.items()
        for lr in learning_rates
        for seed in seeds
    ]


def p1_arch_sweep_pre_fix():
    """Architecture ablation × LR · WSD · seed 47"""
    return arch_sweep_wsd()


def initial_floor_sweep_new(model="P1", schedule="wsd",
                            learning_rates=(1e-3,), seeds=(47,)):
    return "initial_floor_sweep_wsd", [
        RunConfig(
            run=f"{model}_initialsweep_{schedule}_lr_{lr}_seed_{seed}",
            model="P1", seed=seed, schedule=schedule, lr=lr,
            enable_init=True, enable_swiglu=True, enable_rms_norm=True,
            enable_qk_norm=True, use_muon=False, ckpt_final=True)
        for seed in seeds
        for lr in learning_rates
    ]


def p1_seed_noise_post_fix():
    """Rerun of initial seed sweep after fixing weight initialization"""
    return initial_floor_sweep_new(learning_rates=(8e-3,),
                                   seeds=range(47, 47 + 8))


def p1_lr_sweep_post_fix():
    """Resweep for Learning rate after fixing weight initialization"""
    return initial_floor_sweep_new(learning_rates=(6e-3, 8e-3, 1e-2))


def init_variations_sweep(model="P1", schedule="wsd", learning_rates=(8e-3,),
                          seeds=(47,), no_biases_vals=(True, False),
                          custom_init_vals=(True, False),
                          residual_init_vals=(True, False), times=None):
    """B (no_biases) × C (custom_init) × R (residual_init) factorial."""
    return f"{model}_init_variants_sweep", [
        RunConfig(
            run=(f"{model}_init_variants_{schedule}_lr_{lr}_initBCR_"
                 f"{b}_{c}_{r}_seed_{seed}" + ("" if times is None
                                               else f"_t{t}")),
            model=model, seed=seed, schedule=schedule, lr=lr,
            enable_init=False, enable_swiglu=True, enable_rms_norm=True,
            enable_qk_norm=True, use_muon=False,
            no_biases=b, custom_init=c, residual_init=r, ckpt_final=True)
        for seed in seeds
        for lr in learning_rates
        for b in no_biases_vals
        for c in custom_init_vals
        for r in residual_init_vals
        for t in range(times or 1)
    ]


def p1_init_factorial():
    """Initialization variants sweep · BCR × LR · WSD · seed 47"""
    return init_variations_sweep(learning_rates=(2e-3, 4e-3, 8e-3))


def p1_best_init_3_seeds():
    """Best init · 3-seed check · BCR TTF (no_biases + custom_init) · WSD · lr=4e-3

    Seed 47 is the factorial run."""
    return init_variations_sweep(learning_rates=(4e-3,), seeds=(48, 49),
                                 no_biases_vals=(True,),
                                 custom_init_vals=(True,),
                                 residual_init_vals=(False,))


def p1_init_ttt_double_check():
    """Double-check · 3 trials · BCR TTT · WSD · lr=8e-3 · seed 47"""
    return init_variations_sweep(learning_rates=(8e-3,), times=3,
                                 no_biases_vals=(True,),
                                 custom_init_vals=(True,),
                                 residual_init_vals=(True,))


def p1_muon_lr_sweep_post_fix():
    """Muon LR sweep (post init-fix) · WSD · adamw lr=8e-3 · seed 47"""
    return muon_arch_sweep_wsd(muon_lrs=(0.002, 0.006, 0.01, 0.03))


def p1_adamw_lr_sweep_muon_0_01():
    """AdamW LR sweep · muon_lr=0.01 fixed · WSD · seed 47

    8e-3 is the muon sweep run."""
    return muon_arch_sweep_wsd(adamw_lrs=(4e-3, 1.6e-2), muon_lrs=(0.01,))


def p1_muon_best_3_seeds():
    """Muon best config · 3-seed sweep · adamw=8e-3 · muon_lr=0.01 · WSD

    Seed 47 is the muon sweep run."""
    return muon_arch_sweep_wsd(seeds=(48, 49))


def p1_arch_lr_sweep_post_fix():
    """no_swiglu + no_rms_norm LR sweep (post init-fix) · WSD · seed 47"""
    _, a = arch_sweep_wsd(learning_rates=(1e-3, 2e-3, 4e-3, 8e-3),
                          selected_features=["no_swiglu"])
    _, b = arch_sweep_wsd(learning_rates=(1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2),
                          selected_features=["no_rms_norm"])
    return "P1_arch_sweep_wsd", a + b


def p1_no_swiglu_5_seeds():
    """No SwiGLU · 5-seed sweep · WSD · lr=4e-3 (seed 47 reused)"""
    return arch_sweep_wsd(learning_rates=(4e-3,), seeds=range(47, 52),
                          selected_features=["no_swiglu"])


def p1_no_rms_norm_5_seeds():
    """No RMS Norm · 5-seed sweep · WSD · lr=8e-3 (seed 47 reused)"""
    return arch_sweep_wsd(learning_rates=(8e-3,), seeds=range(47, 52),
                          selected_features=["no_rms_norm"])


def p1_no_qk_norm_3_seeds():
    """No QK Norm · 3-seed sweep · WSD · lr=8e-3"""
    return arch_sweep_wsd(learning_rates=(8e-3,), seeds=range(47, 50),
                          selected_features=["no_qk_norm"])


def p1_final_muon_lr_fit():
    """Final config · Muon LR fit · adamw lr=8e-3 · WSD · seed 47"""
    return p1_best_sweep(learning_rates=[8e-3],
                         muon_lrs=[0.003, 0.006, 0.01, 0.02], seeds=[47])


def p1_final_5_seeds():
    """Final config · 5-seed sweep · adamw lr=4e-3 · muon_lr=0.01 · WSD"""
    return p1_best_sweep(learning_rates=[4e-3], muon_lrs=[0.01],
                         seeds=[47, 48, 49, 50, 51])


def p1_adam_only_lr_sweep():
    """Adam only LR Sweep · WSD · seed 47"""
    return adamw_only(model="P1", seeds=[47],
                      learning_rates=[2e-3, 4e-3, 6e-3, 9e-3])


def p1_cooldown_branching():
    """Cooldown branching · WSD · seed 47"""
    return cooldown_branching(model="P1", seed=47, learning_rate=4e-3,
                              muon_lr=0.01)


def p1_adam_refit_3_seeds():
    """P1 multiseed adam refit · muon_lr=0.00816 · WSD · 3 seeds × 5 LRs"""
    return lr_sweep(model="P1", seeds=[47, 48, 49],
                    learning_rates=[1.5e-3, 3e-3, 6e-3, 1.2e-2, 2.4e-2],
                    muon_lrs=[0.00816])


# ------------------------------------------------------------------ P2
def p2_lr_grid():
    """LR Sweep · adamw × muon · WSD · seed 47"""
    cells = [
        ([1e-4, 3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3], [0.01]),
        ([1e-3, 2e-3, 4e-3, 8e-3], [0.005, 0.02]),
        ([2e-3, 4e-3, 8e-3], [0.002]),
        ([2e-3, 4e-3], [0.04]),
    ]
    runs = []
    for lrs, mlrs in cells:
        runs += lr_sweep(model="P2", seeds=[47], learning_rates=lrs,
                         muon_lrs=mlrs)[1]
    return "P2_lr_sweep", runs


def p2_base_5_seeds():
    """Base sweep · 5-seed · adamw lr=4e-3 · muon_lr=0.005 · WSD"""
    return lr_sweep(model="P2", seeds=[47, 48, 49, 50, 51],
                    learning_rates=[4e-3], muon_lrs=[0.005])


def p2_residual_init_5_seeds():
    """Residual init on · 5-seed · adamw lr=4e-3 · muon_lr=0.005 · WSD"""
    return lr_sweep(model="P2", seeds=[47, 48, 49, 50, 51],
                    learning_rates=[4e-3], muon_lrs=[0.005],
                    residual_init=True)


def p2_adam_only_lr_sweep():
    """Adam only LR Sweep · WSD · seed 47"""
    return adamw_only(model="P2", seeds=[47],
                      learning_rates=[2e-3, 4e-3, 6e-3, 9e-3])


def p2_adam_only_5_seeds():
    """AdamW only · 5-seed · adamw lr=4e-3 · WSD"""
    return adamw_only(model="P2", seeds=[47, 48, 49, 50, 51],
                      learning_rates=[4e-3])


def p2_no_qk_norm_3_seeds():
    """No QK-norm · 3-seed · adamw lr=4e-3 · muon_lr=0.005 · WSD"""
    return no_qk_norm(model="P2", seeds=[47, 48, 49], learning_rate=4e-3,
                      muon_lr=0.005)


def p2_cooldown_branching():
    """Cooldown branching · WSD · seed 47 · adamw lr=4e-3 · muon_lr=0.005"""
    return cooldown_branching(model="P2", seed=47, learning_rate=4e-3,
                              muon_lr=0.005)


def p2_cooldown_branching_adamw_only():
    """Cooldown branching · AdamW only · WSD · seed 47 · adamw lr=4e-3"""
    return cooldown_branching(model="P2", seed=47, learning_rate=4e-3,
                              muon_lr=0.005, adamw_only=True)


def p2_batching_validation():
    """Batching validation · P2 seed 47 · adamw lr=4e-3 · muon_lr=0.005 · WSD

    Effective batch 16 = micro_batch_size × micro_batch_count."""
    return "P2_batching_validation", [
        RunConfig(
            run=f"P2_batching_mbs{mbs}_mbc{mbc}", model="P2", seed=47,
            schedule="wsd", lr=4e-3, muon_lr=0.005,
            micro_batch_size=mbs, micro_batch_count=mbc,
            enable_init=False, enable_swiglu=True, enable_rms_norm=True,
            enable_qk_norm=True, use_muon=True,
            no_biases=True, custom_init=True, residual_init=False,
            ckpt_final=True)
        for mbs, mbc in ((16, 1), (8, 2), (4, 4))
    ]


# ------------------------------------------------------------------ P3
def p3_muon_lr_sweep():
    """Muon LR sweep · WSD · adamw lr=4e-3 · seed 47"""
    return lr_sweep(model="P3", seeds=[47], learning_rates=[4e-3],
                    muon_lrs=[0.002, 0.003, 0.0045, 0.007])


def p3_adamw_lr_sweep():
    """AdamW LR sweep · muon_lr=0.00365 fixed · WSD · seed 47"""
    return lr_sweep(model="P3", seeds=[47], learning_rates=[2e-3, 3e-3, 4.5e-3],
                    muon_lrs=[0.00365])


def p3_base_5_seeds():
    """P3 5-seed base · adamw lr=3e-3 · muon_lr=0.00365 · WSD"""
    return lr_sweep(model="P3", seeds=[47, 48, 49, 50, 51],
                    learning_rates=[3e-3], muon_lrs=[0.00365])


def p3_muon_lr_sweep_2():
    """Muon LR sweep 2 · 20× · WSD · adamw lr=3e-3 · seed 47"""
    return lr_sweep(model="P3", seeds=[47], learning_rates=[3e-3],
                    muon_lrs=[0.002, 0.003, 0.0045, 0.007])


def p3_cooldown_branching():
    """P3_cooldown_branching · WSD · seed 47 · adamw lr=3e-3 · muon_lr=0.00365"""
    return cooldown_branching(model="P3", seed=47, learning_rate=3e-3,
                              muon_lr=0.00365)


# ------------------------------------------------------------------ P4
def p4_base_3_seeds():
    """P4 3-seed baseline · adamw lr=2.14e-3 · muon_lr=0.002 · WSD"""
    return lr_sweep(model="P4", seeds=[47, 48, 49], learning_rates=[2.14e-3],
                    muon_lrs=[0.002])


def p4_cooldown_branching():
    """Cooldown branching · WSD · seed 47 · adamw lr=2.14e-3 · muon_lr=0.002"""
    return cooldown_branching(model="P4", seed=47, learning_rate=2.14e-3,
                              muon_lr=0.002)


# ------------------------------------------------- doc mask / RoPE reset
_DOC_ARMS = {
    # model: (adamw lr, muon lr, seeds of base and docmask, seeds of docrope)
    "P1": (2.4e-2, 0.00816, (47, 48, 49), (47, 48, 49)),
    "P2": (4e-3, 0.005, (47, 48, 49), (47,)),
    "P3": (3e-3, 0.00365, (47, 48, 49), (47,)),
    "P4": (2.14e-3, 0.002, (47,), (47,)),
}


def doc_mask_arms(model):
    """Shape × seed × feature · docmask / docrope · WSD · 20×

    base = no doc masking; docmask = doc_masking; docrope = doc_masking +
    doc_rope. docrope has seed 47 only for P2 and P3."""
    lr, muon_lr, seeds, rope_seeds = _DOC_ARMS[model]
    arms = [("base", seeds, {}),
            ("docmask", seeds, dict(doc_masking=True)),
            ("docrope", rope_seeds, dict(doc_masking=True, doc_rope=True))]
    return f"{model}_doc_mask_arms", [
        RunConfig(
            run=f"{model}_seed{seed}_lr{lr}_muon_mlr{muon_lr}_wsd_{arm}",
            model=model, seed=seed, schedule="wsd", lr=lr, muon_lr=muon_lr,
            enable_init=False, enable_swiglu=True, enable_rms_norm=True,
            enable_qk_norm=True, use_muon=True,
            no_biases=True, custom_init=True, residual_init=False,
            ckpt_final=True, **flags)
        for arm, arm_seeds, flags in arms
        for seed in arm_seeds
    ]


def p1_doc_mask_arms():
    """P1 doc mask / doc RoPE · 3-seed · adamw lr=2.4e-2 · muon_lr=0.00816 · WSD"""
    return doc_mask_arms("P1")


def p2_doc_mask_arms():
    """Shape × seed × feature · docmask / docrope · WSD · 20× (P2)"""
    return doc_mask_arms("P2")


def p3_doc_mask_arms():
    """Shape × seed × feature · docmask / docrope · WSD · 20× (P3)"""
    return doc_mask_arms("P3")


def p4_doc_mask_arms():
    """Shape × seed × feature · docmask / docrope · WSD · 20× (P4)"""
    return doc_mask_arms("P4")


# ------------------------------------------------------- 80x LR probes
def _probe_80x(model, adamw_lrs, muon_lrs):
    return f"{model}_80x_probe", [
        RunConfig(
            run=f"{model}_80x_lr{lr}_muon_mlr{mlr}", model=model, seed=47,
            schedule="wsd", lr=lr, muon_lr=mlr, train_token_multiplier=80.0,
            enable_init=False, enable_swiglu=True, enable_rms_norm=True,
            enable_qk_norm=True, use_muon=True,
            no_biases=True, custom_init=True, residual_init=False,
            ckpt_final=True)
        for lr in adamw_lrs
        for mlr in muon_lrs
    ]


def p2_80x_muon_probe():
    """Muon LR probe · 80× · WSD · seed 47 · adamw lr=4e-3"""
    return _probe_80x("P2", [4e-3], [0.00392, 0.00784])


def p2_80x_adamw_probe():
    """AdamW LR probe · 80× · WSD · seed 47 · muon_lr=0.005"""
    return _probe_80x("P2", [2.83e-3, 5.66e-3], [0.005])


def p3_80x_muon_probe():
    """Muon LR probe · 80× · WSD · seed 47 · adamw lr=3e-3"""
    return _probe_80x("P3", [3e-3], [0.00129, 0.00259, 0.00517])


# --------------------------------------------------------------- registry
RUN_SETS = {f.__name__: f for f in (
    initial_floor_sweep, initial_floor_sweep_wsd,
    p1_lr_sweep_cosine, p1_lr_sweep_wsd, p1_seed_noise_wsd,
    p1_muon_lr_sweep_pre_fix, p1_arch_sweep_pre_fix,
    p1_seed_noise_post_fix, p1_lr_sweep_post_fix,
    p1_init_factorial, p1_best_init_3_seeds, p1_init_ttt_double_check,
    p1_muon_lr_sweep_post_fix, p1_adamw_lr_sweep_muon_0_01,
    p1_muon_best_3_seeds, p1_arch_lr_sweep_post_fix,
    p1_no_swiglu_5_seeds, p1_no_rms_norm_5_seeds, p1_no_qk_norm_3_seeds,
    p1_final_muon_lr_fit, p1_final_5_seeds, p1_adam_only_lr_sweep,
    p1_cooldown_branching, p1_adam_refit_3_seeds,
    p2_lr_grid, p2_base_5_seeds, p2_residual_init_5_seeds,
    p2_adam_only_lr_sweep, p2_adam_only_5_seeds, p2_no_qk_norm_3_seeds,
    p2_cooldown_branching, p2_cooldown_branching_adamw_only,
    p2_batching_validation,
    p3_muon_lr_sweep, p3_adamw_lr_sweep, p3_base_5_seeds,
    p3_muon_lr_sweep_2, p3_cooldown_branching,
    p4_base_3_seeds, p4_cooldown_branching,
    p1_doc_mask_arms, p2_doc_mask_arms, p3_doc_mask_arms, p4_doc_mask_arms,
    p2_80x_muon_probe, p2_80x_adamw_probe, p3_80x_muon_probe,
)}
