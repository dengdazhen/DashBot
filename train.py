import argparse
import os

# Prevent numpy from using multiple threads
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np  

import asyncTrainingModule
from pfrl import experiments, utils  
from agentA3C import A3C  
from pfrl.optimizers import SharedRMSpropEpsInsideSqrt  

from chartEnvMonitor import Monitor
from chartEnvGym import chartEnvGym
from model import dashboardA3C
from vega_datasets import data as vega_data
from constrainedSampling import consSampling


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--t-max", type=int, default=100)
    parser.add_argument("--beta", type=float, default=1e-2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--steps", type=int, default=1 * 10 ** 6)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--eval-interval", type=int, default=2500)
    parser.add_argument("--eval-n-steps", type=int, default=1250)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.processes) + args.seed * args.processes
    assert process_seeds.max() < 2 ** 31

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    opts = ['add', 'undo', "change", "terminate"]
    marks = ['bar',	'line',	'point', 'boxplot']
    aggs = [None, 'bin', 'mean', 'count', 'top', 'bottom']
    encodings = ['x', 'y', 'color']
    n_opt = len(opts)
    n_mark = len(marks)
    n_enc = len(encodings)
    max_field_num = 10
    n_agg = len(aggs)
    max_chart = 10

    dashboard_data = [vega_data.cars(), vega_data.movies(), vega_data.seattle_weather(), vega_data.us_employment()]
    dashboard_data = [
		df.iloc[:, 0:max_field_num] if len(df.columns) > max_field_num else df for df in dashboard_data]

    def make_env(process_idx, test):
        # Use different random seeds for train and test envs
        process_seed = process_seeds[process_idx]
        env_seed = 2 ** 31 - 1 - process_seed if test else process_seed
        env = chartEnvGym(
            dashboard_data, 
            marks=marks, 
            max_chart=max_chart, 
            max_field_num = max_field_num,
            opts = opts,  
            aggregates=aggs,
            encodings=encodings,
            max_step=args.t_max)
        # env.update_fields([{"value":"Release Date","type":"temporal"}])
        env.seed(int(env_seed))
        if args.monitor:
            env = Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        return env

    sample_env = make_env(0, False)
    constraint = consSampling(
        dashboard_data, 
        max_field_num,
        sample_env.vectorization.topic_feat_index, 
        field_feat_index = sample_env.vectorization.field_feat_index, 
        field_feat_len = sample_env.vectorization.field_feat_len,
        opts=opts,
        marks = marks,
        encodings = encodings,
        aggregates=aggs
        )

    model = dashboardA3C(sample_env.vectorization.mv_feat_len, action_num = n_opt, mark_num = n_mark, \
	    enc_num = n_enc, field_num = max_field_num, agg_num = n_agg)

    # SharedRMSprop is same as torch.optim.RMSprop except that it initializes
    # its state in __init__, allowing it to be moved to shared memory.
    opt = SharedRMSpropEpsInsideSqrt(model.parameters(), lr=7e-4, eps=1e-1, alpha=0.99)
    assert opt.state_dict()["state"], (
        "To share optimizer state across processes, the state must be"
        " initialized before training."
    )


    agent = A3C(
        model,
        constraint,
        opt,
        t_max=args.t_max,
        gamma=1.0,
        beta=args.beta,
        pi_loss_coef=1e-1,
        # phi=phi,
		keep_loss_scale_same=True,
		normalize_grad_by_t_max=False,
        max_grad_norm=40,
    )

    if args.load:
        agent.load(args.load)

    if args.demo:
        env = make_env(0, True)
        eval_stats = experiments.eval_performance(
            env=env, agent=agent, n_steps=args.eval_n_steps, n_episodes=None
        )
        print(
            "n_steps: {} mean: {} median: {} stdev: {}".format(
                args.eval_n_steps,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            for pg in agent.optimizer.param_groups:
                assert "lr" in pg
                # print("learning rate update: {}".format(value))
                pg["lr"] = value
        # lr = (args.lr - 0)/args.steps*(args.steps - args.step_offset)
        lr_decay_hook = experiments.LinearInterpolationHook(
            args.steps, args.lr, 0, lr_setter
        )

        asyncTrainingModule.train_agent_async(
            agent=agent,
            outdir=args.outdir,
            processes=args.processes,
            make_env=make_env,
            profile=args.profile,
            steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            global_step_hooks=[lr_decay_hook],
            save_best_so_far_agent=True,
        )


if __name__ == "__main__":
    main()