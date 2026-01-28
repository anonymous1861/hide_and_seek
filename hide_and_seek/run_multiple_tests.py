import subprocess

def build_cmd(base_script, params):
    cmd = ["python", base_script]
    for key, value in params.items():
        flag = f"--{key}"
        if key == "batchnorm-hs":
            if value is True:
                cmd.append(flag)  # including flag sets it to True because action="store_true".
        elif key == "return_losses_on_val":
            if value is True:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


# already_run = [8,10,11,12,14,16,19,20]
seeds_list =  [12,13,14,16,17,18,19,20] #[2,3,4,5,6,7,8,9,10,11]#,#[0,1,2,3,4,5,6,7,8,9] #on seed 15 Hide&Seek Syn4 11 features collapsed. This is a 1/120 occurence (20 seeds, 6 datasets). Can be fixed by batching.
# seeds_list = [0,1,2,3,4,5,6,7,9,13,17,18]
# lmbdas = [1.8]#[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1] #0.6 if normalising (no longer doing this)
# seeds = seeds_list[0:1] #seeds_list[6:12]#[10:20] #[0:10]#

# lmbda_s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lmbda_s = [0.1] #[0.01,0.05, 0.1, 0.2, 0.3, 0.4]# [0.01] # [0.05] # 
# batchnorm_hs_s = [True, False]
# hidden_dim_s = [16, 32, 64, 100]
# hidden_layers_s = [2,3,4]
# epochs_s = [500,1000]
# num_syn_features_s = [11, 100]
# lmbda_exponent_s = [0,1,2,3]
lmbda_exponent_s = [2]

if __name__ == "__main__":
    for lmbda in lmbda_s:
        # for batchnorm_hs in batchnorm_hs_s:
            # for hidden_dim in hidden_dim_s:
                # for hidden_layers in hidden_layers_s:
                    # for epochs in epochs_s:
                        # for num_syn_features in num_syn_features_s:
        for seed in seeds_list:
            params = {
                "lmbda": lmbda,
                "seed": seed,
                "syn-idx": 3, #can be over-ridden in other file
                "batchnorm-hs": False,   # handled specially
                "return_losses_on_val": False,
                "epochs": 10_000, #10_000
                "batch-size": 1_000, #1_000
                "num-syn-features": 11,
                "train-N": 10_000,
                "test-N": 10_000,
                "hide-hidden-dim": 32,
                "seek-hidden-dim": 32,
                "hide-num-hidden-layers": 2,
                "seek-num-hidden-layers": 2,
                "lmbda-exponent": 2
            }

            cmd = build_cmd(
                "/home/tyellins/projects/causal_flow/event_causality/AISTATS_2026/hide_and_seek/run_tests.py",
                # "/home/tyellins/projects/causal_flow/event_causality/AISTATS_2026/tests_credit_default/credit_default_tools.py",
                params,
            )

            print("RUNNING:", " ".join(cmd))
            subprocess.run(cmd, check=True)