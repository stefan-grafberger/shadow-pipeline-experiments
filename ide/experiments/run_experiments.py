import pandas as pd
from langchain_community.vectorstores.chroma import Chroma

from ide.experiments import experiment1, experiment2
from ide.experiments.experiment1.fairness import pipeline_llm_fairness_opt, pipeline_ml_fairness_opt, \
    pipeline_llm_fairness_naive, pipeline_ml_fairness_naive
from ide.experiments.experiment1.mislabel import pipeline_llm_mislabel_opt, pipeline_ml_mislabel_opt, \
    pipeline_ml_mislabel_proxy_opt, pipeline_llm_mislabel_naive, pipeline_ml_mislabel_naive
from ide.experiments.experiment1.robustness import pipeline_llm_robustness_opt, pipeline_ml_robustness_opt, \
    pipeline_llm_robustness_naive, pipeline_ml_robustness_naive
from ide.experiments.experiment2.fairness import pipeline_llm_fairness_opt, pipeline_ml_fairness_opt, \
    pipeline_llm_fairness_naive, pipeline_ml_fairness_naive
from ide.experiments.experiment2.mislabel import pipeline_llm_mislabel_opt, pipeline_ml_mislabel_opt, \
    pipeline_ml_mislabel_proxy_opt, pipeline_llm_mislabel_naive, pipeline_ml_mislabel_naive
from ide.experiments.experiment2.robustness import pipeline_llm_robustness_opt, pipeline_ml_robustness_opt, \
    pipeline_llm_robustness_naive, pipeline_ml_robustness_naive
from ide.utils.utils import get_project_root


def run_experiment(experiment_name, scenario_name, variant_name, opt_name):
    if experiment_name == "exp1" and scenario_name == "fairness" and variant_name == "llm" and opt_name == "opt":
        score = experiment1.fairness.pipeline_llm_fairness_opt.execute()
    elif experiment_name == "exp1" and scenario_name == "fairness" and variant_name == "ml" and opt_name == "opt":
        score = experiment1.fairness.pipeline_ml_fairness_opt.execute()
    elif experiment_name == "exp1" and scenario_name == "mislabel" and variant_name == "llm" and opt_name == "opt":
        score = experiment1.mislabel.pipeline_llm_mislabel_opt.execute()
    elif experiment_name == "exp1" and scenario_name == "mislabel" and variant_name == "ml" and opt_name == "opt":
        score = experiment1.mislabel.pipeline_ml_mislabel_opt.execute()
    elif experiment_name == "exp1" and scenario_name == "mislabel" and variant_name == "ml" and opt_name == "proxy_opt":
        score = experiment1.mislabel.pipeline_ml_mislabel_proxy_opt.execute()
    elif experiment_name == "exp1" and scenario_name == "robustness" and variant_name == "llm" and opt_name == "opt":
        score = experiment1.robustness.pipeline_llm_robustness_opt.execute()
    elif experiment_name == "exp1" and scenario_name == "robustness" and variant_name == "ml" and opt_name == "opt":
        score = experiment1.robustness.pipeline_ml_robustness_opt.execute()
    elif experiment_name == "exp2" and scenario_name == "fairness" and variant_name == "llm" and opt_name == "opt":
        score = experiment2.fairness.pipeline_llm_fairness_opt.execute()
    elif experiment_name == "exp2" and scenario_name == "fairness" and variant_name == "ml" and opt_name == "opt":
        score = experiment2.fairness.pipeline_ml_fairness_opt.execute()
    elif experiment_name == "exp2" and scenario_name == "mislabel" and variant_name == "llm" and opt_name == "opt":
        score = experiment2.mislabel.pipeline_llm_mislabel_opt.execute()
    elif experiment_name == "exp2" and scenario_name == "mislabel" and variant_name == "ml" and opt_name == "opt":
        score = experiment2.mislabel.pipeline_ml_mislabel_opt.execute()
    elif experiment_name == "exp2" and scenario_name == "mislabel" and variant_name == "ml" and opt_name == "proxy_opt":
        score = experiment2.mislabel.pipeline_ml_mislabel_proxy_opt.execute()
    elif experiment_name == "exp2" and scenario_name == "robustness" and variant_name == "llm" and opt_name == "opt":
        score = experiment2.robustness.pipeline_llm_robustness_opt.execute()
    elif experiment_name == "exp2" and scenario_name == "robustness" and variant_name == "ml" and opt_name == "opt":
        score = experiment2.robustness.pipeline_ml_robustness_opt.execute()
    elif experiment_name == "exp1" and scenario_name == "fairness" and variant_name == "llm" and opt_name == "naive":
        score = experiment1.fairness.pipeline_llm_fairness_naive.execute()
    elif experiment_name == "exp1" and scenario_name == "fairness" and variant_name == "ml" and opt_name == "naive":
        score = experiment1.fairness.pipeline_ml_fairness_naive.execute()
    elif experiment_name == "exp1" and scenario_name == "mislabel" and variant_name == "llm" and opt_name == "naive":
        score = experiment1.mislabel.pipeline_llm_mislabel_naive.execute()
    elif experiment_name == "exp1" and scenario_name == "mislabel" and variant_name == "ml" and opt_name == "naive":
        score = experiment1.mislabel.pipeline_ml_mislabel_naive.execute()
    elif experiment_name == "exp1" and scenario_name == "robustness" and variant_name == "llm" and opt_name == "naive":
        score = experiment1.robustness.pipeline_llm_robustness_naive.execute()
    elif experiment_name == "exp1" and scenario_name == "robustness" and variant_name == "ml" and opt_name == "naive":
        score = experiment1.robustness.pipeline_ml_robustness_naive.execute()
    elif experiment_name == "exp2" and scenario_name == "fairness" and variant_name == "llm" and opt_name == "naive":
        score = experiment2.fairness.pipeline_llm_fairness_naive.execute()
    elif experiment_name == "exp2" and scenario_name == "fairness" and variant_name == "ml" and opt_name == "naive":
        score = experiment2.fairness.pipeline_ml_fairness_naive.execute()
    elif experiment_name == "exp2" and scenario_name == "mislabel" and variant_name == "llm" and opt_name == "naive":
        score = experiment2.mislabel.pipeline_llm_mislabel_naive.execute()
    elif experiment_name == "exp2" and scenario_name == "mislabel" and variant_name == "ml" and opt_name == "naive":
        score = experiment2.mislabel.pipeline_ml_mislabel_naive.execute()
    elif experiment_name == "exp2" and scenario_name == "robustness" and variant_name == "llm" and opt_name == "naive":
        score = experiment2.robustness.pipeline_llm_robustness_naive.execute()
    elif experiment_name == "exp2" and scenario_name == "robustness" and variant_name == "ml" and opt_name == "naive":
        score = experiment2.robustness.pipeline_ml_robustness_naive.execute()
    else:
        raise ValueError(f"Invalid experiment: {experiment_name} {scenario_name} {variant_name}!")
    return score


num_repetitions = 7
for experiment_name in ["exp1", "exp2"]:
    for variant_name in ["llm", "ml"]:
        for scenario_name in ["fairness"]: #, "robustness", "mislabel"]:
            opt_names = ["naive", "opt"]
            if scenario_name == "mislabel" and variant_name == "ml":
                opt_names.append("proxy_opt")
            for opt_name in opt_names:
                # Warmup
                _ = run_experiment(experiment_name, scenario_name, variant_name, opt_name)
                # This error really took me a while to find... Without this, old documents persist in the
                #  vectorstore because they share a collection name and Chroma is not fully cleaned up between
                #  runs...
                Chroma(collection_name="langchain").delete_collection()
                # TODO: Should we run opt and naive immediately after each other?
                scores = []
                for repetition in range(num_repetitions):
                    scores.append(run_experiment(experiment_name, scenario_name, variant_name, opt_name))
                    # This error really took me a while to find... Without this, old documents persist in the
                    #  vectorstore because they share a collection name and Chroma is not fully cleaned up between
                    #  runs...
                    Chroma(collection_name="langchain").delete_collection()
                result_df = pd.DataFrame.from_records(scores)
                result_df.to_csv(f'{str(get_project_root())}/results/{experiment_name}_{scenario_name}_'
                                 f'{variant_name}_{opt_name}.csv', index=False)
