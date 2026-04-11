# %%
import numpy as np
from dotenv import load_dotenv

from qr_reward import QuratingReward

load_dotenv(override=True)

# %%
qr_reward = QuratingReward()

test_ds = qr_reward._load_dataset(str(qr_reward.dataset_files["core_ed"]))

test_ds_2 = test_ds.map(
    lambda row: {"text": [r[0] for r in row["texts"]]},
    batched=True,
    remove_columns=["texts"],
)

# %%
score_spec_core_primary = {
    "core_ed": {
        "pedagogical_structure": 1,
        "lesson_engagement": 1,
        "factual_accuracy": 1,
        "education_level_primary": 1,
    }
}

score_spec_fl_primary_teacher = {
    **score_spec_core_primary,
    **{
        "fl_teacher": {
            label: 1 for label in qr_reward.labels["fl_teacher"]
        }
    }
}

reward_fun_core_primary = qr_reward.reward_fun_generator(score_spec_core_primary)
reward_fun_fl_primary_teacher = qr_reward.reward_fun_generator(score_spec_fl_primary_teacher)

rng = np.random.default_rng()
index = rng.permutation(len(test_ds_2))[:10].tolist()
completions = [[{"content": row}] for row in test_ds_2[index]["text"]]

reward_core_primary = reward_fun_core_primary(completions)
reward_fl_primary_teacher = reward_fun_fl_primary_teacher(completions)

for i, completion in enumerate(completions):
    print(completion[0]["content"])
    print(reward_core_primary[i])
    print(reward_fl_primary_teacher[i])
    print(f"\n{'#'*50}\n")
