# %%
from typing import Literal
from enum import Enum
from itertools import product
from hashlib import sha256

from datasets import Dataset
import jsonlines
from fdllm import get_caller, register_models, LLMMessage
from pydantic import BaseModel
import pandas as pd

from qurating.constants import TEMPLATES_DIR, ROOT, DATA_DIR

register_models(ROOT / "custom_models.yaml")

model = "gemini-3.1-pro-preview"

# %%
caller = get_caller(model)

out = caller.call(LLMMessage(Role="user", Message="this is a test"))

print(out.Message)

# %%
task_file = DATA_DIR / "education_evals" / "tasks_and_subtasks_master.csv"
task_df = pd.read_csv(task_file)

all_tasks = list(task_df["Task Name"].unique())

TasksType = Literal[tuple(all_tasks)]

# %%
templates_dir = TEMPLATES_DIR / "FLN_teacher-facing"

templates_files = sorted(templates_dir.glob("*.txt"))

# %%
fl = templates_files[2]

with open(fl) as f:
    template_text = f.read()


# %%
message_template = """
Thanks for agreeing to help with this crucial assignment. Allow me to explain what we
need. We are designing a test for trainee teachers in foundational literacy. Right now
we have a definition of the criterion that we want to test the teachers on, and we have
a procedure for testing them - which is a set of pairwise comparisons between outputs
from randomly selected pairs of teachers.

What we are currently missing is a series of tasks to test the teachers on. This is
where you come in. The tasks should be designed in such a way that it is possible to
judge the output according to the criterion of interest. This means that the task
itself should not contain any hints at what is expected in a "good" answer. It should
be possible to provide a "bad" answer (missing all of the key features of the
criterion) as well as a "good" one (covering all of the key features of the criterion).

The criterion is as follows:
<criterion>
{template_text}
</criterion>

What we need is {ntasks} tasks, targetting grade {grade} level. The tasks should all fall
under the family of {task_family}. In addition to the task, we also need an example of a 
"good" answer. It's important to remember that the answers should be treated as a complete
and fully fleshed out output that a teacher would be expected to use with their students.
It should not be a simple sketch of an output.
"""


# %%
class TaskFormat(BaseModel):
    grade: Literal["0", "1", "2", "3"]
    family: TasksType  # type: ignore
    category: str
    content: str
    features_good_answer_should_contain: list[str]
    features_good_answer_should_avoid: list[str]
    example_of_good_answer: str


class ResponseFormat(BaseModel):
    tasks: list[TaskFormat]


ntasks = 10
nreps = 10
respobj_list = []

respobj_cache = DATA_DIR / "education_evals" / "flteach_grpo_dataset.jsonl"
if respobj_cache.exists():
    with jsonlines.open(respobj_cache) as f:
        for obj in f:
            respobj_list.append(obj)

for template_fl, task_family, grade in product(
    templates_files, all_tasks, [0, 1, 2, 3]
):
    for rep in range(nreps):
        item_hash = sha256(
            f"{template_fl.stem}{task_family}{grade}".encode("utf-8")
        ).hexdigest()
        cache_hit = False
        for obj in respobj_list:
            if obj["hash"] == item_hash and obj["rep"] == rep:
                cache_hit = True
                break
        if cache_hit:
            continue

        print(f"Criterion: {template_fl.stem}; Rep: {rep}")
        print(f"Task family: {task_family}; Grade: {grade}")

        complete = False
        while not complete:
            with open(template_fl) as f:
                template_text = f.read()

            message_text = message_template.format(
                template_text=template_text,
                ntasks=ntasks,
                grade=grade,
                task_family=all_tasks[0],
            )
            try:
                out = caller.call(
                    LLMMessage(Role="user", Message=message_text),
                    max_tokens=None,
                    response_schema=ResponseFormat,
                )

                respobj = ResponseFormat.model_validate_json(out.Message)

                assert len(respobj.tasks) == ntasks

                complete = True

            except:
                pass

        respdict = {
            "hash": item_hash,
            "rep": rep,
            "criterion": template_fl.stem,
            **respobj.model_dump(),
        }

        respobj_list.append(respdict)

        with jsonlines.open(respobj_cache, "a") as f:
            f.write(respdict)

# %%
resp_df = pd.DataFrame(
    sum(
        (
            [
                {
                    **{
                        key: val for key, val in respobj.items() if key not in ["tasks"]
                    },
                    **task_row,
                }
                for task_row in respobj["tasks"]
            ]
            for respobj in respobj_list
        ),
        [],
    )
)

resp_ds = Dataset.from_pandas(resp_df)
resp_ds.to_parquet(DATA_DIR / "education_evals" / "flteach_grpo_dataset.parquet")
