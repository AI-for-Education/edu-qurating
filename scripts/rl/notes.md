```
sudo apt install cmake libssl-dev libcurl4-openssl-dev
```

### test1
- model
    - qwen3-4b-base
- reward
    - core_ed:
        - pedagogical_structure: 1
        - lesson_engagement: 0.25
        - factual_accuracy: 1
        - education_level_primary: 0.25
    - fl_teacher:
        - all: 1
- obervations
    - final results are a bit child-like, like aimed at children rather than aimed at teachers
    - remove primary level reward?

### test2
- model
    - qwen3-4b-base
- reward
    - core_ed:
        - pedagogical_structure: 1
        - lesson_engagement: 0.25
        - factual_accuracy: 1
    - fl_teacher:
        - all: 1
- obervations
    - final results are overly long and verbose, tending towards "step 1", "step 2", ... "step 5" type plans, even if question is not about lesson plan
    - add reward for matching target length (based on length of good examples)?


### test3
- model
    - qwen3-4b-base
- reward
    - core_ed:
        - pedagogical_structure: 1
        - lesson_engagement: 0.25
        - factual_accuracy: 1
    - fl_teacher:
        - all: 1
    - length_target
        - reward length that matches good example
- observations
    - adding length target didn't completely fix this issue, although it seems better and average length is less.
    - maybe we are running for too long?
    - step 1000 maybe felt better than step 2000 wrt to this issue?
    - or maybe pedagogical structure reward is pushing this too far into this direction?

### test4
- model
    - qwen3-4b-base
- reward
    - core_ed:
        - pedagogical_structure: 0.1
        - lesson_engagement: 0.25
        - factual_accuracy: 1
    - fl_teacher:
        - all: 1
    - length_target
        - reward length that matches good example

### test5
- model
    - qwen3-4b-base
- reward
    - core_ed:
        - pedagogical_structure: 1
        - lesson_engagement: 0.25
        - factual_accuracy: 1
    - fl_teacher:
        - phonological_awareness: 1
    - length_target
        - reward length that matches good example
- the idea of this one is to test a single dimension of FL by itself

### reasoning/test1
- model
    - qwen3-4b-base
- reward
    