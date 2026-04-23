```
sudo apt install cmake libssl-dev libcurl4-openssl-dev
```

serve llm for judge-based rewards with llama.cpp docker image
```
sudo docker run --gpus all -v ./served_models:/models -p 7654:7654 ghcr.io/ggml-org/llama.cpp:server-cuda -hf unsloth/gemma-4-E4B-it-GGUF --port 7654 --host 0.0.0.0
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

### test6
- model
    - qwen3-4b-base
- reward
    - core_ed:
        - pedagogical_structure: 1
        - lesson_engagement: 0.25
        - factual_accuracy: 1
    - fl_teacher:
        - systematic_phonics: 1
    - length_target
        - reward length that matches good example
- the idea of this one is to test a single dimension of FL by itself

### test7
- model
    - qwen3-4b-base
- reward
    - core_ed:
        - pedagogical_structure: 1
        - lesson_engagement: 0.25
        - factual_accuracy: 1
    - fl_teacher:
        - reading_fluency: 1
    - length_target
        - reward length that matches good example
- the idea of this one is to test a single dimension of FL by itself

### test8
- model
    - qwen3-4b-base
- reward
    - core_ed:
        - pedagogical_structure: 1
        - lesson_engagement: 0.25
        - factual_accuracy: 1
    - fl_teacher:
        - reading_comprehension: 1
    - length_target
        - reward length that matches good example
- the idea of this one is to test a single dimension of FL by itself

### test9
- model
    - qwen3-4b-base
- reward
    - core_ed:
        - pedagogical_structure: 1
        - lesson_engagement: 0.25
        - factual_accuracy: 1
    - fl_teacher:
        - writing_encoding: 1
    - length_target
        - reward length that matches good example
- the idea of this one is to test a single dimension of FL by itself

### test10
- model
    - qwen3-4b-base
- reward
    - core_ed:
        - pedagogical_structure: 1
        - lesson_engagement: 0.25
        - factual_accuracy: 1
    - fl_teacher:
        - oral_language_vocabulary: 1
    - length_target
        - reward length that matches good example
- the idea of this one is to test a single dimension of FL by itself

### reasoning/test1
- model
    - qwen3-4b-base
- reward
    
### instruction_following/test1
- model
    - qwen3-4b-base
- reward
    - core_ed:
        - pedagogical_structure: 1
        - lesson_engagement: 0.25
        - factual_accuracy: 1
    - fl_teacher:
        - writing_encoding: 1
    - correctness_reward
        - reward length that matches good example (3 points)
        - reward formatting that matches good example (3 points)
        - reawrd instruction following compared to good example (3 points)
    - NOTE: both core_ed and fl_teacher used score cap of 8 for any dimension
    - previous tests used score cap of 12

### instruction_following/test2
- same as test1 but with cap 12

### instruction_following/test3
- model
    - qwen3-4b-base
- reward
    - core_ed:
        - pedagogical_structure: 1
        - lesson_engagement: 0.25
        - factual_accuracy: 1
    - fl_teacher:
        - phonological_awareness: 1
    - correctness_reward
        - reward length that matches good example (3 points)
        - reward formatting that matches good example (3 points)
        - reawrd instruction following compared to good example (3 points)
    - qr score cap 12