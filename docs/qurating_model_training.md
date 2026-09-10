# Qurating Model Training

- The type of model that is suitable for this task is a Sequence Classification model​
- This is an encoder-only LLM, that has all of the architecture up to the embeddings layer, but lacks the decoder architecture that would be required to actually generate text output​
- A linear classification head is then added on top with the number of output dimensions matching the number of criteria to be rated​
- The whole model (including the classification head but also including the weights before that) is trained end-to-end on to predict the pairwise probabilities generated in the previous step​
- The original QuRating paper used Sheared-LLaMa-1.3b modified for sequence classification​
- We tested this original model, along with 4 other models adapted for sequence classification:​
	- Qwen-3-Reranker-0.6B​
	- Qwen-3-Reranker-4B​
	- Qwen-3-Reranker-8B​
	- Gemma-3-4B