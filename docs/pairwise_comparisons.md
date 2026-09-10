# Pairwise comparisons

- Based on the method presented in [QuRating: Selecting High-Quality Data for Training Language Models](https://arxiv.org/abs/2402.09739)
	- LLM-as-a-judge to select winners on defined criteria from pairs of text
	- The prompt contains a description of the criterion along with both texts and instructions to choose which one is better according to the criterion
	- Each pair is presented 40 times in total
		- 20 times with Text A appearing before Text B
		- 20 times with Text B appearing before Text A
	- From these 40 samples, empirical probability of preferring Text A to Text B is estimated, accounting for order preference bias with the balanced number of order configurations outlined above
- Initially developed 6 criteria focusing on "core" educational dimensions
- Additionally developed criteria for two sets of 7 dimensions
	- Foundational Literacy - Student-facing
	- Foundational Literacy - Teacher-Facing

### Additional optimisation

- QuRating method is based upon estimating probability of choosing A > B for each pair of documents A, B on each criterion​
- Original method estimated this probability by generating 40 LLM judgements (20 judgements per presentation order – i.e. A, B vs B, A) and calculating the empirical probability from these​
- We found this full process to be prohibitively expensive and time-consuming for iterating on larger samples of judgements​
- Instead of estimating probability of choosing A > B empirically from multiple observations, we can access this probability directly from a single LLM call by observing the relative probability of the tokens "A" and "B" in the response​
- This leads to a 20x reduction in the number of LLM calls required per document pair (so huge cost reductions).​
- Comparing pairwise probabilities between the two approaches on 500 random samples, we observed a Mean Absolute Error < 0.03