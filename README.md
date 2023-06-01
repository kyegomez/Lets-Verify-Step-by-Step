# "Let’s Verify Step by Step"
Implementation of "Improving Mathematical Reasoning with Process Supervision" by OPENAI 


Let's Verify Step by Step: Reinforcement Training Models and Algorithmic Pseudocode
1. Introduction
In this research analysis, we aim to extend the architectural analysis of reinforcement training models by comparing outcome and process supervision. We will also provide a step-by-step algorithmic pseudocode for a to-do list.

2. Methods
2.1 Large-Scale and Small-Scale Regimes
We conduct experiments in two separate regimes: large-scale and small-scale. Each has its own advantages, and they offer complimentary perspectives.

2.1.1 Large-Scale Regime
At large-scale, we finetune all models from GPT-4 (OpenAI, 2023). We focus on advancing the state-of-the-art by training the most reliable ORM and PRM possible.

2.1.2 Small-Scale Regime
In the small-scale regime, we train models where we can conduct a more direct comparison. We use a large-scale model to supervise small-scale model training, enabling us to conduct several important ablations.

2.2 Scope
We use a single fixed model, the generator, to generate all solutions at each model scale. We focus exclusively on training the most reliable reward model possible and evaluate it by its ability to perform best-of-N search over uniformly sampled solutions from the generator.

2.3 Base Models
All large-scale models are finetuned from the base GPT-4 model (OpenAI, 2023). The small-scale base models are similar in design to GPT-4 but pretrained with roughly 200 times less compute. We finetune all models on a dataset of roughly 1.5B math-relevant tokens, called MathMix.

3. Algorithmic Pseudocode: Step-by-Step To-Do List
1. Initialize large-scale and small-scale regimes
2. For each regime:
   a. Finetune models from GPT-4 (OpenAI, 2023)
   b. Train the most reliable ORM and PRM possible
3. Use a large-scale model to supervise small-scale model training
4. Conduct ablations in the small-scale regime
5. For each model scale:
   a. Use a single fixed model (the generator) to generate all solutions
   b. Train the most reliable reward model possible
   c. Evaluate the reward model by performing best-of-N search over uniformly sampled solutions from the generator
6. Finetune all models on the MathMix dataset

2.3 Generator
We train the generator to produce solutions in a newline-delimited step-by-step format by few-shot generating solutions to MATH training problems, filtering those that reach the correct final answer, and finetuning the base model on this dataset for a single epoch.

2.4 Data Collection
2.4.1 Process Supervision Data
We collect process supervision data by presenting human data-labelers with step-by-step solutions to MATH problems sampled by the large-scale generator. Their task is to assign each step a label of positive, negative, or neutral.

2.4.2 PRM800K Dataset
The PRM800K training set contains 800K step-level labels across 75K solutions to 12K problems. We include data from 4.5K MATH test problems in the PRM800K training set and evaluate our models on the remaining 500 MATH test problems.

2.4.3 Solution Selection Strategy
We strategically select which solutions to show data-labelers by surfacing convincing wrong-answer solutions. We iteratively re-train our PRM using the latest data at several points in the data collection process, generating N solutions per problem and surfacing only the top K most convincing wrong-answer solutions to data-labelers.

3. Algorithmic Pseudocode: Step-by-Step To-Do List (Continued)
7. Train the generator to produce solutions in a newline-delimited step-by-step format
8. Collect process supervision data by presenting human data-labelers with step-by-step solutions
9. Create the PRM800K dataset with 800K step-level labels across 75K solutions to 12K problems
10. Strategically select which solutions to show data-labelers by surfacing convincing wrong-answer solutions
11. Iteratively re-train the PRM using the latest data at several points in the data collection process
12. Generate N solutions per problem and surface only the top K most convincing wrong-answer solutions to data-labelers
4. Conclusion
By following this extended step-by-step algorithmic pseudocode, we can effectively compare outcome and process supervision in reinforcement training models. This approach allows us to train reliable reward models, evaluate their performance in selecting the correct solution, and collect valuable process supervision data from human data-labelers.


2.6 Process-Supervised Reward Models (PRMs)
We train PRMs to predict the correctness of each step after the last token in each step. The PRM can be trained in a standard language model pipeline without any special accommodations. To determine the step-level predictions at test time, it suffices to perform a single PRM forward pass over the whole solution.

2.6.1 PRM Scoring
We define the PRM score for a solution to be the probability that every step is correct under the PRM. We implement this as the product of the correctness probabilities for each step.

2.6.2 Process Supervision
When providing process supervision, we deliberately choose to supervise only up to the first incorrect step. This makes the comparison between outcome and process supervision more straightforward and keeps the labeling cost similar for humans.

3. Algorithmic Pseudocode: Step-by-Step To-Do List (Continued)
13. Train PRMs to predict the correctness of each step after the last token in each step
14. Perform a single PRM forward pass over the whole solution to determine step-level predictions at test time
15. Define the PRM score for a solution as the probability that every step is correct under the PRM
16. Implement the PRM score as the product of the correctness probabilities for each step
17. Provide process supervision only up to the first incorrect step
4. Conclusion
By following this extended step-by-step algorithmic pseudocode, we can effectively compare outcome and process supervision in reinforcement training models. This approach allows us to train reliable reward models, evaluate their performance in selecting the correct solution, collect valuable process supervision data from human data-labelers, and train PRMs to predict the correctness of each step in a solution.

2.7 Large-Scale Supervision
We train the large-scale PRM using the step-level labels in PRM800K and the large-scale ORM on 100 uniform samples per problem from the generator. Figure 3 shows how the best-of-N performance of each reward model varies as a function of N. The PRM strongly outperforms both the ORM and majority voting.

2.7.1 ORM Training
Training the ORM solely on PRM800K solutions would be problematic due to the active learning strategy biasing the dataset towards wrong-answer solutions. We explored training the ORM on a superset of PRM800K solutions but found that this did not improve ORM performance.

2.8 Small-Scale Synthetic Supervision
We find that the PRM outperforms the ORM at large-scale, but this result alone paints an incomplete picture. To better compare outcome and process supervision, we need to isolate two confounding factors: the differences in training sets and the impact of final-answer grading on ORM performance.

3. Algorithmic Pseudocode: Step-by-Step To-Do List (Continued)
18. Train the large-scale PRM using the step-level labels in PRM800K
19. Train the large-scale ORM on 100 uniform samples per problem from the generator
20. Compare the best-of-N performance of each reward model as a function of N
21. Investigate the impact of training the ORM on a superset of PRM800K solutions
22. Isolate confounding factors to better compare outcome and process supervision

2.9 Small-Scale Synthetic Supervision (Continued)
2.9.1 Process vs Outcome Supervision
We conduct a direct comparison of outcome and process supervision by training reward models on identical datasets with different forms of supervision. Process supervision significantly outperforms both forms of outcome supervision at all data collection scales.

2.9.2 Active Learning
We investigate the impact of active learning by training a small-scale reward model, PRMselector, on a single sample from each problem. We use this model to score 1000 samples per problem and train larger reward models on selected samples. This form of active learning is approximately 2.6x more data efficient than uniform data labeling.

3. Algorithmic Pseudocode: Step-by-Step To-Do List (Continued)
23. Conduct a direct comparison of outcome and process supervision by training reward models on identical datasets with different forms of supervision
24. Investigate the impact of active learning by training a small-scale reward model, PRMselector, on a single sample from each problem
25. Use PRMselector to score 1000 samples per problem and train larger reward models on selected samples
26. Estimate the data efficiency of active learning compared to uniform data labeling

2.10 PRM Details
2.10.1 Training
We train our PRMs by fine-tuning the MathMix model to predict the probability of positive, negative, and neutral labels given a solution prefix ending in one of our labeled steps. We sweep over hyperparameters using a dataset containing the first ∼10% of PRM800K.

2.10.2 Scoring
We produce a single solution-level score by performing a reduction over step-level scores, where the step-level score is the probability that the step's label is positive. The best performing strategy is to take the product of step-level scores and to consider the neutrals as positives.

2.11 Difficulty Breakdown
We show the performance of our ORM and PRM on each quintile of the MATH dataset. The performance gap is apparent across all difficulties. Increasing the number of samples has the largest positive effect on the highest difficulty problems.

3. Algorithmic Pseudocode: Step-by-Step To-Do List (Continued)
27. Train PRMs by fine-tuning the MathMix model to predict the probability of positive, negative, and neutral labels given a solution prefix ending in one of the labeled steps
28. Produce a single solution-level score by performing a reduction over step-level scores, considering neutrals as positives and taking the product of step-level scores
29. Analyze the performance of ORM and PRM on each quintile of the MATH dataset
30. Investigate the impact of increasing the number of samples on the performance of different difficulty problems
