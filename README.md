# Downlink Power Control for Cell-Free Massive MIMO with Deep Reinforcement Learning

This is a code package is related to the following scientific article:

Lirui Luo, Jiayi Zhang, Shuaifei Chen, Bo Ai, and Derrick Wing Kwan Ng "Downlink Power Control for Cell-Free Massive MIMO with Deep Reinforcement Learning" *IEEE Transactions on Vehicular Technology*.

The package contains a simulation environment, based on Matlab, that reproduces some of the numerical results and figures in the article. *We encourage you to also perform reproducible research!*


## Abstract of Article

Recently, model-free power control approaches have been developed to achieve the near-optimal performance of cell-free (CF) massive multiple-input multiple-output (MIMO) with affordable computational complexity. In particular, deep reinforcement learning (DRL) is one of such promising techniques for realizing effective power control. In this paper, we propose a model-free method adopting the deep deterministic policy gradient algorithm (DDPG) with feedforward neural networks (NNs) to solve the downlink max-min power control problem in CF massive MIMO systems. Our result shows that compared with the conventional convex optimization algorithm, the proposed DDPG method can effectively strike a performance-complexity trade-off obtaining 1,000 times faster implementation speed and approximately the same achievable user rate as the optimal solution produced by conventional numerical convex optimization solvers, thereby offering effective power control implementations for large-scale systems. Finally, we extend the DDPG algorithm to both the max-sum and the max-product power control problems, while achieving better performance than that achieved by the conventional deep learning algorithm.

## Content of Code Package

The package generates the simulation SE results which are used in Figure 2, Figure 3 and Figure 4. To be specific:

- `main`: Main function;
- `data_generateformax`: Generate the training data for the DDPG algorithm;
  - `actor`: Actor network;
- `buffer`: Replay buffer;
- `critic`: Critic network;
- `ddpg`: The main body of the algorithm; 
- `environment`: The interactive environment of the algorithm;
- `utils`: Optimization tools.
- `transform_data_maxmin`: Transform the data for NN.
- `NN_MR_maxmin`: Networks to converge the DDPG data.
- `figureSlides`: The visualization of the transformation data.

See each file for further documentation.


## License and Referencing

This code package is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.
