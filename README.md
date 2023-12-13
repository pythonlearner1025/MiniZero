# AlphaZero for 5x5 Go - An MLX Implementation

AlphaZero for 5x5 Go using the MLX framework. The aim is to train a 5x5 Go agent on a M1 Macbook Pro that can consistently beat me. Although 5x5 Go is significantly scaled down from 19x19 Go, it still has a game tree complexity of $9.3 \times 10^{20}$ with an average game length of 15 moves, making it infeasible to solve (at least for I, who does not have a supercomputer). Therefore, a more intelligent approach is necessary to reach superhuman performance.

The big idea in AlphaZero is that superhuman performance in perfect-information games can be achieved without human knowledge. In practice, AlphaZero achieves this by learning stronger MCTS policies from game data generated through many iterations of self-play. The actual "learning" happens in a deep neural network $f$ that guides the MCTS. 

In MCTS simulations, $f$ is used to evaluate the action probabilities and the value estimate of a novel game state. The MCTS selects moves based on action probabilities and value estimates made by $f$. The policy found from MCTS + $f$ results in much stronger actions than actions predicted by $f$ alone. Therefore, after self-play we update $f$'s parameters to match the policy of MCTS + $f$ and the true value of the game state. These new parameters are used in the next iteration of self-play to make MCTS even stronger. 

# Search During Self-Play

During self-play, at each time step $t$ AlphaZero runs MCTS simulations over game state $s_t$ before sampling action $a_t$ to play from the MCTS policy $\pi$. A single MCTS game tree is built out for each iteration of self-play. 

At each iteration of the simulation: 
- If state $s$ is not in the tree, we evaluate and add $s$ as a node in the tree. we evaluate the state using $f$ to return $P(s, a), v = f(s)$, where $P(s, a)$ is the probability distribution over all possible actions and $v$ is the estimated value of $s$ ($+v$ or $-v$ depending on perspective of player). In our new node $s$ we,  
  - set $P(s, a)$
  - initialize mean action values $Q(s, a) = 0$ 
  - initialize action counts $N(s, a) = 0$. 
- Then, we backprop the value $v$ up the trajectory of edges $(s_0, a_0), (s_1, a_1), \ldots$ that lead to the leaf node $s$. Along this trajectory, we 
  - update the mean action value of each edge $Q(s_x, a_x) = \frac{N(s_x, a_x) \times Q(s_x, a_x) + v}{N(s_x, a_x) + 1}$ 
  - increment the visit count of the action $N(s_x, a_x)$ by 1.     
- We then select the best move $a$ to play from the current state $s$ using the formula $a = \text{argmax} \left[ Q(s, a) + \text{cpuct} \times P(s, a) \times \frac{\sqrt{N(s, b)}}{1 + N(s, a)} \right]$
- We then transition to the next state $s'$ from $s$ by taking action $a$. We recursively run search over this state $s'$ until the game terminates.

We can interpret the simulation as building a game tree using this search control strategy: 
- $Q(s, a) + U(s, a)$, where $U(s, a) = \text{cpuct} \times P(s, a) \times \frac{\sqrt{N(s, b)}}{1 + N(s, a)}$
- $\text{cpuct}$ can be seen as a constant determining the level of exploration.
- Initially, the search control strategy prefers actions with high probability and low visit count, but asymptotically prefers actions with high action value

After rounds of simulation, we sample action $a$ for state $s$ using the MCTS policy, which is the probability distribution over all legal actions. The policy is given by:
- $\pi = \frac{N(s, a)^{\frac{1}{t}}}{\sum_b N(s, b)^{\frac{1}{t}}}$

In essence, the policy's likelihood of choosing an action is proportional to its visit count relative to all other actions' visit counts.

# Training After Self-Play

Throughout self-play we collect experience data $(s_t, \pi_t, v_t)$ corresponding to game state, MCTS policy at that state, and true value of that state (assigned as either $+1$ or $-1$ after the game has terminated & been scored) at time step $t$ (equal to move number in Go). 

We then sample training data uniformly from this experience buffer for training. The neural network is updated to match its predictions $P, v = f(s_t)$ with $\pi_t$ and $v_t$ from the experience buffer. This can be interpreted as projecting the MCTS policy into the neural network function space, such that it will learn to mimic the stronger policy learned through self-play.

# Installation 

```bash
git clone https://github.com/pythonlearner1025/AlphaZero.git
cd AlphaZero
pip install -r requirements.txt
python main.py
```

Saving the weights and playing against a human player is yet unimplemented. The ```main.py``` file only trains the AlphaZero agent on a 5x5 board for now.

