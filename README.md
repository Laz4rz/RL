# Reinforcement Learning

Shenanigans while following Naklecha (https://naklecha.notion.site/a-reinforcement-learning-guide). Parts of this texts are rewritten/copied from the original blog, and some will be added by me to elevate my understanding.

### Value function

$V(s) = \textrm{max}(V(s'))$

where $s$ — current state ($\in S$), $s'$ — next state after taking some action $a\in A$.

##### Discount ($\gamma$)

To differentiate between close and distant wins, we introduce the discount factor $\gamma$.

$V(s) = \textrm{max}(\gamma * V(s'))$

Now in eg. chess ($\gamma=0.9$): 

```
V(checkmate position) = 100
V(checkmate in 1 position) = 0.9 * 100 = 90
V(checkmate in 2 position) = 0.9 * 90 = 81
V(checkmate in 3 position) = 0.9 * 81 = 72.9
```

Instead of all positions being valued the same. 

Discount factor is especially important in environments that are non-determninistic (have hidden information), like DOTA, as you'd value winning the game as fast as possible, minimizng the amount of unknowns that could negatively impact your current win condtions.   

##### Reward ($R(s,a)$)

If we only based our value function on the result of finishing the game, then our value would only "flow backwards", while in reality certain positions may be more valuable then others even if they lead to winning in the same number of steps (this is non-intuitive, at least in chess, and I'm unsure whether I fully see it — I'd think about computational efficiency here). 

We therefor modify the Value function to assign value based on both intermediate (current state) reward, and discounted end-game value. This modified Value function is called: **The Bellman Equation**.

$V(s) = \underset{a \in A}{\max}(\textrm{R}(s,a) + \gamma * V(s'))$

The rewards for some well studied problems, are... well-known. Eg. chess again: 

```python
chess_reward_table = {
    # Material values (traditional piece values)
    'piece_values': {
        'pawn': 1.0,
        'knight': 3.0,
        'bishop': 3.0,
        'rook': 5.0,
        'queen': 9.0
    },
    
    # Positional rewards
    'position_rewards': {
        'control_center': 0.2,     # Bonus for controlling center squares
        'connected_rooks': 0.3,    # Bonus for connected rooks
        'bishop_pair': 0.3,        # Small bonus for having both bishops
        'doubled_pawns': -0.2,     # Penalty for doubled pawns
        'isolated_pawns': -0.2,    # Penalty for isolated pawns
    },
    
    # Development rewards (early game)
    'development': {
        'piece_developed': 0.1,    # Piece moved from starting square
        'king_castled': 0.5,       # Successfully castled king
        'controlling_open_file': 0.2  # Rook on open file
    },
    
    # King safety
    'king_safety': {
        'pawn_shield': 0.3,        # Pawns protecting king
        'king_exposed': -0.4,      # Penalty for exposed king
    },
    
    # Game ending states
    'game_end': {
        'checkmate_win': 100.0,    # Winning the game
        'checkmate_loss': -100.0,  # Losing the game
        'stalemate': 0.0,          # Draw
    }
}
```

With the example reward being calculated as:

```
R(s, a) = +100 for checkmates, +9 for capturing the enemy's queen, and +9.3 (9+0.3) for capturing a queen and connecting 2 rooks together
```

So called action-value function is often used as an intermediate step for calculating the Value function.

$Q(s,a) = R(s,a) + \gamma * V(s')$

##### Exploration vs Exploitation

We are still facing the problem of having to go through all the combinations of different moves, to calculate the values assigned to each state and action. This usually gets so mindblowingly computationally costly, that we have to do with some partial tree traversal. 

**Exploration:** When learning in an uncertain environment, an agent needs to try new, potentially suboptimal actions to discover better strategies it might have missed.

**Exploitation:** Once an agent has discovered reliable strategies, it can exploit this knowledge by repeatedly choosing actions that lead to rewards.

One of the algorithms utilizing Exploration vs Exploitation through an epsilon-greedy strategy is real-time dynamic programming.

```python
def rtdp(state_space, actions, initial_state):
   V = {s: 0 for s in state_space}
   epsilon = epsilon_start
   
   gamma=0.99 # discount factor from bellman equation
   epsilon=0.9 # exploration rate, ε as described above
   max_episodes=1000 # number of iterations
   epsilon_decay = 0.995 # epsilon decay amount after each episode

   for episode in range(max_episodes):
       state = initial_state
       while not terminal(state):
           # next action decided using epsilon-greedy method
           best_action = argmax(actions, lambda a: V[get_next_state(state, a)])
           next_action = random_action() if random() < epsilon else best_action
           next_state = transition(state, next_action)
           # Bellman update 
           max_future_value = max(V[get_next_state(state, a)] for a in actions)
           V[state] = reward(state,next_action) + gamma * max_future_value
           state = next_state
       epsilon *= epsilon_decay
   return V
```

An example of solving a very simple maze (it's not really a maze, just learning the shortest path on bounded rectangle) is implemented in `rtdp.py`.





