import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

'''
This script is an implementation of the Real-Time Dynamic Programming (RTDP) algorithm.
The problem solved is probably the simplest problem available, but shouldn't be too hard
to adapt to more complex problems.

The problem is a NxM maze, where the agent starts at some (x, y) position and has to reach
a terminal state at some (x', y') position. The agent can move up, down, left, or right 
freely, only being blocked by the walls of the maze. The agent has no knowledge of the maze.

The heatmaps show both the value function V(s) and the (one of) optimal path to the terminal 
state.
'''

def available_actions(state, state_space):
    actions = []
    if state[0] > 0:
        actions.append('up')
    if state[0] < state_space[0]-1:
        actions.append('down')
    if state[1] > 0:
        actions.append('left')
    if state[1] < state_space[1]-1:
        actions.append('right')
    return actions

# move to new location (state) function
def transition(state, action):
    if action == 'up':
        return (state[0]-1, state[1])
    elif action == 'down':
        return (state[0]+1, state[1])
    elif action == 'left':
        return (state[0], state[1]-1)
    elif action == 'right':
        return (state[0], state[1]+1)
    
def get_next_state(state, action):
    return transition(state, action)

    
def terminal(state, terminal_state):
    if state == terminal_state:
        return True
    else:
        return False

def rtdp(state_space, actions, initial_state, terminal_state):
    np.random.seed(0)

    V = np.zeros(state_space) # maze is 10x10
    V[terminal_state] = 100 # terminal state

    gamma=0.99 # discount factor from bellman equation
    epsilon=0.9 # exploration rate, ε as described above
    max_episodes=1000 # number of iterations
    epsilon_decay = 0.995 # epsilon decay amount after each episode

    for episode in tqdm(range(max_episodes)):
        state = initial_state
        while not terminal(state, terminal_state):
            actions = available_actions(state, state_space)
            # next action decided using epsilon-greedy method
            best_action = max(actions, key=lambda a: V[get_next_state(state, a)])
            next_action = np.random.choice(actions) if np.random.random() < epsilon else best_action
            next_state = transition(state, next_action)
            # Bellman update 
            max_future_value = max(V[get_next_state(state, a)] for a in actions)
            V[state] = 0 + gamma * max_future_value # we have no known rewards, so we assume 0?
            state = next_state
        epsilon *= epsilon_decay
    return V

def solutions(V, state_space, initial_state, terminal_state):
    visual = np.zeros(state_space)
    state = initial_state
    visual[state] = 1
    while not terminal(state, terminal_state):
        actions = available_actions(state, V.shape)
        best_action = max(actions, key=lambda a: V[get_next_state(state, a)])
        state = transition(state, best_action)
        visual[state] = 1
    return visual

def plot_maze(maze):
    plt.figure(figsize=(10, 10))
    sns.heatmap(maze, square=True, cbar=False, cmap='coolwarm', linecolor='black', linewidth=1, annot=True, fmt=".4f")
    plt.show()


V = rtdp(
    state_space=(3, 5), 
    actions=["up", "down", "left", "right"], 
    initial_state=(1, 1), 
    terminal_state=(2, 4)
)

plot_maze(V)

# print(V)

visual = solutions(V, (3, 5), (1, 1), (2, 4))

# print(visual)

plot_maze(visual)


