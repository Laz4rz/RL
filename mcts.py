import numpy as np
from tqdm import tqdm

'''
This script is an implementation of the Monte Carlo Tree Search (MCTS) algorithm.
The problem solved is probably the simplest problem available, but shouldn't be too hard
to adapt to more complex problems.

The problem is a NxM maze, where the agent starts at some (x, y) position and has to reach
a terminal state at some (x', y') position. The agent can move up, down, left, or right 
freely, only being blocked by the walls of the maze. The agent has no knowledge of the maze.

Some ideas for better convergence to optimal path length:
 - add global depth, discount based on the global depth in backprop (gamma)
 - track best path, adapt reward to whether best path was beaten
 - change C constant in UCB score
'''

class Node:
    def __init__(self, action, state, parent=None):
        self.action = action
        self.state = state
        self.parent = parent
        self.children = {}  # maps actions to nodes
        self.visits = 0
        self.score = 0

    def __repr__(self):
        return f"Node(action={self.action}, state={self.state}, visits={self.visits}, score={self.score})"


def softmax(x):
    """Compute softmax values for each element of array x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def ucb_score(parent, child, C=1.4):
    if child.visits == 0 or parent.visits == 0:
        return float('inf')
    return child.score/child.visits + C*np.sqrt(2*np.log(parent.visits)/child.visits)

def select(node, state_space, action_picker='softmax'):
    """Select a path through the tree to a leaf node."""
    current = node
    
    while current.children:  # while we have children
        possible_actions = get_possible_actions(current.state, state_space)
        
        if not all(a in current.children for a in possible_actions):
            return current
            
        ucb_values = {
            action: ucb_score(current, child)
            for action, child in current.children.items()
        }
        # max picking
        if action_picker == 'max':
            best_action = max(ucb_values.items(), key=lambda x: x[1])[0]
        # softmax multinomial sampling
        elif action_picker == 'softmax':
            probabilities = softmax(list(ucb_values.values()))
            best_action = np.random.choice(list(ucb_values.keys()), p=probabilities)
        current = current.children[best_action]
    
    return current

def expand(node, possible_actions):
    """Add a new child node with an unexplored action."""
    # Get list of unexplored actions
    unexplored_actions = [a for a in possible_actions if a not in node.children]
    
    if not unexplored_actions:
        return node  # No expansion possible
        
    # Choose random unexplored action
    action = np.random.choice(unexplored_actions)
    
    # Create new child node
    new_state = transition(node.state, action)
    child = Node(action, new_state, parent=node)
    node.children[action] = child
    
    return child

def playout(state, state_space, terminal_state, max_depth=10):
    """Run a random simulation from state to terminal state or max_depth."""
    current_state = state
    depth = 0
    
    while depth < max_depth:
        if current_state == terminal_state:
            return 100 - depth
            
        possible_actions = get_possible_actions(current_state, state_space)
            
        if not possible_actions:
            break
        
        action = np.random.choice(possible_actions)
        current_state = transition(current_state, action)
        depth += 1
    
    return -depth

def backpropagate(node, reward):
    """Update node statistics back up the tree."""
    current = node
    while current:
        current.visits += 1
        current.score += reward
        current = current.parent

def transition(state, action):
    """Compute new state after taking action."""
    if action == 'up':
        return (state[0]-1, state[1])
    elif action == 'down':
        return (state[0]+1, state[1])
    elif action == 'left':
        return (state[0], state[1]-1)
    elif action == 'right':
        return (state[0], state[1]+1)

def get_possible_actions(state, state_space):
    """Get list of valid actions from current state."""
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

def mcts(initial_state, terminal_state, state_space, max_iter=1000, debug=False, action_picker='softmax'):
    root = Node("", initial_state)
    
    for i in range(max_iter):
        if debug and i % 100 == 0:
            print(f"\nIteration {i}:")
        
        # 1. Selection
        node = select(root, state_space)
        if debug and i % 100 == 0:
            print(f"Selected node: {node}")
        
        # 2. Expansion
        if node.state != terminal_state:
            possible_actions = get_possible_actions(node.state, state_space)
            node = expand(node, possible_actions)
            if debug and i % 100 == 0:
                print(f"Expanded to: {node}")
        
        # 3. Simulation
        reward = playout(node.state, state_space, terminal_state)
        if debug and i % 100 == 0:
            print(f"Playout reward: {reward}")
        
        # 4. Backpropagation
        backpropagate(node, reward)
        
        if debug and i % 100 == 0:
            print("Root children stats:")
            for action, child in root.children.items():
                print(f"{action}: visits={child.visits}, score={child.score}")
    
    if not root.children:
        return None
        
    return root  

def print_state_space(state_space, current_state, terminal_state):
    """Visualize the state space with current and terminal states."""
    for i in range(state_space[0]):
        for j in range(state_space[1]):
            if (i, j) == current_state:
                print('C', end=' ')
            elif (i, j) == terminal_state:
                print('T', end=' ')
            else:
                print('.', end=' ')
        print()

def children_stats(node):
    """Print statistics of children nodes."""
    for action, child in node.children.items():
        print(f"{action}: visits={child.visits}, score={child.score}")

def best_path(node):
    """Find the best path from the root node."""
    current = node
    path = []
    
    while current.children:
        best_action = max(current.children, key=lambda a: current.children[a].score)
        path.append(best_action)
        current = current.children[best_action]
    
    return path

if __name__ == "__main__":
    state_space = (3, 5)
    initial_state = (1, 1)
    terminal_state = (2, 4)
    
    print("\nInitial state:")
    print_state_space(state_space, initial_state, terminal_state)
    
    root = mcts(initial_state, terminal_state, state_space, max_iter=1000, debug=False)
    print("\nBest path:")
    print(best_path(root))
    print("\nFinal state space best path:")
    state = initial_state
    for action in best_path(root):
        state = transition(state, action)
        print_state_space(state_space, state, terminal_state)
        print()

    for max_iter in [50, 100, 500, 1000, 5000]:
        avg_path_length = 0
        for _ in range(10):
            root = mcts(initial_state, terminal_state, state_space, max_iter=1000, debug=False)
            path = best_path(root)
            avg_path_length += len(path)
        avg_path_length /= 10
        print(f"Average path length for {max_iter} iterations: {avg_path_length}")
