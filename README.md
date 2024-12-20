# K4-Media-Technologies

The code provided sets up a Deep Q-Network (DQN) to play a game using reinforcement learning. Here's a breakdown of how the code works and how it plays the game:

### Components of the Code

1. **Game Environment (`GameEnv`)**:
   - **Initialization**: Sets up the action space (4 discrete actions) and observation space (84x84 grayscale image). The environment runs for a maximum of 100 steps per episode.
   - **Reset**: Resets the game to the initial state and returns the initial game screen.
   - **Step**: Takes an action, updates the game state, calculates the reward, and checks if the episode is done.
   - **Private Methods**: 
     - `_get_game_screen`: Reads and processes the game screen image.
     - `_apply_action`: Placeholder for applying the game action.
     - `_get_reward`: Returns a reward of 1 for each step survived.
     - `_check_done`: Checks if the episode has reached the maximum steps.

2. **DQN Model**:
   - A convolutional neural network (CNN) that processes the game screen and outputs Q-values for each action.

3. **Replay Memory (`ReplayMemory`)**:
   - Stores experiences `(state, action, reward, next_state, done)` to train the DQN.

4. **Training Loop**:
   - **Hyperparameters**: Set up parameters like batch size, discount factor, exploration rate, learning rate, and number of episodes.
   - **Initialization**: Creates the environment, the DQN model, and the target model. Initializes the optimizer and loss function.
   - **Episode Loop**: For each episode:
     - **Reset Environment**: Gets the initial state.
     - **Step Loop**: For each step:
       - **Action Selection**: Chooses an action using an epsilon-greedy policy.
       - **Environment Interaction**: Takes the action in the environment and observes the next state and reward.
       - **Store Experience**: Saves the experience in replay memory.
       - **Training**: If there are enough experiences, samples a batch from memory, calculates target Q-values, computes loss, and updates the model.
       - **Target Network Update**: Periodically updates the target network to match the main DQN model.
     - **Epsilon Decay**: Reduces exploration rate.

### How the Code Plays the Game

1. **Initialization**:
   - The game environment (`GameEnv`) is created.
   - The DQN model is created to approximate the Q-values for each action.
   - A target model is also created and periodically updated to stabilize training.

2. **Training Loop**:
   - For each episode, the environment is reset, and the game screen is processed.
   - For each step within an episode:
     - An action is selected using an epsilon-greedy policy (exploration vs. exploitation).
     - The environment executes the action, returning the next state, reward, and whether the episode is done.
     - The experience is stored in replay memory.
     - If there are enough experiences in memory, a batch is sampled to train the DQN:
       - The model predicts Q-values for the current states and next states.
       - Target Q-values are calculated using the rewards and the maximum Q-values from the next states.
       - The DQN model is trained to minimize the difference between its Q-value predictions and the target Q-values.
     - The target model is updated periodically to match the main model.

3. **Action Execution**:
   - During each step, the selected action is applied to the environment, which simulates the game’s response.
   - The reward is obtained based on the action's result, and the next state is captured.

4. **End of Episode**:
   - The episode ends when the maximum number of steps is reached or another termination condition is met.
   - The exploration rate (`epsilon`) is decayed to reduce random actions over time.

### Key Points

- The environment simulates a game by loading and processing game screen images.
- The DQN model learns to play the game by approximating the optimal Q-values through experience replay and target network updates.
- The training loop iteratively improves the model's performance by balancing exploration and exploitation.
- The code assumes that `game_screen.png` is available and represents the game state. This file should be dynamically updated to reflect the game's current state for each step.

If you encounter any issues or need further customization, let me know!
