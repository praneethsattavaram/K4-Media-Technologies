import cv2
import numpy as np
import gym
import tensorflow as tf
from keras import models, layers, Input
from collections import deque
import random

# Define the game environment
class GameEnv(gym.Env):
    def __init__(self):
        super(GameEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.max_steps = 100  # Maximum steps per episode
        self.current_step = 0

    def reset(self):
        self.state = self._get_game_screen()
        self.current_step = 0
        return self.state

    def step(self, action):
        self._apply_action(action)
        next_state = self._get_game_screen()
        reward = self._get_reward()
        done = self._check_done()
        return next_state, reward, done, {}

    def _get_game_screen(self):
        screen = cv2.imread("game_screen.png")
        if screen is None:
            raise FileNotFoundError("The file game_screen.png does not exist or cannot be read.")
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (84, 84))
        return screen[:, :, np.newaxis]

    def _apply_action(self, action):
        # Simulate action: this is where you would add logic to update the game state based on the action
        self.current_step += 1

    def _get_reward(self):
        # Simple reward function: give a reward of 1 for every step survived
        return 1

    def _check_done(self):
        # End the episode after max_steps or other stopping condition
        return self.current_step >= self.max_steps

# Define the DQN model
def create_dqn_model(input_shape, num_actions):
    model = models.Sequential()
    model.add(Input(shape=input_shape))
    model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_actions, activation='linear'))
    return model

# Initialize replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Initialize replay memory
memory = ReplayMemory(10000)

def preprocess_state(state):
    return np.expand_dims(state, axis=0)

# Hyperparameters
batch_size = 32
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update_freq = 10
num_episodes = 10
learning_rate = 0.00025

# Create the DQN and target network
env = GameEnv()
model = create_dqn_model(env.observation_space.shape, env.action_space.n)
target_model = create_dqn_model(env.observation_space.shape, env.action_space.n)
target_model.set_weights(model.get_weights())

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()

# Training loop
epsilon = epsilon_start
for episode in range(num_episodes):
    state = env.reset()
    state = preprocess_state(state)
    total_reward = 0

    for t in range(env.max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values)

        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(next_state)
        total_reward += reward

        memory.push((state, action, reward, next_state, done))
        state = next_state

        if done:
            break

        if len(memory) > batch_size:
            batch = memory.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = np.concatenate(states)
            next_states = np.concatenate(next_states)

            q_values_next = target_model.predict(next_states)
            target_q_values = np.array(rewards) + gamma * np.max(q_values_next, axis=1) * (1 - np.array(dones))

            masks = tf.one_hot(actions, env.action_space.n)
            with tf.GradientTape() as tape:
                q_values = model(states)
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                loss = loss_fn(target_q_values, q_action)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if t % target_update_freq == 0:
            target_model.set_weights(model.get_weights())

    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    print(f'Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}')

print("Training completed.")
