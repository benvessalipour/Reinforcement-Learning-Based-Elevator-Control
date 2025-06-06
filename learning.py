import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from Environment.environment import Environment
from Environment.constants import ACTIONS, NUMBER_OF_FLOORS, DIRECTION_UP, DIRECTION_DOWN, DIRECTION_NONE, DOOR_OPEN, \
    DOOR_CLOSED, ACTION_DOOR, ACTION_UP, ACTION_DOWN, ACTION_STOP, ACTION_NOOP


def simplify_state(state):
    current_floor, move_direction, door_state, cabin_buttons, call_buttons = state
    return (
        current_floor,
        move_direction,
        door_state,
        tuple(cabin_buttons),
        tuple(call_buttons)  # call_buttons ist flache Liste
    )


def effective_reward(env, state, action, next_state, prev_persons):
    reward = -0.05  # Kleine negative Belohnung pro Schritt

    current_floor, move_direction, door_state, cabin_buttons, call_buttons = state

    # Große Belohnung für Türöffnen bei wartenden Passagieren
    if action == ACTION_DOOR and door_state == DOOR_CLOSED:
        if call_buttons[current_floor] or cabin_buttons[current_floor]:
            reward += 10

    # Belohnung für Richtungswechsel bei Bedarf
    if action in [ACTION_UP, ACTION_DOWN] and move_direction == DIRECTION_NONE:
        reward += 2

    # Sehr hohe Belohnung für Passagierablieferung
    new_persons = env.get_active_persons()
    delivered = prev_persons - new_persons
    if delivered > 0:
        reward += delivered * 50

    # Strafe für unnötiges Öffnen/Schließen
    if action == ACTION_DOOR:
        if door_state == DOOR_OPEN and not (call_buttons[current_floor] or cabin_buttons[current_floor]):
            reward -= 5

    return reward


def choose_action(state, epsilon):
    allowed_actions = Environment.get_available_actions(state)

    if random.random() < epsilon:
        return random.choice(allowed_actions)

    state_key = simplify_state(state)
    q_values = Q[state_key]

    best_value = -float('inf')
    best_action = allowed_actions[0]

    for action in allowed_actions:
        if q_values[action] > best_value:
            best_value = q_values[action]
            best_action = action

    return best_action


# Hyperparameter optimiert
alpha = 0.3  # Höhere Lernrate
gamma = 0.9  # Weniger Fokus auf langfristige Belohnung
epsilon = 0.3  # Mehr Exploration
episodes = 3000
steps_per_episode = 400

Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
rewards = []
moving_avgs = []

for ep in range(episodes):
    env = Environment(render_mode="none")
    state = env.reset()
    total_reward = 0

    for step in range(steps_per_episode):
        prev_persons = env.get_active_persons()
        action = choose_action(state, epsilon)
        next_state = env.step(action)

        reward = effective_reward(env, state, action, next_state, prev_persons)
        total_reward += reward

        # Q-Learning Update mit Fehlerbehandlung
        state_key = simplify_state(state)
        next_state_key = simplify_state(next_state)

        current_q = Q[state_key][action]
        next_max = max(Q[next_state_key].values()) if next_state_key in Q else 0
        new_q = current_q + alpha * (reward + gamma * next_max - current_q)

        Q[state_key][action] = new_q
        state = next_state

    rewards.append(total_reward)

    # Gleitenden Durchschnitt berechnen
    if ep >= 100:
        moving_avg = np.mean(rewards[-100:])
        moving_avgs.append(moving_avg)

    # Epsilon verringern
    epsilon = max(0.05, epsilon * 0.995)

    # Fortschrittsausgabe
    if ep % 200 == 0:
        avg_reward = np.mean(rewards[-50:]) if len(rewards) > 50 else total_reward
        print(f"Episode {ep:04d}/{episodes} | Reward: {total_reward:7.1f} | Avg: {avg_reward:7.1f} | ε: {epsilon:.3f}")

# Lernkurve mit gleitendem Durchschnitt plotten
plt.figure(figsize=(12, 6))
plt.plot(rewards, alpha=0.3, label='Episode Rewards')
if moving_avgs:
    plt.plot(range(100, 100 + len(moving_avgs)), moving_avgs, 'r-', linewidth=2, label='Moving Avg (100 episodes)')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Verbesserte Lernkurve des Aufzug-Agents")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("improved_learning_curve.png")
plt.show()