from pathlib import Path
from tqdm import tqdm
import numpy, pygame, imageio, matplotlib, random
from Environment.environment import Environment
from Environment.constants import ACTIONS
from Environment.policy import up
from types import SimpleNamespace
import matplotlib.pyplot as plt
from collections import defaultdict
from Environment.constants import DOOR_OPEN, ACTION_DOOR, ACTION_UP, ACTION_DOWN, ACTION_NOOP, NUMBER_OF_FLOORS

def simplify_state(state):
    current_floor, move_direction, door_state, cabin_buttons, call_buttons = state
    return (
        current_floor,
        move_direction,
        door_state,
        tuple(cabin_buttons),
        tuple(call_buttons),
    )


def greedy_policy(state):
    q_vals = Q[simplify_state(state)]
    allowed = Environment.get_available_actions(state)
    return max(allowed, key=lambda a: q_vals[a])

def rollout(policy_fn = up, length = 1):
    env = Environment()
    state = env.reset()
    tau = SimpleNamespace(x=[],u=[],x2=[])

    for _ in range(length):
        if random.random() < 0.1:
            action = random.choice(Environment.get_available_actions(state))
        else:
            action = policy_fn(state)

        next_state = env.step(action)
        env.render()

        tau.x.append(state)
        tau.u.append(action)
        tau.x2.append(next_state)

        state = next_state
    return tau


# Reward-Funktion
def g1(state, action, next_state, prev_persons, new_persons):
    return -1 + (prev_persons - new_persons) * 20

# Zweite Kostenfunktion (dünn / sparse)
def g2(state, action, next_state, prev_persons, new_persons):
    return 50 * (prev_persons - new_persons)


# Reward-Funktion
def g1(state, action, next_state, prev_persons, new_persons):
    return -1 + (prev_persons - new_persons) * 20

# Q-Learning Parameter
alpha = 0.1      # Lernrate
gamma = 0.99     # Diskontfaktor 0.95,0.9?
epsilon = 0.1    # Epsilon-Greedy
episodes = 500   # Anzahl Trainingsepisoden
steps_per_episode = 100  # Schritte pro Episode

# Kostenfunktion wählen
cost_function = g1  # oder g2

# Q-Table: Q[state][action] → float
Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

# Logging
episode_rewards = []

# Epsilon-greedy Aktionswahl
def choose_action(state):
    allowed = Environment.get_available_actions(state)

    if random.random() < epsilon:
        return random.choice(allowed)

    q_vals = Q[simplify_state(state)]
    # Wichtig: Nur erlaubte Aktionen betrachten!
    return max(allowed, key=lambda a: q_vals[a])

# Training Loop
for ep in range(episodes):
    env = Environment(render_mode="none")  # kein Pygame-Fenster
    state = env.reset()
    total_reward = 0

    for step in range(steps_per_episode):
        action = choose_action(state)
        prev_persons = env.get_active_persons()  # Personen VOR der Aktion
        next_state = env.step(action)  # Aktion wird NUR EINMAL ausgeführt
        new_persons = env.get_active_persons()  # Personen NACH der Aktion
        reward = cost_function(state, action, next_state, prev_persons, new_persons)

        # Q-Learning Update
        allowed_next = Environment.get_available_actions(next_state)
        best_next = max(Q[simplify_state(next_state)][a] for a in allowed_next) if allowed_next else 0
        Q[simplify_state(state)][action] = (1 - alpha) * Q[simplify_state(state)][action] + alpha * (reward + gamma * best_next)

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)
    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1}: Total Reward = {total_reward}")

# Vergleich von g1 und g2
def train(cost_fn, label):
    Q_local = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    rewards = []

    for ep in range(episodes):
        env = Environment(render_mode="none")
        state = env.reset()
        total_reward = 0

        for step in range(steps_per_episode):
            allowed = Environment.get_available_actions(state)
            if random.random() < epsilon:
                action = random.choice(allowed)
            else:
                q_vals = Q_local[simplify_state(state)]
                action = max(allowed, key=lambda a: q_vals[a])

            prev_persons = env.get_active_persons()
            next_state = env.step(action)
            new_persons = env.get_active_persons()
            reward = cost_fn(state, action, next_state, prev_persons, new_persons)

            allowed_next = Environment.get_available_actions(next_state)
            best_next = max(Q_local[simplify_state(next_state)][a] for a in allowed_next) if allowed_next else 0
            Q_local[simplify_state(state)][action] = (1 - alpha) * Q_local[simplify_state(state)][action] + alpha * (reward + gamma * best_next)

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        if (ep + 1) % 50 == 0:
            print(f"{label} Episode {ep+1}: Total Reward = {total_reward}")
    return rewards

rewards_g1 = train(g1, "g1")
rewards_g2 = train(g2, "g2")

# Fortschritt anzeigen
plt.plot(rewards_g1, label="g1: -1 + 20 * (prev-new)")
plt.plot(rewards_g2, label="g2: 50 * (prev-new)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Vergleich der Kostenfunktionen")
plt.legend()
plt.show()

def reference_policy(state):
    current_floor, move_direction, door_state, cabin_buttons, call_buttons = state
    allowed = Environment.get_available_actions(state)

    # Wenn Tür offen, schließe sie
    if door_state == DOOR_OPEN:
        return ACTION_DOOR

    # Falls jemand aussteigen will → anhalten & Tür öffnen
    if cabin_buttons[current_floor] or call_buttons[current_floor]:
        if ACTION_DOOR in allowed:
            return ACTION_DOOR

    # Falls Kabinenziel gedrückt ist → fahre in Richtung Ziel
    for dir_action, dir_delta in [(ACTION_UP, +1), (ACTION_DOWN, -1)]:
        next_floor = current_floor + dir_delta
        if 0 <= next_floor < NUMBER_OF_FLOORS:
            if cabin_buttons[next_floor] or call_buttons[next_floor]:
                if dir_action in allowed:
                    return dir_action

    # Suche nächstes Ziel in beliebiger Richtung
    for f in range(NUMBER_OF_FLOORS):
        if cabin_buttons[f] or call_buttons[f]:
            if f > current_floor and ACTION_UP in allowed:
                return ACTION_UP
            elif f < current_floor and ACTION_DOWN in allowed:
                return ACTION_DOWN

    # Kein Ziel: nichts tun
    return ACTION_NOOP

print("Evaluating greedy policy:")
tau = rollout(policy_fn=greedy_policy, length=100)
print("Steps:", len(tau.x))

print("Evaluating reference policy:")
tau = rollout(policy_fn=reference_policy, length=100)
print("Steps:", len(tau.x))