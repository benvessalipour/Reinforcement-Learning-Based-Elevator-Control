from pathlib import Path
from tqdm import tqdm
import numpy, pygame, imageio, matplotlib

from Environment import *

def baseline(state):
  """
  This is a handcrafted baseline policy.
  The policy will move the lift to the nearest floor with a call button pressed.
  If the lift is already at the desired floor, it will serve the passengers.
  Afterwards, it heads to the next floor with a cabin button pressed.
  If the lift is at the target location, it will repeat this logic.

  This policy serves as a baseline to see the benefits of using reinforcement learning.
  Hopefully, the RL agent will learn a better policy than this one.

  Parameters
  ----------
  state : tuple
    A state of the environment.

  Returns
  -------
  action : str
    The action to take.
  """

  # Unpack the state for easier access
  current_floor, move_direction, door_state, cabin_buttons, call_buttons = state

  raise RuntimeError("The baseline policy could no produce a valid action")

def run(policy, iterations=30, progress_bar=True):

  # A fresh environment is created for each run
  # The frames_dir parameter specifies where the render output is saved
  env = Environment(frames_dir=Path('./images/frames'))

  # The environment is reset to its initial (random) state
  state = env.reset()

  # The simulation loop, for convenience it can be wrapped in a tqdm progress bar
  iterations = range(iterations)

  if progress_bar:
    iterations = tqdm(iterations)

  for _ in iterations:

    # Create or update the visualisation
    env.render()

    # The policy function is called with the current state as input to define an action
    action = policy(state)

    # The action is executed in the environment and the new state is returned
    state = env.step(action)

  # The environment is closed and the visualisation is saved
  env.close()




  return

if __name__ == "__main__":
  run(policy.alternate)
  # run(baseline)
  # run(policy.keyboard, iterations=1000, progress_bar=False)
