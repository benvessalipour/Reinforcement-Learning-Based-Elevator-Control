from .constants import *
from .environment import Environment

def up(state):
  """
  This policy always chooses the action "up".

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

  # Close the door if it is open
  if door_state == DOOR_OPEN:
    return ACTION_DOOR

  allowed = Environment.get_available_actions(state)

  # Move up when possible
  if ACTION_UP in allowed:
    return ACTION_UP

  return ACTION_NOOP

def alternate(state):
  """
  This policy alternates between the actions "up" and "down".

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

  # Close the door if it is open
  if door_state == DOOR_OPEN:
    return ACTION_DOOR

  elif current_floor == 0 and move_direction == DIRECTION_NONE:
    return ACTION_UP

  elif current_floor == NUMBER_OF_FLOORS - 1 and move_direction == DIRECTION_NONE:
    return ACTION_DOWN

  elif move_direction == DIRECTION_NONE:
    return ACTION_UP

  else:
    return ACTION_NOOP

def keyboard(state):
  """
  This policy allows the user to control the lift using the keyboard.

  Parameters
  ----------
  state : tuple
    A state of the environment.

  Returns
  -------
  action : str
     The action to take.
  """

  # Show the available actions
  print("Available actions:")

  choices = Environment.get_available_actions(state)
  for i, action in enumerate(choices):
    print(f"{i:2d}: {action}")

  # The default value is the first action (in case of invalid input)
  action = choices[0]

  # The user has 5 tries to enter a valid action
  for i in range(5):
    try:
      choice = int(input("Choose an action: "))
      action = choices[choice]
      break
    except ValueError:
      print("Please enter a number.")
      continue
    except IndexError:
      print("Please enter a valid action index.")
      continue
  else:
    print("No valid input received. Choosing the first action.")

  return action
