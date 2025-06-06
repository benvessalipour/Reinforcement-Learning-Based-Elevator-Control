import numpy as np

# Names for movement directions
DIRECTION_UP = "up"
DIRECTION_NONE = "none"
DIRECTION_DOWN = "down"
DIRECTIONS = [DIRECTION_UP, DIRECTION_NONE, DIRECTION_DOWN]

# Names for the door states
DOOR_OPEN = "open"
DOOR_CLOSED = "closed"
DOORS = [DOOR_OPEN, DOOR_CLOSED]

# Names of the actions
ACTION_UP = "up"
ACTION_DOWN = "down"
ACTION_STOP = "stop"
ACTION_DOOR = "door"
ACTION_NOOP = "noop"
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_STOP, ACTION_DOOR, ACTION_NOOP]

# A stochastic description of people using the lift.
# Each row indicates the start floor of a person.
# Each column corresponds to the desired destination.
# The number in each cell is the probability that such a person appears in one timestep.
PASSENGER_DISTRIBUTION = np.array([
  [0.0000, 0.0016, 0.0008, 0.0034, 0.0082, 0.0089, 0.0056],
  [0.0012, 0.0000, 0.0005, 0.0028, 0.0070, 0.0100, 0.0067],
  [0.0006, 0.0004, 0.0000, 0.0001, 0.0001, 0.0003, 0.0001],
  [0.0022, 0.0015, 0.0001, 0.0000, 0.0001, 0.0006, 0.0009],
  [0.0056, 0.0036, 0.0001, 0.0002, 0.0000, 0.0011, 0.0012],
  [0.0058, 0.0061, 0.0002, 0.0003, 0.0005, 0.0000, 0.0004],
  [0.0048, 0.0049, 0.0001, 0.0007, 0.0009, 0.0010, 0.0000]])

NUMBER_OF_FLOORS = PASSENGER_DISTRIBUTION.shape[0]


