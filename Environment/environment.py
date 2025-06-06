from .constants import *

class Person:
  """
  This class represents a person in the lift world.

  This is a simple class to collect required variables and eases the debugging process.
  """

  def __init__(self, start, destination):
    """
    Initializes a person with a start and destination floor.

    Parameters
    ----------
    start : int
      The start floor of the person.

    destination : int
      The destination floor of the person.
    """
    self.start = start
    self.destination = destination

  def __repr__(self):
    return f"Person(start={self.start}, destination={self.destination})"

class Environment:
  """
  This class represents the state and actions space as well as the transition probabilities of an lift.
  The environment is an infinite-horizon decision-making task, where the goal is to transport people between floors.

  The state is represented as a tuple of the following elements:
  * current_floor: The current floor of the lift as integer.
  * move_direction: The direction in which the lift is moving.
  * door_state: The state of the lift door (open or closed).
  * cabin_buttons: A tuple of booleans indicating which buttons are pressed in the cabin.
  * call_buttons: A tuple of booleans indicating which buttons are pressed on the floors.

  The actions are given by strings, where each string represents an action that can be taken in the environment.

  The reward function is defined outside the environment and can be tailored towards the learning algorithm.

  The environment can be rendered in two modes: "human" and "rgb_array".
  * Human: The environment is rendered in a window through Pygame.
  * rgb_array: The environment is rendered as a numpy array for use in other applications.
  """

  metadata = {
    "render_modes": ["human", "rgb_array"],
    "render_fps": 5,
  }

  def __init__(self, max_capacity=4, render_mode='human', frames_dir=None, seed=None):
    """ Creates a fresh instance of the lift environment. """

    self.render_mode = render_mode
    self.screen_width = 600
    self.screen_height = 800
    self.screen = None
    self.clock = None

    self.seed = seed
    self.max_capacity = max_capacity
    self.person_counter = 0
    self.frame_count = 0
    self.frames_dir = frames_dir

    self.buffer_cabin = []
    self.buffer_floor = {i: [] for i in range(NUMBER_OF_FLOORS)}
    self.state = None

    self.reset()
    return

  def step(self, action):
    """
    Take a step in the environment.

    Parameters
    ----------
    action : str
      The action to take in the environment. Must be one of the actions in the action space.

    Returns
    -------
    new_state : tuple
      The new state of the environment after taking the action.
    """

    #Überprüfe erlaubten actions
    valid_actions = Environment.get_available_actions(self.state)

    if action not in valid_actions:
      raise ValueError(f"It is not allowed to execute <{action}> in state {self.state}.\n"
                       f"Valid actions are {valid_actions}.")

    # Unpack the state for easy access
    current_floor, move_direction, door_state, cabin_buttons, call_buttons = self.state

    # Convert the buttons to a list for easier manipulation
    cabin_buttons = list(cabin_buttons)
    call_buttons = list(call_buttons)

    # The call button is active on every floor, where people are waiting.
    self._update_call_buttons(call_buttons)

    # If the door is open, people leave and enter the cabin
    if door_state == DOOR_OPEN:

      # The buttons for the current floor are turned off since that floor is served
      cabin_buttons[current_floor] = False
      call_buttons[current_floor] = False

      # Let people out, they arrived at their desired floor and are removed from the buffer
      at_destination = [p for p in self.buffer_cabin if p.destination == current_floor]

      for p in at_destination:
        self.buffer_cabin.remove(p)

      # Let people in (as long as there is space) and let the press the cabin buttons
      # If not all fit, then the call button is activated again during the next step
      self._move_in_cabin(current_floor)
      self._update_cabin_buttons(cabin_buttons)

      if action == ACTION_DOOR:
        door_state = DOOR_CLOSED

    # If the door is closed, the lift moves or waits
    else:

      # If waiting, doors can be opened or the lift can start to move
      if move_direction == DIRECTION_NONE:

        # Open the door
        if action == ACTION_DOOR:
          door_state = DOOR_OPEN

        # Move the lift upwards
        elif action == ACTION_UP:
          move_direction = DIRECTION_UP

        # Move the lift downwards
        elif action == ACTION_DOWN:
          move_direction = DIRECTION_DOWN

        # Do nothing (needed to wait at the optimal floor if no one is requesting the lift)
        elif action == ACTION_NOOP:
          pass

        else:
          raise RuntimeError("This should not happen")

      # If moving, the lift updates its current floor and can be asked to stop
      else:

        # The lift moves to the next floor (or stops if the top floor is reached)
        if move_direction == DIRECTION_UP:
          next_floor = current_floor + 1

          # Issue a stop command if the top floor is about to be reached
          if next_floor == NUMBER_OF_FLOORS - 1:
            move_direction = DIRECTION_NONE

          current_floor = next_floor

        # The lift moves to the previous floor (or stops if the ground floor is reached)
        elif move_direction == DIRECTION_DOWN:
          next_floor = current_floor - 1

          # Issue a stop command if the ground floor is about to be reached
          if next_floor == 0:
            move_direction = DIRECTION_NONE

          current_floor = next_floor

        else:
          raise RuntimeError("This should not happen")

        # The lift needs to stop at the next floor
        if action == ACTION_STOP:
          move_direction = DIRECTION_NONE

        # Keep moving
        elif action == ACTION_NOOP:
          pass

        else:
          raise RuntimeError("This should not happen")

      # end if moving
    # end if door

    # Combine all parts into the next state.
    # Lists are converted back to tuples to ensure immutability of the state
    self.state = current_floor, move_direction, door_state, tuple(cabin_buttons), tuple(call_buttons)

    # Generate new persons with random start and destination floors for the next step.
    # This is done after the state transition to avoid the new persons to appear in the cabin in the same step.
    # This would look weird during rendering
    self._new_persons()

    return self.state

  def reset(self):
    """
    Reset the environment to its initial state.

    Returns
    -------
    state : tuple
      The fresh initial state of the environment.

    info : dict
      Additional information about the environment for debugging purposes.
    """

    if self.seed is not None:
      np.random.seed(self.seed)

    self.frames = []
    self.frame_count = 0
    self.person_counter = 0

    self.buffer_cabin = []
    self.buffer_floor = {i: [] for i in range(NUMBER_OF_FLOORS)}

    # Random starting position
    current_floor = np.random.randint(NUMBER_OF_FLOORS)

    # These are fixed because this makes the resetting behaviour easier to implement
    move_direction = DIRECTION_NONE
    door_state = DOOR_CLOSED

    # No buttons are pressed at the beginning
    # Can be changed if persons are spawned in the later code
    cabin_buttons = list([False for _ in range(NUMBER_OF_FLOORS)])
    call_buttons = list([False for _ in range(NUMBER_OF_FLOORS)])

    # Create persons at the beginning by exploiting the binomial distribution
    # in the existing function. This will handle buffers and counters correctly
    persons_to_create = np.random.randint(10)
    attempts_left = 15

    while self.get_active_persons() < persons_to_create and attempts_left > 0:
      attempts_left -= 1
      self._new_persons()

    # Move persons into the cabin on the floor where the lift starts
    # It is fine if there are more persons than the max capacity
    self._move_in_cabin(current_floor)

    # Adjust the state of buttons according to the existing persons
    self._update_cabin_buttons(cabin_buttons)
    self._update_call_buttons(call_buttons)

    self.state = (current_floor,
                  move_direction,
                  door_state,
                  tuple(cabin_buttons),
                  tuple(call_buttons))

    return self.state











































  def render(self):
    """
    Render the current state of the environment.

    Returns
    -------
    image : np.ndarray or None
      The rendered image of the environment, if render_mode is "rgb_array".
    """

    # The import is done here to avoid a dependency on Pygame if the environment is not rendered
    # E.g. training on a headless server
    import pygame

    # Unpack the state for easy access
    current_floor, move_direction, door_state, cabin_buttons, call_buttons = self.state

    # Initialise the Pygame window if it has not been initialised yet
    if self.screen is None:
      pygame.init()

      # The environment can be rendered in two modes: "human" and "rgb_array".
      # Human: The environment is rendered in a window through Pygame.
      # rgb_array: The environment is rendered as a numpy array for use in other applications.
      if self.render_mode == "human":
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
      else:
        self.screen = pygame.Surface((self.screen_width, self.screen_height))

    # Initialize the clock for the Pygame window, which is used to control the frame rate
    if self.clock is None:
      self.clock = pygame.time.Clock()

    # Define the dimensions of the lift and the floors
    info_height = 150
    floor_height = (self.screen_height - info_height) / NUMBER_OF_FLOORS
    lift_width = 100

    # Create the surfaces for the lift, text, and figures
    lift_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
    text_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
    figure_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)

    def draw_stickman(position, color=(0, 0, 0)):
      """ Draw a stick figure at the given position. """

      pos_x, pos_y = position

      # Head
      pygame.draw.circle(figure_surface, color, center=(pos_x, pos_y - 15), radius=5)
      # Body
      pygame.draw.line(figure_surface, color, start_pos=position, end_pos=(pos_x, pos_y - 15))

      # Left leg
      pygame.draw.line(figure_surface, color, start_pos=position, end_pos=(pos_x - 10, pos_y + 20))
      # Right leg
      pygame.draw.line(figure_surface, color, start_pos=position, end_pos=(pos_x + 10, pos_y + 20))

      # Left arm
      pygame.draw.line(figure_surface, color, start_pos=(pos_x, pos_y - 5), end_pos=(pos_x - 10, pos_y - 10))
      # Right arm
      pygame.draw.line(figure_surface, color, start_pos=(pos_x, pos_y - 5), end_pos=(pos_x + 10, pos_y - 10))

      return

    # Draw floors and people waiting
    for i in range(NUMBER_OF_FLOORS):
      pygame.draw.line(lift_surface,
                       color=(150, 150, 150),
                       start_pos=(0, i * floor_height),
                       end_pos=((self.screen_width - lift_width) // 2, i * floor_height),
                       width=1)
      pygame.draw.line(lift_surface,
                       color=(150, 150, 150),
                       start_pos=((self.screen_width + lift_width) // 2, i * floor_height),
                       end_pos=(self.screen_width, i * floor_height),
                       width=1)
      font = pygame.font.SysFont('Arial', 18)
      text = font.render(f"{i}", True, (0, 0, 0))
      text_surface.blit(text, (10, (NUMBER_OF_FLOORS - i - 1) * floor_height + 10))
      num_people_waiting = len(self.buffer_floor[i])
      text = font.render(f"Waiting: {num_people_waiting}", True, (0, 0, 0))
      text_surface.blit(text, (10, (NUMBER_OF_FLOORS - i) * floor_height - 24))

      if num_people_waiting > 0:
        stick_figure_pos = (self.screen_width // 2 - lift_width, (NUMBER_OF_FLOORS - i) * floor_height - 40)
        draw_stickman(stick_figure_pos)

    # Draw the bottom line
    pygame.draw.line(lift_surface,
                     color=(0, 0, 0),
                     start_pos=(0, NUMBER_OF_FLOORS * floor_height),
                     end_pos=(self.screen_width, NUMBER_OF_FLOORS * floor_height),
                     width=5)

    # Draw lift shaft
    pygame.draw.line(lift_surface,
                     color=(0, 0, 0),
                     start_pos=((self.screen_width - lift_width) // 2, 0),
                     end_pos=((self.screen_width - lift_width) // 2, self.screen_height - info_height),
                     width=5)
    pygame.draw.line(lift_surface,
                     color=(0, 0, 0),
                     start_pos=((self.screen_width + lift_width) // 2, 0),
                     end_pos=((self.screen_width + lift_width) // 2, self.screen_height - info_height),
                     width=5)

    # Draw cabin
    dist = 5

    # If doors are closed add grey background
    if door_state == DOOR_CLOSED:
      pygame.draw.rect(lift_surface,
                       color=(150, 150, 150),
                       rect=((self.screen_width - lift_width) // 2 + dist,
                             (NUMBER_OF_FLOORS - current_floor - 1) * floor_height + dist,
                             lift_width - dist * 2,
                             floor_height - dist * 2),
                       width=0)
      pygame.draw.line(lift_surface,
                       color=(0, 0, 0),
                       start_pos=(self.screen_width // 2, (NUMBER_OF_FLOORS - current_floor - 1) * floor_height + dist),
                       end_pos=(self.screen_width // 2, (NUMBER_OF_FLOORS - current_floor) * floor_height - dist - 2),
                       width=3)

    pygame.draw.rect(lift_surface, color=(0, 0, 0),
                     rect=((self.screen_width - lift_width) // 2 + dist,
                           (NUMBER_OF_FLOORS - current_floor - 1) * floor_height + dist,
                           lift_width - dist * 2,
                           floor_height - dist * 2),
                     width=5)

    # Draw people in the cabin
    if len(self.buffer_cabin) > 0:
      stick_figure_pos = (self.screen_width // 2 - 10, (NUMBER_OF_FLOORS - current_floor) * floor_height - 40)
      draw_stickman(stick_figure_pos)

    # Draw info
    font = pygame.font.SysFont('Arial', 18)

    num_people_in_cabin = len(self.buffer_cabin)

    text = font.render(f"People in cabin: {num_people_in_cabin}", True, (0, 0, 0))
    text_surface.blit(text, (10, self.screen_height - info_height + 10))

    pressed_buttons = [index for index, value in enumerate(cabin_buttons) if value]
    text = font.render(f"Cabin buttons: {', '.join(map(str, pressed_buttons))}", True, (0, 0, 0))
    text_surface.blit(text, (10, self.screen_height - info_height + 40))

    pressed_buttons = [index for index, value in enumerate(call_buttons) if value]
    text = font.render(f"Call buttons: {', '.join(map(str, pressed_buttons))}", True, (0, 0, 0))
    text_surface.blit(text, (10, self.screen_height - info_height + 70))

    text = font.render(f"Moving direction: {move_direction}", True, (0, 0, 0))
    text_surface.blit(text, (10, self.screen_height - info_height + 100))

    self.screen.fill((255, 255, 255))
    self.screen.blit(lift_surface, (0, 0))
    self.screen.blit(text_surface, (0, 0))
    self.screen.blit(figure_surface, (0, 0))

    # Save the frame
    if self.frames_dir is not None:
      self.frames_dir.mkdir(exist_ok=True, parents=True)
      frame_filename = self.frames_dir / f"frame_{self.frame_count:03d}.png"

      pygame.image.save(self.screen, frame_filename)

    self.frame_count += 1

    if self.render_mode == "human":
      pygame.event.pump()
      self.clock.tick(self.metadata["render_fps"])
      pygame.display.flip()
      return None

    elif self.render_mode == "rgb_array":
      return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    return None

  def close(self):
    """
    Close the environment, including the Pygame window (if any).
    Creates a gif if the frames directory is specified and render outputs have been stored.
    """

    # There was no screen, hence nothing to process
    if self.screen is None:
      return

    # A directory for the frames was specified, hence fetch the stored frames and create a gif
    if self.frames_dir is not None:
      import imageio
      gif_filename = self.frames_dir / 'animation.gif'
      frames = [imageio.imread(frame) for frame in sorted(self.frames_dir.glob("frame_*.png"))]
      imageio.mimsave(gif_filename, frames, 'GIF', fps=self.metadata["render_fps"], loop=True)

    import pygame
    pygame.display.quit()
    pygame.quit()

    self.screen = None
    self.clock = None

    return

  def _new_persons(self):
    """
    Generate new persons with random start and destination floors.

    Regarding the settings for the binomial distribution:
    * With n=1, one obtains a realistic behaviour.
    * With n=2, one obtains significantly more people, but the training should be easier since more stuff happens.

    Returns
    -------
    spawned : bool
      Indicates whether new persons have been spawned.
    """

    # N rounds of a pick-and-replace random event
    # The resulting matrix indicates, at which floors new persons with destinations are waiting
    person_locations = np.random.binomial(1, PASSENGER_DISTRIBUTION)

    # Not a single person has been created, hence there is not a single non-zero element
    if not person_locations.any():
      return False

    # Get the indices of the non-zero elements, which represent the start and destination floors
    non_zero_indices = np.where(person_locations != 0)

    # Unpack the indices and spawn people
    for start_floor, dest_floor in zip(*non_zero_indices):
      self.buffer_floor[start_floor].append(Person(start_floor, dest_floor))
      self.person_counter += 1

    return True

  def _update_call_buttons(self, call_buttons):
    """
    The call button is active on every floor, where people are waiting.

    Parameters
    ----------
    call_buttons : list
      A list of booleans indicating which buttons are pressed on the floors.
    """

    for floor, people in self.buffer_floor.items():
      if people:
        call_buttons[floor] = True
    return

  def _update_cabin_buttons(self, cabin_buttons):
    """
    The cabin button is active for every destination of a person in the cabin.

    Parameters
    ----------
    cabin_buttons : list
      A list of booleans indicating which buttons are pressed in the cabin.
    """

    for person in self.buffer_cabin:
      cabin_buttons[person.destination] = True
    return

  def _move_in_cabin(self, current_floor):
    """
    Move people from a floor buffer to the cabin buffer.
    The number of people transferred is limited by the max_capacity and the number of people already in the cabin.

    Parameters
    ----------
    current_floor : int
      The current floor of the lift.

    Returns
    -------
    transferred : int
      The number of people transferred from the floor buffer to the cabin buffer.
    """

    counter = 0

    while len(self.buffer_cabin) < self.max_capacity and self.buffer_floor[current_floor]:
      p = self.buffer_floor[current_floor].pop(0)
      self.buffer_cabin.append(p)
      counter += 1

    return counter

  def get_active_persons(self):
    """
    Get the number of active persons in the lift.

    Returns
    -------
    active_persons : int
      The number of active persons in the lift.
    """
    return len(self.buffer_cabin) + sum(map(len, self.buffer_floor.values()))

  @staticmethod
  def get_available_actions(state):
    """
    Get the available actions for the given state.

    Parameters
    ----------
    state : tuple
      A state of the environment.

    Returns
    -------
    actions : list
      A list of available actions for the given state.
    """

    # Unpack the state for easy access
    current_floor, move_direction, door_state, cabin_buttons, call_buttons = state

    # If the door is open, the only available action is to close the door
    if door_state == DOOR_OPEN:
      return [ACTION_DOOR]

    # The lift has stopped: open the door or start moving again
    if move_direction == DIRECTION_NONE:
      actions = [ACTION_NOOP, ACTION_DOOR]

      if current_floor < NUMBER_OF_FLOORS - 1:
        actions.append(ACTION_UP)

      if current_floor > 0:
        actions.append(ACTION_DOWN)

      return actions

    # The lift is moving: stop at the next floor or keep going
    return [ACTION_NOOP, ACTION_STOP]
