import pyglet
import time
from pyglet.window import mouse
from pyglet.window import key

from game import rendering, game 

class Play:
  def __init__(self, size, start):
    self.go = game.Go(size)
    self.size = size
    screen = pyglet.canvas.get_display().get_default_screen()
    window_width = int(min(screen.width, screen.height) * 2 / 3)
    window_height = int(window_width * 1.2)
    self.window = pyglet.window.Window(window_width, window_height) 
    self.black_area = []
    self.white_area = []

  def render(self):
    screen = pyglet.canvas.get_display().get_default_screen()
    window_width = int(min(screen.width, screen.height) * 2 / 3)
    window_height = int(window_width * 1.2)
    window = self.window     

    # Set Cursor
    cursor = window.get_system_mouse_cursor(window.CURSOR_CROSSHAIR)
    window.set_mouse_cursor(cursor)

    # Outlines
    lower_grid_coord = window_width * 0.075
    board_size = window_width * 0.85
    upper_grid_coord = board_size + lower_grid_coord
    delta = board_size / (self.size - 1)
    piece_r = delta / 3.3  # radius
    batch = pyglet.graphics.Batch()
    fps_display = pyglet.window.FPSDisplay(window=window)

    @window.event
    def on_draw():
      window.clear()
      pyglet.gl.glClearColor(0.7, 0.5, 0.3, 1)
      a = rendering.draw_grid(batch, delta, self.size, lower_grid_coord, upper_grid_coord)
      # info on top of the board
      b = rendering.draw_info(batch, window_width, window_height, upper_grid_coord, self.go, self.go.turn)
      # Inform user what they can do
      c = rendering.draw_command_labels(batch, window_width, window_height)
      d = rendering.draw_title(batch, window_width, window_height)
      e = rendering.draw_pieces(batch, lower_grid_coord, delta, piece_r, self.size, self.go.board)
      f = rendering.draw_areas(batch, lower_grid_coord, delta, self.black_area, self.white_area)
      fps_display.draw()
      batch.draw()

    @window.event
    def on_mouse_press(x, y, button, modifiers):
      if button == mouse.LEFT:
        grid_x = (x - lower_grid_coord)
        grid_y = (y - lower_grid_coord)
        x_coord = round(grid_x / delta)
        y_coord = round(grid_y / delta)
        try:
          move = (x_coord, y_coord)
          self.go.move(move)

        except Exception as e:
          print(e)
          pass

    @window.event
    def on_key_press(symbol, modifiers):
      if symbol == key.P:
        print('p detected')
        move = (-1,-1)
        self.go.move(move)
        print(self.go.history[-1])
        if self.go.game_ended():
          print('game ended')
          winner,black,white = self.go.score()
          self.black_area = black
          self.white_area = white

      elif symbol == key.R:
        self.go = game.Go(self.size)
        self.black_area = []
        self.white_area = []
      elif symbol == key.E:
        self.window.close()
        pyglet.app.exit()
        self.user_action = -1

    pyglet.app.run()