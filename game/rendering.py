import numpy as np
import pyglet

from game import game

def draw_command_labels(batch, window_width, window_height):
  return pyglet.text.Label('Pass (p) | Reset (r) | Exit (e)',
                    font_name='Helvetica',
                    font_size=11,
                    x=20, y=window_height - 20, anchor_y='top', batch=batch, multiline=True, width=window_width)

def draw_info(batch, window_width, window_height, upper_grid_coord, go, turn):
  turn_str = 'B' if turn == game.BLACK else 'W'
  prev_player_passed = go.prev_player_passed()
  game_ended = go.game_ended()
  info_label = "Turn: {}\nPassed: {}\nGame: {}".format(turn_str, prev_player_passed,
                                                        "OVER" if game_ended else "ONGOING")

  l1 = pyglet.text.Label(info_label, font_name='Helvetica', font_size=11, x=window_width - 20, y=window_height - 20,
                    anchor_x='right', anchor_y='top', color=(0, 0, 0, 192), batch=batch, width=window_width / 2,
                    align='right', multiline=True)
  return l1 

def draw_title(batch, window_width, window_height):
  return pyglet.text.Label("Go", font_name='Helvetica', font_size=20, bold=True, x=window_width / 2, y=window_height - 20,
                      anchor_x='center', anchor_y='top', color=(0, 0, 0, 255), batch=batch, width=window_width / 2,
                      align='center')

def draw_grid(batch, delta, board_size, lower_grid_coord, upper_grid_coord):
  shapes = []
  label_offset = 20
  coord = lower_grid_coord
  for i in range(board_size):
    # horizontal
    h = pyglet.shapes.Line(lower_grid_coord, coord,
                        upper_grid_coord, coord, 
                        color=(0,0,0),
                        width=3, 
                        batch=batch
                      )
    # vertical
    v = pyglet.shapes.Line(coord, lower_grid_coord,
                        coord, upper_grid_coord,
                        color=(0,0,0),
                        width=3,
                        batch=batch
                        )

    l = pyglet.text.Label(str(i),
                        font_name='Couriee', font_size=11,
                        x=lower_grid_coord - label_offset, y=coord,
                        anchor_x='center', anchor_y='center',
                        color=(0, 0, 0, 255), batch=batch)
    # label on the bottom
    l2 = pyglet.text.Label(str(i),
                        font_name='Courier', font_size=11,
                        x=coord, y=lower_grid_coord - label_offset,
                        anchor_x='center', anchor_y='center',
                        color=(0, 0, 0, 255), batch=batch)
    coord += delta
    shapes.extend([h,v,l,l2])
  return shapes

def draw_pieces(batch, coord, delta, piece_r, size, state):
  ps = []
  for i in range(size):
    for j in range(size):
      # black piece
      if state[i, j] == game.BLACK:
        s = pyglet.shapes.Circle(coord+i*delta,coord+j*delta,piece_r,color=(0,5,2),batch=batch)
        ps.append(s)

      # white piece
      elif state[i, j] == game.WHITE:
        s = pyglet.shapes.Circle(coord+i*delta,coord+j*delta,piece_r,color=(255,255,255),batch=batch)
        ps.append(s)
  return ps

def draw_areas(batch, coord, delta, black_area, white_area):
  def draw_rect(areas, color):
    ss = []
    for pos in areas:
      i,j = pos 
      side_length = 20
      half_side = side_length / 2
      x = coord + i * delta - half_side
      y = coord + j * delta - half_side
      s = pyglet.shapes.Rectangle(x, y, side_length, side_length, color=color, batch=batch)
      ss.append(s)
    return ss
  bs = []
  a = draw_rect(black_area,(0,0,0))
  b = draw_rect(white_area, (255,255,255))
  bs.extend(a+b)
  return bs


      