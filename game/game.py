import numpy as np
import time
import copy
from typing import *

BLACK = 1
WHITE = 2

def neighbors(board, move): 
  ret = set()
  for i,j in [(1,0),(-1,0),(0,1),(0,-1)]: 
    x,y = move
    if (x+i >= 0) and (y+j >= 0) and (x+i < len(board)) and (y+j < len(board)):
      ret.add((x+i,y+j))
  return ret

class Go:
  def __init__(self, size):
    self.board = np.zeros((size,size),dtype=np.int8)
    self.size = size
    self.turn = BLACK
    self.history = []
  
  def move(self, move: Tuple[int, int], simulate=False):
    if not simulate and not self.is_legal_move(move): 
      return False, []
    board = np.copy(self.board)
    if move == (-1,-1):
      if not simulate:
        self.history.append((self.turn, board, board, move, []))
        self.turn = self.turn % 2 + 1
      return True, []
    self.board[move] = self.turn
    dead = self.remove_dead(move)
    self.history.append((self.turn, board, np.copy(self.board), move, dead))
    if not simulate:
      self.turn = self.turn % 2 + 1
    return True, dead
  
  def get_legal_moves(self):
    cand = np.argwhere(self.board == 0)
    ret = []
    for arr in cand:
      move = (arr[0],arr[1])
      if self.is_legal_move(move):
        ret.append(move)
    return ret + [(-1,-1)] 
  
  def is_legal_move(self, move):
    if move == (-1,-1):
      return True
    if self.board[move] != 0:
      return False
    # check repeat violation 
    ret = True
    if len(self.history) > 6:
      dead = self.history[-1][-1]
      if len(dead)==1:
        oldboard = self.history[-2][2]
        self.move(move, simulate=True)
        if np.array_equal(self.board, oldboard):
          ret = False
        self.board = self.history[-1][1]
        self.history = self.history[:-1]

    if not ret: return False
    # check suicide
    ns = neighbors(self.board, move)
    ns_vals = self.board[[i for i,_ in ns],[j for _,j in ns]].tolist()
    if 0 not in ns_vals:
      self.move(move, simulate=True)
      if len(self.count_liberties(move)[1]) == 0:
        ret = False
      oldboard = self.history[-1][1]
      self.board = oldboard
      self.history = self.history[:-1]

    return ret

  def remove_dead(self, move):
    opp = WHITE if self.turn==BLACK else BLACK
    ns = neighbors(self.board, move)
    enemies = list() 
    for n in ns:
      if self.board[n]==opp: 
        enemies.append(n)
    dead = set()
    while len(enemies) > 0:
      seen,liberties = self.count_liberties(enemies.pop())
      if len(liberties) == 0:
        self.board[[i for i,_ in seen], [j for _,j in seen]] = 0
        dead.update(seen)
      enemies = list(set(enemies)-seen)
    return dead
      
  def count_liberties(self, stone):
    player = self.board[stone]
    liberties = set()
    seen,queue = set(),list()
    seen.add(stone)
    queue.append(stone)
    while len(queue) > 0:
      ns = neighbors(self.board, queue.pop())
      for n in ns:
        if n not in seen:
          if self.board[n]==0: 
            liberties.add(n)
          elif self.board[n]==player: 
            seen.add(n)
            queue.append(n)
    return seen, liberties
  
  def prev_player_passed(self):
    if len(self.history) > 0: 
      if self.history[-1][-2] == (-1,-1): return True
    return False
  
  def game_ended(self):
    if len(self.history) > 1:
      m1,m2 = self.history[-1][-2],self.history[-2][-2]
      if m1 == m2 and m1 == (-1,-1): return True
    return False
  
  # TODO:in empty board, all board is counted as area for both players
  def score(self):
    if len(self.history) < 3:
      return None, None, None

    def get_area(player):
      opp = WHITE if player == BLACK else BLACK
      possible = {(pos[0],pos[1]) for pos in np.argwhere(self.board == 0)}
      area = copy.deepcopy(possible)
      while len(possible) > 0:
        mv = possible.pop()
        ns = neighbors(self.board, mv)
        if opp in self.board[[i for i,j in ns],[j for i,j in ns]].tolist():
          # exhaustively find entire chain, and discard
          queue = [n for n in ns if self.board[n] == 0]
          notarea = set(queue + [mv])
          while len(queue) > 0:
            ns = neighbors(self.board, queue.pop())
            zeros = [n for n in ns if self.board[n] == 0]
            for n in zeros:
              if n not in notarea:
                queue.append(n)
            notarea.update(set(zeros))
          possible -= notarea
          area -= notarea
      return area
    
    black, white = get_area(BLACK), get_area(WHITE)

    if len(black) > len(white):
      return BLACK, black, white
    elif len(white) > len(black):
      return WHITE, black, white
    else:
      return 0, black, white

  