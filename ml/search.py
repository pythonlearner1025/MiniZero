from game.game import Go
from ml.proc import features
import mlx.core as mx
import numpy as np
import math

def idx2move(idx,size):
  if idx == size*size:
    return -1,-1
  j = idx%size
  i = (idx-j)//size
  return i,j

def move2idx(move,size):
  if move == (-1,-1): return size**2
  return int(move[0]*size+move[1]) 

class MCTS:
  def __init__(self):
    self.P = dict()
    self.Q = dict()
    self.N = dict()
    self.visited = set()

  def search(self, game: Go, net, cpuct=0.3, root=False):
    if game.game_ended():
      winner, _, _ = game.score()
      if game.turn == winner:
        return -1 
      elif winner == 0:
        return 0
      else:
        return 1

    s = game.board.tobytes()
    # implementing dirichlet in mlx will be 5x speedup
    if s not in self.visited:
      self.visited.add(s)
      x = features(game)
      x = mx.reshape(x,[1,*x.shape], stream=mx.cpu)
      p,v = net(x)
      self.P[s] = np.array(p.squeeze())
      v = v.item()
      if root:
        self.P[s] = 0.75*self.P[s]+0.25*np.random.dirichlet([0.3]*self.P[s].shape[-1])
      self.Q[s] = np.zeros(len(self.P[s]))
      self.N[s] = np.zeros(len(self.P[s]),dtype=np.int32)
      return -v

    best_u, best_mv = float('-inf'), None
    for move in game.get_legal_moves():
      a = move2idx(move,game.size) 
      u = self.Q[s][a]+cpuct*self.P[s][a]*(math.sqrt(self.N[s].sum()-self.N[s][a])/(1+self.N[s][a]))
      if u > best_u:
        best_u = u
        best_mv = move,a
    
    best_move,a = best_mv
    #st = time.perf_counter()
    game.move(best_move)
    #e = time.perf_counter()
    #print(f'inference: {(e-st)*1000:7.3f} ms')
    v = self.search(game, net)
    # count actions that result in ties as losses
    av = -1 if v==0 else v
    self.Q[s][a] = (self.N[s][a]*self.Q[s][a]+av)/(self.N[s][a]+1)
    self.N[s][a] += 1
    return -v 
  
  # TODO add temp
  def pi(self, game: Go, t=1):

    def softmax(x):
      shift_x = x-np.max(x)
      exps = np.exp(shift_x)
      return exps / np.sum(exps)

    s = game.board.tobytes()
    legal_moves = game.get_legal_moves()
    legals = np.zeros(len(legal_moves))
    for k,move in enumerate(legal_moves):
      a = move2idx(move,game.size)
      if self.N[s].sum()-self.N[s][a] > 0:
        legals[k] = self.N[s][a]**(1/t) / (self.N[s].sum()-self.N[s][a])**(1/t)
      else:
        legals[k] = 1

    legals = softmax(legals)    
    pi = np.zeros(len(self.N[s]))
    for k,move in enumerate(legal_moves):
      a = move2idx(move,game.size)
      pi[a] = legals[k]
    return pi




