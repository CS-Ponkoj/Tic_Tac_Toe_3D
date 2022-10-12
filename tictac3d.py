"""
Basic 3D Tic Tac Toe with Minimax and Alpha-Beta pruning, using a simple
heuristic to check for possible winning moves or blocking moves if no better
alternative exists.
"""


from curses import isendwin
from shutil import move

from colorama import Back, Style, Fore
import numpy as np

inf = 9999999999
class TicTacToe3D(object):
    """3D TTT logic and underlying game state object.

    Attributes:
        board (np.ndarray)3D array for board state.
        difficulty (int): Plyand number of moves to look ahead.
        depth_count (int): Used in conjunction with ply to control depth.

    Args:
        player (str): Player that makes the first move.
        player_1 (Optional[str]): player_1's character.
        player_2 (Optional[str]): player_2's character.
        ply (Optional[int]): Number of moves to look ahead.
    """

   

    def __init__(self, board = None, player=-1, player_1=-1, player_2=1, ply=3):
        if board is not None:
            assert type(board) == np.ndarray, "Board must be a numpy array"
            assert board.shape == (3,3,3), "Board must be 3x3x3"
            self.np_board = board
        else:
            self.np_board = self.create_board()
        self.map_seq_to_ind, self.map_ind_to_seq = self.create_map()
        self.allowed_moves = list(range(pow(3, 3)))
        self.difficulty = ply
        self.depth_count = 0
        self.player = player
        if player == player_1:
            self.player_1_turn = True
        else:
            self.player_1_turn = False
        self.player_1 = player_1
        self.player_2 = player_2
        self.players = (self.player_1, self.player_2)

    def create_map(self):
        """Create a mapping between index of 3D array and list of sequence, and vice-versa.

        Args: None

        Returns:
            map_seq_to_ind (dict): Mapping between sequence and index.
            map_ind_to_seq (dict): Mapping between index and sequence.
        """
        self.maxDepth = 5
       
        a = np.hstack((np.zeros(9),np.ones(9),np.ones(9)*2))
        a = a.astype(int)
        b = np.hstack((np.zeros(3),np.ones(3),np.ones(3)*2))
        b = np.hstack((b,b,b))
        b = b.astype(int)
        c = np.array([0,1,2],dtype=int)
        c = np.tile(c,9)
        mat = np.transpose(np.vstack((a,b,c)))
        ind = np.linspace(0,26,27).astype(int)
        map_seq_to_ind = {}
        map_ind_to_seq = {}
        for i in ind:
            map_seq_to_ind[i] = (mat[i][0],mat[i][1],mat[i][2])
            map_ind_to_seq[(mat[i][0],mat[i][1],mat[i][2])] = i
        return map_seq_to_ind, map_ind_to_seq
    
    def reset(self):
        """Reset the game state."""
        self.allowed_moves = list(range(pow(3, 3)))
        self.np_board = self.create_board()
        self.depth_count = 0


    @staticmethod
    def create_board():
        """Create the board with appropriate positions and the like

        Returns:
            np_board (numpy.ndarray):3D array with zeros for each position.
        """
        np_board = np.zeros((3,3,3), dtype=int)
        return np_board
    
    def isComplete(self):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if self.np_board[i][j][k] == 0:
                        return False
        
        return True
    
    def isWin(self, player):
        # To check 2D Diagonal lines
        for i in range(3):
            if self.np_board[0][0][i] == player and self.np_board[1][1][i] == player and self.np_board[2][2][i] == player:
                return True

            if self.np_board[2][0][i] == player and self.np_board[1][1][i] == player and self.np_board[0][2][i] == player:
                return True

            if self.np_board[0][i][0] == player and self.np_board[1][i][1] == player and self.np_board[2][i][2] == player:
                return True
            if self.np_board[2][i][0] == player and self.np_board[1][i][1] == player and self.np_board[0][i][2] == player:
                return True

            if self.np_board[i][0][0] == player and self.np_board[i][1][1] == player and self.np_board[i][2][2] == player:
                return True
                
            if self.np_board[i][2][0] == player and self.np_board[i][1][1] == player and self.np_board[i][0][2] == player:
                return True

        
        # To check diagonal lines of the cube
        if self.np_board[0][0][0] == player and self.np_board[1][1][1] == player and self.np_board[2][2][2] == player:
            return True

        if self.np_board[0][0][2] == player and self.np_board[1][1][1] == player and self.np_board[2][2][0] == player:
           return True

        if self.np_board[2][0][0] == player and self.np_board[1][1][1] == player and self.np_board[0][2][2] == player:
           return True

        if self.np_board[2][0][2] == player and self.np_board[1][1][1] == player and self.np_board[0][2][0] == player:
           return True


        # To check straight lines
        for i in range(3):
            for j in range(3):
                temp = 1
                for k in range(3):
    	            if self.np_board[i][j][k] != player:
                         temp = 0
                if temp:
                    return True

            temp = 1
            for k in range(3):
    	        if self.np_board[i][k][j] != player:
                     temp = 0
            if temp:
                return True

            
            temp = 1
            for k in range(3):
    	        if self.np_board[k][i][j] != player:
                     temp = 0
            if temp:
                return True
        
        return False

    
    def minimax(self, depth, turn, alpha=-inf, beta=inf):
        
        if self.isWin(1):
            return self.maxDepth + 1 - depth
        if self.isWin(-1):
            return depth-self.maxDepth-1
        if self.isComplete():
            return 0
        
        if depth >= self.maxDepth:
            return 0
        
        best_value = 0
        val = 0 

        if turn == 1:
            best_value = 10000000000
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        if self.np_board[i][j][k] == 0:
                            self.np_board[i][j][k] = 1
                            val = self.minimax(depth=depth+1, turn=-1, alpha=alpha, beta=beta)
                            self.np_board[i][j][k] = 0

                            best_value = min(best_value, val)
                            beta = min(beta, best_value)

                            if beta <= alpha:
                                return best_value
            
            
            return best_value
        
        else:
            best_value = -10000000000
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        if self.np_board[i][j][k] == 0:
                            self.np_board[i][j][k] = -1
                            val = self.minimax(depth=depth+1, turn=1, alpha=alpha, beta=beta)
                            self.np_board[i][j][k] = 0

                            best_value = max(best_value, val)
                            alpha = max(alpha, best_value)

                            if beta <= alpha:
                                return best_value
            
            return best_value



    def play_game(self):
        """Primary game loop.
        
        Until the game is complete we will alternate between computer and
        player turns while printing the current game state.
        """
        while not self.isComplete():
            a = [-1,-1,-1]
            val = -inf
            if self.player_1_turn:
                curr_player = -1
                opp_player = 1
            else:
                curr_player = 1
                opp_player = -1
            for i in  range(3):
                for j in range(3):
                    for k in range(3):
                        if self.np_board[i][j][k] == 0:
                            self.np_board[i][j][k] = curr_player
                            t = self.minimax(0, opp_player, -inf, inf)
                            if t > val:
                                val = t
                                a[0] = i
                                a[1] = j
                                a[2] = k
                            self.np_board[i][j][k] = 0
            
            self.np_board[a[0]][a[1]][a[2]] = curr_player
                
        try:
            if self.isWin(-1):
                final_winner = -1
            elif self.isWin(1):
                final_winner = 1
            else:
                final_winner = 0
            
            return self.np_board, final_winner
        except KeyboardInterrupt:
            print('\n ctrl-c detected, exiting')




if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '--player', dest='player', help='Player that plays first, 1 or -1',\
                type=int, default=-1, choices=[1,-1]
    )
    parser.add_argument(
        '--ply', dest='ply', help='Number of moves to look ahead', \
                type=int, default=6
    )
    args = parser.parse_args()
    brd,winner = TicTacToe3D(player=args.player, ply=args.ply).play_game()
    print("final board: \n{}".format(brd))
    print("winner: player {}".format(winner))
