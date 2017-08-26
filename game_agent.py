"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

from copy import deepcopy


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass
    
    """
    The following Parameters and Returns are same for all the heuristic functions
    
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    
def custom_score(game, player):
    """When the two players have same number of moves, pick the move closest to the center of the board; 
       Otherwise we use the conservative evaluation approach with a weight of 1.5 for the player's number of moves.
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    # When both players have same number of moves
    # Pick the move that is closest to the center of the board
    if own_moves == opp_moves:
        w, h = game.width / 2., game.height / 2.
        y, x = game.get_player_location(player)
        return float((h - y)**2 + (w - x)**2)
    # if the players have different number of moves, pick the move that have more moves than opponent.
    else:
        return float(1.5*own_moves - opp_moves)


def custom_score_2(game, player):
    """Calculate the square of player's numer of moves minus the square of opponent's number of moves.
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
 
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves^2 - opp_moves^2)

def custom_score_3(game, player):
    """This is a conservitave heuristic that calculate 1.5 times the player's move minus opponent's move.
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
 
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(1.5*own_moves - opp_moves)

def open_improved(game, player):
    """ Calculated the combination of open heuristic (i.e. number of player's move) 
    plus the improved heuristic (i.e. number of player's move monus number of opponent's move).
    It also indicates this is a conservative player that try to get the max of his/her moves.
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
 
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(2*own_moves - opp_moves)

def random_improved(game, player):
    """This is an improved heuristic with random factor added to the function 
    that we first create a random number between (-1,1). Then the evaluation is 
    (1+random number) times player's number of moves minus (1-random number) times opponent's number of moves. 
    
    Here we noted that both (1+random number) and (1-randeom number) are in range (1,2) 
    and the sum of these two numbers equals to 2.
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    rand = random.randint(-1,1)
    return float( (1+rand)*own_moves - (1-rand)*opp_moves)

def aggressive(game, player):
    """This is a aggressive heuristic that calculate the player's move minus 1.5 times opponent's move.
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
 
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 1.5*opp_moves)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

         For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm.
        
        This is a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        Helper functions
        ----------
        min_value(self, game, depth):
            Return the value if no child nodes or no moves, otherwise return the minimum value over all legal child nodes.
           
        max_value(self, game, depth):
            Return the value if no child nodes or no moves, otherwise return the maximum value over all legal child nodes.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        """
        # Check the time left for calculation - always add time check to each helper functions 
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        # Check if there is any legal moves
        legal_moves = game.get_legal_moves()        
        # No legal move scenario
        if not legal_moves or depth <= 0:
            return (-1, -1)
           
        # Calculate best move scenario
        # Algorithm:
        # assign default score/move -> loop through all moves -> 
        # (ctn) -> evaluate scores from child nodes using helper function -> update best score/move
        best_score = float("-inf")
        best_move = None
        for m in legal_moves:
            v = self.min_value(game.forecast_move(m), depth - 1)
            if v > best_score:
                best_score = v
                best_move = m
        return best_move       
    
    # Helper Funtions (No.1)
    def min_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        if depth <= 0 or not game.get_legal_moves():
            return self.score(game, self)

        v = float("inf")
        for move in game.get_legal_moves() :
            v = min(v, self.max_value(game.forecast_move(move), depth - 1))
        return v

    # Helper Funtions (No.2)
    def max_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        if depth <= 0 or not game.get_legal_moves() :
            return self.score(game, self)
        
        v = float("-inf")
        for move in game.get_legal_moves() :
            v = max(v, self.min_value(game.forecast_move(move), depth - 1))
        return v     

        

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. The player must return a good move before 
    the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        search_depth = 1
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            while True:          
                best_move = self.alphabeta(game, search_depth)
                search_depth += 1
                
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning.

        This is a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        Helper functions
        ----------
        min_value_ab(self, game, depth, alpha, beta):
            Return the value if no child nodes or no moves, otherwise return the minimum value over limited legal child nodes.
           
        max_value_ab(self, game, depth, alpha, beta):
            Return the value if no child nodes or no moves, otherwise return the maximum value over limited legal child nodes.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        """
        # Check the time left for calculation - always add time check to each helper functions 
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        # Check if there is any legal moves
        legal_moves = game.get_legal_moves()        
        # No legal move scenario
        if not legal_moves or depth <= 0:
            return (-1, -1)
           
        # Calculate best move scenario
        # Algorithm:
        # assign default score/move -> loop through all moves -> 
        # (ctn) -> evaluate scores from child nodes using helper function, using alpha and beta to limit visited nodes 
        # (ctn) -> update best score/move
        best_score = float("-inf")
        best_move = None
        for m in legal_moves:
            v = self.min_value_ab(game.forecast_move(m), depth-1, alpha, beta)   
            if v > best_score:
                best_score = v
                best_move = m            
            alpha = max(alpha, best_score)
        return best_move       
    
    # Helper Funtions (No.1)
    def min_value_ab(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        if depth <= 0 or not game.get_legal_moves():
            return self.score(game, self)

        v = float("inf")
        for move in game.get_legal_moves():
            v = min(v, self.max_value_ab(game.forecast_move(move), depth - 1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    # Helper Funtions (No.2)
    def max_value_ab(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        if depth <= 0 or not game.get_legal_moves() :
            return self.score(game, self)
        
        v = float("-inf")
        for move in game.get_legal_moves():
            v = max(v, self.min_value_ab(game.forecast_move(move), depth - 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v  
 
