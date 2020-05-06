import numpy as np
import pygame
import sys
import math
import random
import copy

from os.path import join

from pygame.image import load

from copy import deepcopy

ROW_COUNT = 6 
COL_COUNT = 7 


WHITE =(255,255,255)
BLUE = (0,0,255)
LIGHTBLUE = (204,229,255)

SQUARESIZE = 100
RADIUS = int(SQUARESIZE/2 - 5)

width = COL_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

#for defining turn
MINIMAX = "minimax"
MCTS = "mcts"

#for defining player tiles
RED = -1 # minimax
BLACK = 1 # mcts

EMPTY = 0
startTime = 0

HIGHVALUE = 100000000000000



def main():
    global FPSCLOCK, SCREEN, REDPILERECT, BLACKPILERECT, REDTOKENIMG, myfont
    global BLACKTOKENIMG, BOARDIMG, HUMANWINNERIMG
    global COMPUTERWINNERIMG, WINNERRECT, TIEWINNERIMG
    global MCTSWINNER, MINIMAXWINNER
    global winnerCountMinimax,winnerCountMcts, tieCount


    pygame.init()
    SCREEN = pygame.display.set_mode((width, height))
    SCREEN.fill(LIGHTBLUE)
    FPSCLOCK = pygame.time.Clock()
    pygame.display.set_caption('Connect Four')
    myfont = pygame.font.SysFont("monospace", 75)
    BOARDIMG = pygame.image.load(join("img","board.png"))
    REDTOKENIMG = pygame.image.load(join("img","redTile.png"))
    BLACKTOKENIMG = pygame.image.load(join("img","blackTile.png"))
    HUMANWINNERIMG = pygame.image.load(join("img","humanwinner.png"))
    COMPUTERWINNERIMG = pygame.image.load(join("img","computerwinner.png"))
    MCTSWINNER = pygame.image.load(join("img","MCTSwinner.png"))
    MINIMAXWINNER = pygame.image.load(join("img","minimaxWinner.png"))
    TIEWINNERIMG = pygame.image.load(join("img","tie.png"))
    WINNERRECT = HUMANWINNERIMG.get_rect()
    WINNERRECT.center = (int(width / 2), int(height / 2))

    myfont = pygame.font.SysFont("monospace", 75)

    isFirstGame = True

    winnerCountMinimax = 0
    winnerCountMcts = 0
    tieCount = 0
    countGame = 0

    while True:
        countGame += 1
        winnerCountMinimax , winnerCountMcts, tieCount = runGame(isFirstGame,winnerCountMinimax,winnerCountMcts,tieCount)
        isFirstGame = False
        print("Total game: " + str(countGame))
        print("Count Minimax Winning : " + str(winnerCountMinimax))
        print("Count MCTS Winning : " + str(winnerCountMcts))
        print("Count Ties : " + str(tieCount))



def runGame(isFirstGame,winnerCountMinimax,winnerCountMcts,tieCount):

    if isFirstGame:
        turn = MCTS
        print("HAVE FUN !!")
    else:
        # first player is chose randomly
        if random.randint(0, 1) == 0:
            turn = MCTS
        else:
            turn = MINIMAX

    #creates board object
    board = Board()
    # prints board with all zeros
    board.printBoard(board.board)
    board.drawBoard(board.board)
    pygame.display.update()
    display = True

    # Main Game Loop
    while True:

        if turn == MINIMAX: # minimax player turn
            getMinimaxMove(board)
            if board.isWinner(board.board,RED):
                winnerCountMinimax += 1
                winnerImg = MINIMAXWINNER
                print("Minimax-alphabeta WINNER!!")
                break
            turn = MCTS # switches to the other player

        else:
            getMonteMove(board) # MCTS player turn
            if board.isWinner(board.board,BLACK):
                winnerCountMcts += 1
                winnerImg = MCTSWINNER
                print("MonteCarlo WINNER!!")
                break
            turn = MINIMAX # switches to the other player

        # checks if board is completed if it is, that means it is tie
        if board.isBoardFull(board.board):
            tieCount += 1
            winnerImg = TIEWINNERIMG

    # displays the winner for a while and repeats the game until player quits
    while display:
        board.drawBoard(board.board)
        SCREEN.blit(winnerImg, WINNERRECT)
        pygame.display.update()
        pygame.time.wait(2500)
        display = False

    return winnerCountMinimax, winnerCountMcts, tieCount
        


def getMinimaxMove(board):
    '''
        This function calls minimax algorithm to find the best column(move) to play and the score of it.
        If the column valid for the current state of the board, it is drew and printed.
    '''
    startTime = pygame.time.get_ticks()

    col, minimaxScore = minimax(board, 2, -math.inf, math.inf, True)

    if board.isValidMove(col):
        board.setState(board.changeState(RED,col,board.board))
        board.drawBoard(board.board)
        board.printBoard(board.board)
        timeSinceThink = pygame.time.get_ticks() - startTime
        message = "Milliseconds since alpha beta think :" + str(timeSinceThink)
        print(message)
        

    pygame.display.update()
    FPSCLOCK.tick(60)


    

def getMonteMove(board):
    '''
        This function calls MCTS algorithm to find the best column(move) to play.
        If the column valid for the current state of the board, it is drew and printed.
    '''
    startTime = pygame.time.get_ticks()
    ai = MonteCarlo(1,board,BLACK)

    col = ai.run(board.board)

    if board.isValidMove(col):
        board.setState(board.changeState(BLACK,col,board.board))
        board.drawBoard(board.board)
        board.printBoard(board.board)
        timeSinceThink = pygame.time.get_ticks() - startTime
        message = "Milliseconds since monte-carlo think :" + str(timeSinceThink)
        print(message)
        

    pygame.display.update()
    FPSCLOCK.tick(60)



class Board(object):

    def __init__(self):
        super(Board,self).__init__()
        self.board = np.zeros((ROW_COUNT, COL_COUNT))

    def drawBoard(self,state):

        SCREEN.fill(LIGHTBLUE) # clears the board when the game start again

        spaceRect = pygame.Rect(0,0,SQUARESIZE,SQUARESIZE)

        # draws the rectangles from the BOARDING img
        for c in range(COL_COUNT):
            for r in range(ROW_COUNT):
                spaceRect.topleft = (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE)
                SCREEN.blit(BOARDIMG, spaceRect)

        # draws the tokens
        for c in range(COL_COUNT):
            for r in range(ROW_COUNT):
                spaceRect.topleft = (c*SQUARESIZE, height- r*SQUARESIZE-SQUARESIZE)
                if state[r][c] == RED:
                    SCREEN.blit(REDTOKENIMG, spaceRect)
                elif state[r][c] == BLACK:
                    SCREEN.blit(BLACKTOKENIMG, spaceRect)

        pygame.display.update()


    def printBoard(self,state):
        print("-----------------------------------")
        print(np.flip(state, 0))
        print("-----------------------------------")



    def changeState(self,player,move,state):
        # If the top column is not full
        if state[5][move] == 0:
            # Go down until it's not empty
            # goes down until it finds the non-empty place
            for i in reversed(range(6)): # reversed because starting point(0,0) is on left-down
                if state[i][move] != 0:
                    state[i+1][move] = player

                    return state
            # if we are here after the iterations it means the bottom should be filled
            state[0][move] = player
        return state



    def setState(self,state):
        self.board = state


    def isValidMove(self,col):

        return self.board[ROW_COUNT-1][col] == 0


    def getValidMoves(self,state):
        moves = [];
        for col in range(COL_COUNT):
            if state[ROW_COUNT-1][col] == 0:
                moves.append(col)
        return moves


    def isBoardFull(self,state):
    # returns True if there are no empty spaces anywhere on the board.
        for x in range(ROW_COUNT):
            for y in range(COL_COUNT):
                if state[x][y] == 0:
                    return False
        return True


    def isWinner(self,state, tile):
        # Check horizontal locations for win
        for c in range(COL_COUNT - 3):
            for r in range(ROW_COUNT):
                if state[r][c] == tile and state[r][c + 1] == tile and state[r][c + 2] == tile and state[r][c + 3] == tile:
                    return True

        # Check vertical locations for win
        for c in range(COL_COUNT):
            for r in range(ROW_COUNT - 3):
                if state[r][c] == tile and state[r + 1][c] == tile and state[r + 2][c] == tile and state[r + 3][c] == tile:
                    return True

        # Check positively sloped diaganols
        for c in range(COL_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if state[r][c] == tile and state[r + 1][c + 1] == tile and state[r + 2][c + 2] == tile and state[r + 3][c + 3] == tile:
                    return True

        # Check negatively sloped diaganols
        for c in range(COL_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if state[r][c] == tile and state[r - 1][c + 1] == tile and state[r - 2][c + 2] == tile and state[r - 3][c + 3] == tile:
                    return True




#----------------------------MONTE CARLO TREE SEARCH-----------------------------



class Node(object):

    def __init__(self,factor,parent, game,state,player,move):
        self.factor = factor
        self.parent = parent
        self.game = game
        self.state = state
        self.player = player
        self.move = move
        self.moves = game.getValidMoves(state)
        self.wins = 0
        self.visits = 0
        self.children = []



    def simulate(self):
        """This function simulates step for nodes.Function is splitted into 3 cases.
        -For the nodes that is not visited yet and has no children nodes,
        -visited but has no children nodes,
        -visited and has children nodes.
        It returns the result that the number represents of players(which 1 for MCTS and -1 for Minimax, 0 for else)
        """

        if self.moves == []:
            return 0
        # If the node is new, not visited and has no children, It runs the roll-out.
        if (not self.children) and self.visits == 0:
            self.visits = 1
            result = self.rollout(deepcopy(self.state))
            self.wins = self.wins + result#*self.player
            return result

        # If the roll-out is ran for the leaf node, creating children is needed.
        # Expanding the tree
        elif (not self.children):  # if children array is empty for the leaf node
            # We need to create children:
            for move in self.moves:
                node = Node(self.factor,self,self.game,self.game.changeState(self.player,move,deepcopy(self.state)),self.player*-1,move)
                self.children.append(node)

            result = self.children[0].simulate()
            self.visits = self.visits + 1
            self.wins = self.wins + result
            return result
        #If the node has child and alreay visited. We need to choose the best child and make simulation according this node.
        # selecting best child
        else:
            result = self.selectBestChild().simulate()

            self.visits = self.visits + 1

            self.wins = self.wins + result
            return result


    def selectBestChild(self):
        '''This function selects the child node which is higher than UCB1 value.
        UCB1 = wi/ni + C sqrt(ln(Ni)/ni)
        wi = number of wins after i-th move
        ni = number of simulations(visits) for the node considered after the i-th move
        Ni = total number of simulations(visits) after the i-th move run by the parent node
        C =  is the exploration parameter(factor for balancing exploitation and exploration
        '''

        bestChild = 0
        maxUCB1 = 0
        for i in range(len(self.children)):
            # If the current leaf node is not visited yet, it break the loop and return this children to simulate.
            if self.children[i].visits == 0:
                bestChild = i
                break
            # local calculation
            exploit = self.children[i].wins / self.children[i].visits
            # global calculation
            explore = math.sqrt(math.log(self.visits) / float(self.children[i].visits))
            # calculation according to the UCB1 formula
            score = exploit + self.factor * explore

            if score > maxUCB1:
                maxUCB1 = score
                bestChild = i

        return self.children[bestChild]



    def rollout(self,state):
        '''
        MCTS is based on many roll-outs.In this function, the game is played with random moves. It returns the final game result.
        This result is then used to calculate total wins for the node so that better nodes are to be chosen in future roll-outs.
        '''

        moves = self.game.getValidMoves(state)  # define valid moves
        player = self.player

        # while there are move to make in the state continous
        while(moves):
            # it choose a move randomly and change the current copy state according this move
            state = self.game.changeState(player,random.choice(moves),deepcopy(state))
            # it checks the winner.(if player is MCTS reslt = 1, player is Minimax result = -1 otherwise result is 0
            if self.game.isWinner(state,player):
                result = player
                if result !=0:
                    return result
            # changes player the turn(1,-1)
            player = player*-1
            moves = self.game.getValidMoves(state)

        return 0

    def getBestMove(self):
        ''' This function is returns the best moves as comparing the most visited node.
        '''
        best = self.children[0].visits
        move = self.children[0].move
        wins = self.children[0].wins
        for child in self.children:
            if child.visits > best:
                best = child.visits
                move = child.move
                wins = child.wins

        return move


class MonteCarlo(object):
    def __init__(self,factor,game,player):
        self.factor = factor
        self.game = game
        self.player = player

    def run(self,state):
        root = Node(self.factor,None,self.game,state,self.player,None)
        # it plays better with more iteration
        for i in range(1000):
            # simulates the node till iteration finish
            root.simulate()
        # returns the best move root can make according the simulations
        return root.getBestMove()
        # return a number between(0,6)




#-------------------------MINIMAX-ALPHA BETA PRUNNING-----------------------



def minimax(state, depth, alpha, beta, maximizingPlayer):
    '''
        This function is for choosing the best move(column)
        param state: current state of the board
        param depth: node index in the tree (difficulty)
        param alpha: assigned as -math.inf for the prunning
        param beta: assigned as math.inf for the prunning
        param maximizingPlayer: True for Minimax player, False for MCTS player.
    '''

    is_terminal = isTerminal(state)
    if depth == 0:
        return (None,scorer(state)) #score_position
    if is_terminal:
        return terminalState(state)
    if maximizingPlayer:
        return maxPlayer(state, depth, alpha, beta, maximizingPlayer)
    else:
        return minPlayer(state, depth, alpha, beta,maximizingPlayer)


def isTerminal(board):
    # checks if there is a winner or all valid moves run out
    return board.isWinner(board.board,RED) or board.isWinner(board.board,BLACK) or len(board.getValidMoves(board.board)) == 0


def terminalState(board):
    # if winner is Player Minimax
    if board.isWinner(board.board,RED):
        # returns col, minimaxScore
        return (None, HIGHVALUE)
    # if winner is Player MCTS
    if board.isWinner(board.board,BLACK):
        # returns col, minimaxScore
        return (None,-HIGHVALUE)
    else:
        # returns col, minimaxScore
        return (None,0)



def maxPlayer(state, depth, alpha, beta, maximizingPlayer):
    '''
        This function is called recursively to maximize the player Minimax's movements.
        This call minPlayer on each child of the board, which calls maxPlayer on each grandchild, and so on and so forth…
        The algorithm performs “depth-first search” .
    '''

    valid_locations = state.getValidMoves(state.board)
    value = -math.inf
    column = random.choice(valid_locations)
    for col in valid_locations:
        # copy the current board state not to change it.
        b_copy = copy.deepcopy(state)
        # change the state of copied board according the columns in valid locations
        b_copy.setState(b_copy.changeState(RED,col,b_copy.board))
        # calls minimax recursively, maximazingPlayer = False to minimize player MCTS
        new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
        if new_score > value:
            value = new_score
            column = col
        alpha = max(alpha,value)
        if alpha >=beta:
            break
    return column, value
    

def minPlayer(state, depth, alpha, beta, maximizingPlayer):
    '''
        This function is called recursively to minimize the player MCTS's movements.
        This call maxPlayer on each child of the board, which calls minPlayer on each grandchild, and so on and so forth…
        The algorithm performs “depth-first search” .
    '''

    valid_locations = state.getValidMoves(state.board)
    value = math.inf
    column = random.choice(valid_locations)
    for col in valid_locations:
        # copy the current board state not to change it.
        b_copy = copy.deepcopy(state)
        # change the state of copied board according the columns in valid locations
        b_copy.setState(b_copy.changeState(BLACK,col,b_copy.board))
        # calls minimax recursively, maximazingPlayer = False to minimize player MCTS
        new_score = minimax(b_copy, depth-1, alpha, beta,True)[1]
        if new_score < value:
            value = new_score
            column = col
        beta = min(beta,value)
        if alpha >= beta:
            break
    return column, value



def scanFours(board):
    '''
        This function returns two list:
        scan: all possible windows of 4
        locations: locations of all possible windows of 4

    '''

    scan = []
    locations = []

    for r in range(ROW_COUNT):
        for c in range(COL_COUNT):
            # horizontal
            if c + 3 < COL_COUNT:
                scan.append([board.board[r][c+i] for i in range(4)])
                locations.append([(r,c+i) for i in range(4)])
            # vertical
            if r + 3 < ROW_COUNT:
                scan.append([board.board[r+i][c] for i in range(4)])
                locations.append([(r+i,c) for i in range(4)])
            # negative sloped
            if r - 3 >= 0 and c + 3 < COL_COUNT:
                scan.append([board.board[r-i][c+i] for i in range(4)])
                locations.append([(r-i,c+i) for i in range(4)])
            # positive sloped
            if r + 3 < ROW_COUNT and c + 3 < COL_COUNT:
                scan.append([board.board[r+i][c+i] for i in range(4)])
                locations.append([(r+i,c+i) for i in range(4)])

    return scan,locations


def scorer(board):
    '''
        This function score combinations made by Minimax and MCTS players heuristicly.
    '''
    score = 0
    scan, locations = scanFours(board)

    # score positively for Minimax tile(RED) in center column
    centerColumn = [board.board[i][COL_COUNT // 2] for i in range(ROW_COUNT)]
    score += 3 * centerColumn.count(RED)

    for i in range(len(scan)):
        # score positively for combinations made by Minimax Player
        if scan[i].count(RED) == 4:
            score += 100

        if scan[i].count(RED) == 3 and scan[i].count(EMPTY) == 1:
            score += 10

        elif scan[i].count(RED) == 2 and scan[i].count(EMPTY) == 2:
            score += 4

        # score negatively for combinations made by MCTS Player
        if scan[i].count(BLACK) == 4:
            score += -100

        if scan[i].count(BLACK) == 3 and scan[i].count(EMPTY) == 1:
            score += -10

        elif scan[i].count(BLACK) == 2 and scan[i].count(EMPTY) == 2:
            score += -4



    return score



if __name__ == '__main__':
    main()




