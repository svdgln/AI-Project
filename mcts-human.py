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
HUMAN = "human"
MCTS = "mcts"

#for defining player tiles
RED = -1 # human
BLACK = 1 # mcts

EMPTY = 0
startTime = 0

HIGHVALUE = 100000000000000



def main():
    global FPSCLOCK, SCREEN, REDPILERECT, BLACKPILERECT, REDTOKENIMG, myfont
    global BLACKTOKENIMG, BOARDIMG, HUMANWINNERIMG
    global COMPUTERWINNERIMG, WINNERRECT, TIEWINNERIMG
    global MCTSWINNER, MINIMAXWINNER
    global winnerCountHuman,winnerCountMcts, tieCount


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

    winnerCountHuman = 0
    winnerCountMcts = 0
    tieCount = 0
    countGame = 0

    while True:
        countGame += 1
        winnerCountHuman , winnerCountMcts, tieCount = runGame(isFirstGame,winnerCountHuman,winnerCountMcts,tieCount)
        isFirstGame = False
        print("Total game: " + str(countGame))
        print("Count Human Winning : " + str(winnerCountHuman))
        print("Count MCTS Winning : " + str(winnerCountMcts))
        print("Count Ties : " + str(tieCount))



def runGame(isFirstGame,winnerCountHuman,winnerCountMcts,tieCount):

    if isFirstGame:
        turn = HUMAN
        print("HAVE FUN !!")
    else:
        # first player is chose randomly
        if random.randint(0, 1) == 0:
            turn = MCTS
        else:
            turn = HUMAN

    #creates board object
    board = Board()
    # prints board with all zeros
    board.printBoard(board.board)
    board.drawBoard(board.board)
    pygame.display.update()
    display = True

    # Main Game Loop
    while True:

        if turn == HUMAN: # human player turn
            getHumanMove(board)
            if board.isWinner(board.board,RED):
                winnerCountHuman += 1
                winnerImg = HUMANWINNERIMG
                print("Human WINNER!!")
                break
            turn = MCTS # switches to the other player

        else:
            getMonteMove(board) # MCTS player turn
            if board.isWinner(board.board,BLACK):
                winnerCountMcts += 1
                winnerImg = MCTSWINNER
                print("MonteCarlo WINNER!!")
                break
            turn = HUMAN # switches to the other player

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

    return winnerCountHuman, winnerCountMcts, tieCount
        


def getHumanMove(board):

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(SCREEN,LIGHTBLUE , (0,0, width, SQUARESIZE))
                posx = event.pos[0]
                SCREEN.blit(REDTOKENIMG, (posx - RADIUS,0))
            pygame.display.update()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(SCREEN,LIGHTBLUE , (0,0, width, SQUARESIZE))
                
                posx = event.pos[0]
                col = int(math.floor(posx/SQUARESIZE))
                if board.isValidMove(col):
                    board.setState(board.changeState(RED,col,board.board))
                    board.drawBoard(board.board)
                    board.printBoard(board.board)
                    pygame.display.update()
                    return


        pygame.display.update()
        FPSCLOCK.tick()

    

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
        It returns the result that the number represents of players(which 1 for MCTS and -1 for Human, 0 for else)
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
            # it checks the winner.(if player is MCTS reslt = 1, player is Human result = -1 otherwise result is 0
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





if __name__ == '__main__':
    main()




