from connect_four import *

new_board = make_board()
select_space(new_board, 6, "X")
select_space(new_board, 5, "X")
select_space(new_board, 6, "X")
select_space(new_board, 5, "X")
select_space(new_board, 6, "O")
select_space(new_board, 5, "O")
select_space(new_board, 6, "O")
select_space(new_board, 5, "O")
# print_board(new_board)


def two_ai_game():
    my_board = make_board()
    while not game_is_over(my_board):
      #The "X" player finds their best move.
      result = minimax(my_board, True, 6, -float("Inf"), float("Inf"), my_evaluate_board)
      print( "X Turn\nX selected ", result[1])
      print(result[1])
      select_space(my_board, result[1], "X")
      print_board(my_board)

      if not game_is_over(my_board):
        #The "O" player finds their best move
        result = minimax(my_board, False, 6, -float("Inf"), float("Inf"), codecademy_evaluate_board)
        print( "O Turn\nO selected ", result[1])
        print(result[1])
        select_space(my_board, result[1], "O")
        print_board(my_board)
    if has_won(my_board, "X"):
        print("X won!")
    elif has_won(my_board, "O"):
        print("O won!")
    else:
        print("It's a tie!")

def random_eval(board):
  return random.randint(-100, 100)

def my_evaluate_board(board):
  if has_won(board, "X"):
    return float("Inf")
  elif has_won(board, "O"):
    return -float("Inf")

  x_two_streak = 0
  o_two_streak = 0
  for col in range(len(board)):
    for row in range(len(board[0])-2):
      if board[col][row] == "O" and board[col][row+1] == "O" and board[col][row+2] == "O":
        o_two_streak += 2
      if board[col][row] == "X" and board[col][row+1] == "X" and board[col][row+2] == "X":
        x_two_streak += 2

  for col in range(len(board)-2):
    for row in range(len(board[0])):
      if board[col][row] == "X" and board[col+1][row] == "X" and board[col+2][row] == "X":
        x_two_streak += 2
      if board[col][row] == "O" and board[col+1][row] == "O" and board[col+2][row] == "O":
        o_two_streak += 2

  for col in range(len(board)):
    for row in range(len(board[0])-1):
      if board[col][row] == "O" and board[col][row+1] == "O":
        o_two_streak += 1
      if board[col][row] == "X" and board[col][row+1] == "X":
        x_two_streak += 1

  for col in range(len(board)-1):
    for row in range(len(board[0])):
      if board[col][row] == "X" and board[col+1][row] == "X":
        x_two_streak += 1
      if board[col][row] == "O" and board[col+1][row] == "O":
        o_two_streak += 1
      
  return x_two_streak - o_two_streak

# print(my_evaluate_board(new_board))
two_ai_game()
