class GameState:
    def __init__(self, remaining_numbers, current_sum_p1, current_sum_p2, current_player):
        self.remaining_numbers = remaining_numbers
        self.current_sum_p1 = current_sum_p1
        self.current_sum_p2 = current_sum_p2
        self.current_player = current_player

def game_over(state):
    return len(state.remaining_numbers) == 0

def evaluate_state(state):
    return state.current_sum_p1 - state.current_sum_p2

def possible_actions(state):
    return state.remaining_numbers

def result(state, action):
    new_remaining_numbers = state.remaining_numbers[:]
    new_remaining_numbers.remove(action)
    new_player=None
    if state.current_player == 'p0':
        new_sum_p1 = state.current_sum_p1 + action
        new_sum_p2 = state.current_sum_p2
        if new_sum_p1>=new_sum_p2:
            new_player = 'p1'
    else:
        new_sum_p1 = state.current_sum_p1
        new_sum_p2 = state.current_sum_p2 + action
        if new_sum_p2>=new_sum_p1:
            new_player = 'p1'
    return GameState(new_remaining_numbers, new_sum_p1, new_sum_p2, new_player)

def min_max(state, depth, is_maximizer):
    if depth == 0 or game_over(state):
        return evaluate_state(state)
    
    if is_maximizer:
        max_eval = float('-inf')
        for action in possible_actions(state):
            eval = min_max(result(state, action), depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for action in possible_actions(state):
            eval = min_max(result(state, action), depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval

def alpha_beta_pruning(state, depth, alpha, beta, is_maximizer):
    if depth == 0 or game_over(state):
        return evaluate_state(state)
    
    if is_maximizer:
        max_eval = float('-inf')
        for action in possible_actions(state):
            eval = alpha_beta_pruning(result(state, action), depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for action in possible_actions(state):
            eval = alpha_beta_pruning(result(state, action), depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def choose_best_move(state, depth):
    if state.current_player == 'p1':
        best_move = None
        best_eval = float('-inf')
        for action in possible_actions(state):
            eval = min_max(result(state, action), depth - 1, False)
            if eval > best_eval:
                best_eval = eval
                best_move = action
        return best_move
    else:
        best_move = None
        best_eval = float('inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in possible_actions(state):
            eval = alpha_beta_pruning(result(state, action), depth - 1, alpha, beta, False)
            if eval < best_eval:
                best_eval = eval
                best_move = action
            beta = min(beta, eval)
        
        return best_move

def play_game_with_ai():
    n = int(input("Enter the highest natural number: "))
    depth = int(input("Enter the depth for AI search: "))
    state = GameState(list(range(1, n + 1)), 0, 0, 'p1')

    print('state',state)

    while not game_over(state):
        print("Remaining numbers:", state.remaining_numbers)
        print("Current sum P1:", state.current_sum_p1)
        print("Current sum P2:", state.current_sum_p2)

        if state.current_player == 'p1':
            print("Player 1's turn:")
            move = int(input("Choose a number: "))
            while move not in state.remaining_numbers:
                print("Invalid move! Please choose from remaining numbers.")
                move = int(input("Choose a number: "))
            state = result(state, move)
            while state.current_sum_p1 <= state.current_sum_p2 and not game_over(state):
                print("Player 1's sum is less than or equal to Player 2's sum. Player 1 can continue selecting numbers.")
                move = int(input("Choose another number: "))
                while move not in state.remaining_numbers:
                    print("Invalid move! Please choose from remaining numbers.")
                    move = int(input("Choose a number: "))
                state = result(state, move)
            state.current_player = 'p2'  # Switch to Player 2's turn

        else:
            print("Player 2's turn:")
            move = choose_best_move(state, depth)
            print("Player 2 (AI) chooses:", move)
            state = result(state, move)
            while state.current_sum_p2 <= state.current_sum_p1 and not game_over(state):
                print("Player 2's sum is less than or equal to Player 1's sum. Player 2 can continue selecting numbers.")
                move = choose_best_move(state, depth)
                print("Player 2 (AI) chooses:", move)
                while move not in state.remaining_numbers:
                    print("Invalid move! Please choose from remaining numbers.")
                    move = int(input("Choose a number: "))
                state = result(state, move)
            state.current_player = 'p1'  # Switch to Player 1's turn

    print("Game Over!")
    print("Final scores:")
    print("Player 1:", state.current_sum_p1)
    print("Player 2 (AI):", state.current_sum_p2)

    if state.current_sum_p1 > state.current_sum_p2:
        print("Player 1 wins!")
    elif state.current_sum_p1 < state.current_sum_p2:
        print("Player 2 (AI) wins!")
    else:
        print("It's a tie!")

# Example usage:
play_game_with_ai()