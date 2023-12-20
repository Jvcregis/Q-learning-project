#Aqui vocês irão colocar seu algoritmo de aprendizado
import connection as cn
import numpy as np
import random as rd
# Número de estados e ações


# Inicialização da tabela Q com zeros
#q_table = np.zeros((num_estados, num_acoes))

s = cn.connect(2037)
q_table = np.loadtxt(r'D:\João\FACULDADE\Q-learning-project\Qlearning-main\result.txt')
np.set_printoptions(precision = 6)
q_table.fill(0)




def choose_action(state,epsilon,actions):
    if rd.random() < epsilon:
        action = actions[rd.randint(0,2)]
        print(f'Ação aleatória escolhida para o estado {curr_state}: {action}')

    else:
        #state = int(state)
        number = np.argmax(q_table[curr_state, :])
        action = actions[number]
        print(f'Melhor ação escolhida para o estado {curr_state}: {action}')
    return action

def bellman_equation( r, s_prime, gamma):
    """
    Q: Q-values matrix
    s: current state
    a: chosen action
    r: reward for the action
    s_prime: next state
    gamma: discount factor
    """
    max_q_prime = np.max(q_table[s_prime, :])
    pontos = r + gamma * max_q_prime
    return pontos




curr_state = 0
curr_reward = -14
actions = ["left","right","jump"]
alpha = 0.1
gamma = 0.99
#estado, recompensa = cn.get_state_reward(s,"jump")

while True:
    action = choose_action(curr_state,0.4,actions)
    print(f'a')
    if action == "left":
        col_action = 0
    elif action == "right":
        col_action = 1
    else:
        col_action = 2
    print(action)
    state,reward = cn.get_state_reward(s,action)
    state = state[2:]

    state = int(state,2)
    next_state = state
    
    print(f'valor anterior dessa ação: {q_table[curr_state][col_action]}')
    q_table[curr_state][col_action] = q_table[curr_state][col_action] + alpha*(bellman_equation(curr_reward,next_state,gamma)) - q_table[curr_state][col_action]
    #print(f'valor novo dessa ação: {q_table[curr_state][action] + alpha*(bellman_equation(reward, curr_reward,gamma) - q_table[curr_state][action])}')

    curr_state = next_state
    curr_reward = reward

    np.savetxt(r'D:\João\FACULDADE\Q-learning-project\Qlearning-main\result.txt', q_table, fmt="%f")