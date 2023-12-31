import connection as cn
import numpy as np
import random as rd

s = cn.connect(2037)
#Deverá colocar o path para o result.txt do seu computador
q_table = np.loadtxt(r'C:\Users\Windows\Downloads\Q-learning-project-1\Qlearning-main\resultado.txt')
np.set_printoptions(precision = 6)

#Define, de acordo com uma constante epsilon, se ele realizará uma ação aleatória ou a melhor ação possível, retorna tal ação
def choose_action(epsilon,actions):
    if rd.random() < epsilon:
        action = actions[rd.randint(0,2)]
        print(f'Ação aleatória escolhida para o estado {curr_state}: {action}')

    else:
        number = np.argmax(q_table[curr_state])
        action = actions[number]
        print(f'Melhor ação escolhida para o estado {curr_state}: {action}')
    return action

def bellman_equation( r, s_prime, gamma):
    
    max_q_prime = np.max(q_table[s_prime])
    pontos = r + gamma * max_q_prime
    return pontos




curr_state = 0
curr_reward = -14
actions = ["left","right","jump"]
alpha = 0
gamma = 0.99
epslon = 0
while True:
    
    action = choose_action(epslon,actions)

    # Imprimindo epsilon ao longo das execuções
    if epslon > 0.4:
        epslon -= 0.0001
    print(f'epslon: {epslon}')

    if action == "left":
        col_action = 0
    elif action == "right":
        col_action = 1
    else:
        col_action = 2
    print(action)
    state,reward = cn.get_state_reward(s,action)
    print(reward)
    state = state[2:]

    state = int(state,2)
    next_state = state
    
    print(f'valor anterior dessa ação: {q_table[curr_state][col_action]}')
    q_table[curr_state][col_action] = q_table[curr_state][col_action] + alpha*(bellman_equation(reward,next_state,gamma) - q_table[curr_state][col_action])

    curr_state = next_state
    curr_reward = reward
    #Deverá colocar o path para o result.txt do seu computador
    np.savetxt(r'C:\Users\Windows\Downloads\Q-learning-project-1\Qlearning-main\resultado.txt', q_table, fmt="%f")
