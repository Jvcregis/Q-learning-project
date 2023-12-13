#Aqui vocês irão colocar seu algoritmo de aprendizado
import connection as cn
import numpy as np

socket = cn.connect(2037)

Q_table = []

def choose_action(Q_table, current_state, eps):
    # Exploration: escolhe ação aleatoriamente
    if np.random.rand() < eps: # Gera um número aleatório entre 0 e 1 e verifica se é menor que epsilon
        action = np.random.choice(len(Q_table[current_state]))
    # Exploration: escolhe a ação com o maior valor conhecido da Q_table
    else:
        action = np.argmax(Q_table[current_state])
    return action


while True:
    # Ação Inicial
    estado_inicial = "0b0000000"
    #while estado != estado_terminal
    estado, recompensa = cn.get_state_reward(socket, "jump")