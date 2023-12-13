#Aqui vocês irão colocar seu algoritmo de aprendizado
import connection as cn
import numpy as np

socket = cn.connect(2037)

# Número de estados e ações
num_estados = 96
num_acoes = 3

# Inicialização da tabela Q com zeros
q_table = np.zeros((num_estados, num_acoes))

def choose_action(Q_table, current_state, eps):
    # Exploration: escolhe ação aleatoriamente
    if np.random.rand() < eps: # Gera um número aleatório entre 0 e 1 e verifica se é menor que epsilon
        action = np.random.choice(len(Q_table[current_state]))
    # Exploitation: escolhe a ação com o maior valor conhecido da Q_table
    else:
        action = np.argmax(Q_table[current_state])
    return action

# Estado inicial
estado = "0b0000000"

# Parâmetros
alpha = 0.5
gamma = 1 - alpha

while True:
    acao = choose_action(estado)
    novo_estado, recompensa = cn.get_state_reward(socket, acao)
    current_q = q_table[estado, acao] # Arrumar esses índices com uma função de transição
    max_next_q = np.max(q_table[novo_estado, :])
    novo_q = current_q + alpha * (recompensa + gamma * max_next_q - current_q)
    q_table[estado, acao] = novo_q

    estado = novo_estado