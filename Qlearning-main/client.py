#Aqui vocês irão colocar seu algoritmo de aprendizado
import connection as cn

socket = cn.connect(2037)

while True:
    estado, recompensa = cn.get_state_reward(socket, "jump")