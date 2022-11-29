import time

from Node import *
from async_simulation import *

import numpy as np
import networkx as nx
import SimulationSpecification
import SimulationFunctionXTX_BTX



import matplotlib.pyplot as plt

def Message(node_id, sigma_yi, sigma_zi, msg_rel):
    msg = {'node_id': node_id,
           'sigma_yi': sigma_yi,
           'sigma_zi': sigma_zi,
           'msg_rel': msg_rel}
    return msg


# ###### Useful constant and flag ########
CONVERGENCE_FLAG = False
sim_spec = SimulationSpecification.SimulationSpecification()
agent_identifier = [i for i in range(sim_spec.number_of_nodes)]
epsilon = sim_spec.epsilon
convergence = np.zeros(sim_spec.number_of_nodes)
MAX_ITER = sim_spec.MAX_ITER
random_selection = random_selection(sim_spec.number_of_nodes)

# ######## GENERATE NETWORK ##########
is_graph_needed = True
p = sim_spec.p  # probability of node connection
while is_graph_needed:
    network_graph = nx.gnp_random_graph(sim_spec.number_of_nodes, p)
    graph_matrix = nx.to_numpy_array(network_graph)
    is_graph_needed = False
    for i in range(len(graph_matrix)):
        sum = 0
        for j in range(len(graph_matrix[i])):
            sum += graph_matrix[i][j]
        if sum == 0:
            is_graph_needed = True
            break
nx.draw(network_graph)
plt.show()

# ######### INIT #############
iter = 0

number_of_neighbors = [np.sum(graph_matrix[i]) for i in range(sim_spec.number_of_nodes)]
nodes = []

# choose type of cost function
cost_function = ['exp', 'poly_4', 'poly_2']
simulationFunction = SimulationFunctionXTX_BTX.SimulationFunctionXTX_BTX()

for i in range(sim_spec.number_of_nodes):
    node = Node(i,
                sim_spec.x0,
                sim_spec.epsilon,
                sim_spec.c,
                sim_spec.costfun,
                sim_spec.min_accepted_divergence,
                graph_matrix[i],
                simulationFunction
                )
    nodes.append(node)
message_container = [[] for i in range(sim_spec.number_of_nodes)]
msg_rel = 1
buffer = [[] for i in range(sim_spec.number_of_nodes)]

# ########### LOOP ###################
iter = 1
while not CONVERGENCE_FLAG:
    # randomly activate some agents
    #id_of_agent_activated = random.sample(agent_identifier, k=random.randint(1, sim_spec.number_of_nodes))  # chose which agent activate
    id_of_agent_activated = random_selection.persistent_communication()


    # usefull breakpoint to debug
    if iter == 2000:
        print(f"iter: {iter}")
    if iter%1000 == 0:
        print(f"iter: {iter}")
    if iter%100 == 0:
        print(f"iter: {iter}")


    # trasmit data from the randomly activated agents
    for i in id_of_agent_activated:
        # Transmission
        nodes[i].transmit_data()
        # Broadcast
        message_container[i] = Message(nodes[i].node_id, nodes[i].sigma_yi, nodes[i].sigma_zi, msg_rel)
        for j in range(sim_spec.number_of_nodes):
            if nodes[i].adjacency_vector[j] == 1:
                buffer[j].append(message_container[i])

    # receive message and update state
    for i in range(sim_spec.number_of_nodes):
        if len(buffer[i]) != 0 :
            msg = buffer[i].pop()
            nodes[i].receive_data(msg)
            nodes[i].update_estimation(iter)
        if nodes[i].has_result_founded():
            convergence[i] = 1

    # check if convergence is reached
    if (convergence == 1).all() or iter > MAX_ITER:
        CONVERGENCE_FLAG = True
        print(f"Reached convergence at iter:{iter}")
    iter += 1

distances = [[] for a in range(sim_spec.number_of_nodes)]
fig, axs = plt.subplots(1, 2, figsize=(13, 8))
for j in range(sim_spec.number_of_nodes):
    axs[0].plot(nodes[j].evolution_costfun, '-', label=f'J0_{j}')
    opt = nodes[j].xi
    '''for i in range(len(nodes[j].all_calculated_xis)):
        xi = np.array(nodes[j].all_calculated_xis[i])
        dst = np.sqrt(np.sum((opt - xi) ** 2))
        distances[j].append(dst)
    axs[2].plot(distances[j], '-', label=f'node_{j}')'''
    axs[1].plot(nodes[j].all_calculated_xis, label=f'node_{j}')
axs[0].legend(loc='upper right', ncol=1)
axs[0].set_title('Evolution of J0_i')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('cost function')
axs[1].legend(loc='upper right', ncol=1)
axs[1].set_title('Evolution xi')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Point-value evolution')
axs[0].grid()
axs[1].grid()
plt.savefig('multiple_plot1.png')
plt.show()

# logarithmic plot of MSE and distance until optimum
MSE = []
distances = [[] for a in range(sim_spec.number_of_nodes)]
fig, axs = plt.subplots(1, 2, figsize=(13, 8))
for k in range(iter-2):
    mse_k = 0
    for i in range(sim_spec.number_of_nodes):
        mse_k += ((np.abs(nodes[i].all_calculated_xis[k]-nodes[i].xi))**2)/sim_spec.number_of_nodes
        dst = np.sqrt(np.sum((nodes[i].xi - nodes[i].all_calculated_xis[k]) ** 2))
        distances[i].append(dst)
    MSE.append(mse_k)
for i in range(sim_spec.number_of_nodes):
    axs[1].plot(distances[i],'-', label=f'node_{i}')
axs[0].plot(MSE)
#axs[0].legend(loc='upper right', ncol=1)
axs[0].set_title('Mean Square Error')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('MSE')
axs[1].legend(loc='upper right', ncol=1)
axs[1].set_title('Distance until optimum found')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Distance until optimum')
axs[0].grid()
axs[1].grid()
axs[0].set_yscale('log')
axs[1].set_yscale('log')
plt.savefig('multiple_plot_log.png')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(13, 8))
for j in range(sim_spec.number_of_nodes):
    axs[0].plot(nodes[j].ratio_evol[0:2000], label=f'g/h_{j}')
    axs[1].plot(nodes[j].zi_evol[0:2000], label=f'z_{j}')
axs[0].legend(loc='upper right', ncol=1)
axs[0].set_title('Evolution of ratio consensus')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Sum(g)/Sum(h)')
axs[1].legend(loc='upper right', ncol=1)
axs[1].set_title('Evolution zi')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('z_i')
axs[0].grid()
axs[1].grid()
plt.savefig('consensus_signal.png')
plt.show()

for j in range(sim_spec.number_of_nodes):
    print(f'            node_{j}:\n'
          f'coeff:{nodes[j].coeff}\n'
          f'init:{nodes[j].all_calculated_xis[0]}\n'
          f'ending point:{nodes[j].xi}\n'
          f'----------------------------------------------------\n')




'''nodes[i].yi = (1 / (nodes[i].number_of_neighbors+1)) * nodes[i].yi
            nodes[i].zi = (1 / (nodes[i].number_of_neighbors + 1)) * nodes[i].zi
            nodes[i].sigma_yi = nodes[i].sigma_yi + nodes[i].yi
            nodes[i].sigma_zi = nodes[i].sigma_zi + nodes[i].zi'''