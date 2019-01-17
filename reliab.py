# Python program to print DFS traversal from a
# given given graph
from collections import defaultdict
import numpy as np
import random
import time

# This class represents a directed graph using
# adjacency list representation
class Graph:

    # Constructor
    def __init__(self):

        # default dictionary to store graph
        self.graph = defaultdict(list)

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # A function used by DFS
    def DFSUtil(self, v, visited, target):

        # Mark the current node as visited and print it
        visited[v]= True
        # print(v),
        global xx
        if v == target:
            # print("__1__")
            xx = 1

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if not visited[i]:
                self.DFSUtil(i, visited, target)



    # The function to do DFS traversal. It uses
    # recursive DFSUtil()
    def DFS(self, v, target):

        # Mark all the vertices as not visited
        visited = [False]*(len(self.graph))

        # Call the recursive helper function to print
        # DFS traversal
        self.DFSUtil(v,visited, target)

def P_j_i(n_vertices, prob):
    prob_y_k = np.zeros((n_vertices**2+1, n_vertices**2))  \
        # to save the probabilities (i in columns and j in rows for Pj(i))
    prob = prob.reshape(n_vertices**2, )  \
        # number the edges in left to right in a row of adjacency matrix then next row

    prob_y_k[1, 0] = prob[0]  # P_1(1)
    prob_y_k[2:, 0] = 0  # P_{k>1}(1)
    prob_y_k[0, 0] = 1 - prob[0]
    for i in range(1, n_vertices**2):
        prob_y_k[0, i] = (1 - prob[i]) * prob_y_k[0, i-1] # first row in the matrix

    for j in range(1, n_vertices**2):
        for i in range(1, n_vertices**2):
            prob_y_k[j, i] = prob_y_k[j-1, i-1]* prob[i] + prob_y_k[j, i-1]* (1-prob[i])

    # prob_Y_k = prob_y_k[:, n_vertices ** 2-1]  # last column shows the P(Y=k)
    return prob_y_k

def simulate_X_given_y_k(n_vertices, prob, k, P_j_i):
    X = np.zeros(n_vertices**2)  # the adjacency matrix reshaped (numbered from 0 to n**2-1
    prob = prob.reshape(n_vertices ** 2, )
    art_floor = 1e-5
    prob_X_is_1_given_y_k_and_other_Xs = max(prob[n_vertices**2-1] * P_j_i[k-1, n_vertices**2-1-1]/\
                                         P_j_i[k, n_vertices**2-1], art_floor) # start with X_n**2
    for nn in range((n_vertices**2)-1, -1, -1):
        X[nn] = np.maximum(np.sign(prob_X_is_1_given_y_k_and_other_Xs - np.random.rand()), 0)
        k_i_n = int(k - 1 - sum(X[nn:n_vertices**2]))
        k_i_n_1 = k_i_n + 1
        if k_i_n < 0 or k_i_n_1 < 0:
            prob_X_is_1_given_y_k_and_other_Xs = 0
        else:
            prob_X_is_1_given_y_k_and_other_Xs = max(np.nan_to_num(prob[nn-1] * P_j_i[k_i_n, nn-2]/\
                                                 P_j_i[k_i_n_1, nn-1]), art_floor)
    if k != np.sum(X):
        if k > np.sum(X):
            while k > np.sum(X):
                Index_to_ones = random.choice(list(enumerate(X[X == 0])))[0]
                X[Index_to_ones] = 1
        elif k < np.sum(X):
            while k < np.sum(X):
                Index_to_ones = random.choice(list(enumerate(X[X == 1])))[0]
                X[Index_to_ones] = 0
    if k != np.sum(X):
        print(k - np.sum(X))

    return X


if __name__ == '__main__':
    max_ite = 5000
    small_sim_ite = 100  # not smaller than number of edges to have some simulations for each y_s
    n_vertices = 30  # number of vertices
    # prob = 0.45   # probability of an edge working (Mark1)
    Lpi, Upi = 0, 0.1
    print(max_ite, small_sim_ite, n_vertices, Lpi, Upi)
    np.random.seed(232)
    prob = np.random.uniform(Lpi, Upi, (n_vertices, n_vertices))  # a matrix of probability of each edge working
    np.fill_diagonal(prob, 1)

    ####
    ### simple simulation
    ####
    t0 = time.time()
    ave = np.zeros(max_ite)  # if iteration worked is 1 else 0
    sum_y = np.zeros(max_ite)  # y which is sum of X's in iteration
    for ite in range(0, max_ite):
        # generating the matrix
        # adj_mat = np.random.choice(2, (n_vertices, n_vertices), p=[1- prob, prob])  # (Mark1)
        adj_mat = np.maximum(np.sign(prob - np.random.rand(n_vertices, n_vertices)), 0)
        sum_y[ite] = np.sum(adj_mat)
        np.fill_diagonal(adj_mat, 1)  # to add self edges to have the visited matrix with the same size as n_vertices

        g = Graph()  # for now it is a directed graph
        for i in range(0, n_vertices):
            for j in range(0, n_vertices):
                if adj_mat[i, j] != 0:
                    g.addEdge(i, j)
        start = 0
        target = n_vertices - 1
        # Driver code for an specific example
        # g = Graph()
        # g.addEdge(0, 1)
        # g.addEdge(1, 2)
        # g.addEdge(2, 4)
        # g.addEdge(3, 4)
        # g.addEdge(5, 4)
        # # g.addEdge(2, 3)
        # # g.addEdge(2, 4)
        # # g.addEdge(3, 4)
        # # g.addEdge(4, 5)
        # start  = 0
        # target = 5

        # print ("Following is DFS from (starting from vertex "+str(start)+")")
        xx = 0
        g.DFS(start, target)
        ave[ite] = xx

    print("Simple simulation ave = ", np.mean(ave), "Variance =", np.var(ave)/max_ite, "Time", time.time()-t0, "Straight calculation of Var=", np.mean(ave)*(1-np.mean(ave))/max_ite)

    unique, counts = np.unique(sum_y, return_counts=True)
    # print("(Y: Proportion):", dict(zip(unique, counts/max_ite)))

    P_w_y_k = []  # E(X| y=K)
    Var_y_k = []  # var(X | y=k)
    t00 = time.time()
    P_j_i = P_j_i(n_vertices, prob)
    print("Time of calc P_j_i", time.time()-t00)
    P_j_n = P_j_i[:, n_vertices ** 2 - 1]  # P(Y=k) for all possible k
    P_j_y_k = []  # P(y=k) for those have happened here
    for Y in unique:
        # print("P(work|Y=", Y, ")=", np.mean(ave[sum_y == Y]) )
        P_w_y_k.append(np.mean(ave[sum_y == Y]))
        Var_y_k.append(np.var(ave[sum_y == Y]))
        P_j_y_k.append(P_j_n[int(Y)])

    P_w_y_k = np.array(P_w_y_k)
    Var_y_k = np.array(Var_y_k)
    P_j_y_k = np.array(P_j_y_k)

    # probabilities for the post stratifying and conditional simulation (stratifying)
    print("Post stratifying average =", sum(P_j_y_k * P_w_y_k), "Var=", sum(Var_y_k * P_j_y_k**2/counts), "time", time.time() - t0 \
          , "Straight calculation of Var=", sum(P_w_y_k*(1-P_w_y_k) * P_j_y_k**2/counts))

    ####
    ## stratified sampling
    ####
    t0 = time.time()
    ave = []  # if iteration worked is 1 else 0
    var = []
    counts_y_k = []
    sum_y = []  # y which is sum of X's in iteration
    P_j_y_k = P_j_n[P_j_n != 0]
    y_s = np.nonzero(P_j_n)[0]  # Non zero indices of P_j_n + 1 so Y=k for P(Y=k)!=0
    P_j_y_k_non_trivial = []
    counter = 0  # to keep track in the P_j_y_k
    for y in y_s:
        ave_y_k = []
        iteration = int(P_j_y_k[counter] * max_ite)
        if iteration != 0:
            counts_y_k.append(iteration)
            P_j_y_k_non_trivial.append(P_j_y_k[counter])
            for ite in range(0, iteration):
                # generating the matrix
                adj_mat = simulate_X_given_y_k(n_vertices, prob, int(y), P_j_i)
                sum_y.append(np.sum(adj_mat))
                adj_mat = adj_mat.reshape((n_vertices, n_vertices))
                np.fill_diagonal(adj_mat, 1)  # to add self edges to have the visited matrix with the same size as n_vertices
                g = Graph()  # for now it is a directed graph
                for i in range(0, n_vertices):
                    for j in range(0, n_vertices):
                        if adj_mat[i, j] != 0:
                            g.addEdge(i, j)

                start = 0
                target = n_vertices - 1
                xx = 0
                g.DFS(start, target)
                ave_y_k.append(xx)
            if len(ave_y_k) != 0:
                ave.append(np.mean(ave_y_k))
                var.append(np.var(ave_y_k))
            else:
                ave.append(0)
                var.append(0)
        counter += 1
    P_j_y_k_non_trivial = np.array(P_j_y_k_non_trivial)
    ave = np.array(ave)
    print("Stratified aveg=", sum(ave * P_j_y_k_non_trivial), "Stratified var=", sum(var * P_j_y_k_non_trivial**2 / counts_y_k), \
          "time", time.time()- t0, "Stratified var (straight) =", sum(ave*(1 - ave) * P_j_y_k_non_trivial ** 2 / counts_y_k))


    ### Conditional simulation (stratified) with optimized n_i
    t0 = time.time()
    ## s_i estimation
    # n_i = N*p_i*s_i/sum(p_i*s_i)
    sample_var = []
    counter = 0  # to keep track in the P_j_y_k
    for y in y_s:
        ave_y_k = []
        for ite in range(0, small_sim_ite):
            # generating the matrix
            adj_mat = simulate_X_given_y_k(n_vertices, prob, int(y), P_j_i)
            adj_mat = adj_mat.reshape((n_vertices, n_vertices))
            np.fill_diagonal(adj_mat,
                             1)  # to add self edges to have the visited matrix with the same size as n_vertices
            g = Graph()  # for now it is a directed graph
            for i in range(0, n_vertices):
                for j in range(0, n_vertices):
                    if adj_mat[i, j] != 0:
                        g.addEdge(i, j)

            start = 0
            target = n_vertices - 1

            # print ("Following is DFS from (starting from vertex "+str(start)+")")
            xx = 0
            g.DFS(start, target)
            # print(xx)
            ave_y_k.append(xx)
            # print("xx", xx)
        if len(ave_y_k) != 0:
            sample_var.append(np.var(ave_y_k))
        counter += 1
    sample_var = np.array(sample_var)
    n_i = max_ite * P_j_y_k * sample_var / sum(P_j_y_k * sample_var)
    n_i = n_i.astype(int) + 1

    ## stratified with optimal n_i
    ave = []  # if iteration worked is 1 else 0
    var = []
    counter = 0  # to keep track in the P_j_y_k
    for y in y_s:
        ave_y_k = []
        for ite in range(0, n_i[counter]):
            # generating the matrix
            adj_mat = simulate_X_given_y_k(n_vertices, prob, int(y), P_j_i)
            sum_y.append(np.sum(adj_mat))
            adj_mat = adj_mat.reshape((n_vertices, n_vertices))
            np.fill_diagonal(adj_mat, 1)  # to add self edges to have the visited matrix with the same size as n_vertices
            g = Graph()  # for now it is a directed graph
            for i in range(0, n_vertices):
                for j in range(0, n_vertices):
                    if adj_mat[i, j] != 0:
                        g.addEdge(i, j)

            start = 0
            target = n_vertices - 1

            xx = 0
            g.DFS(start, target)
            ave_y_k.append(xx)
        if len(ave_y_k) != 0:
            ave.append(np.mean(ave_y_k))
            var.append(np.var(ave_y_k))
        else:
            ave.append(0)
            var.append(0)
        counter += 1

    ave = np.array(ave)
    print("Optimal Stratified aveg =", sum(ave * P_j_y_k), "Optimal Stratified var =", sum(var * P_j_y_k** 2 / n_i), \
          "time", time.time()-t0, "Optimal Stratified var (straight) =", sum(ave*(1 - ave ) * P_j_y_k ** 2 / n_i))










