from __future__ import division
from numpy import *

import numpy as np
import heapq as hpq
import tkinter
import gym

#  import matplotlib.pyplot as plt
#from pandas import Series
from matplotlib import pyplot

# get next state (C,A) and reward from environment
# C = (pic: probablity of staleness)
# A = (ta: latency of operations)
# def get_C_A_BlackboxStorage():

# WARS model for Dynamo storage for read and write latencies (http://www.bailis.org/papers/pbs-vldbj2014.pdf)
class WARS:
    #production_type = 'LNKD_DISK'
    #def __init__(self, production_type):
        #production_type = self.production_type # LNKD-SSD, LNKD-DISK, YMMR
    def nextW(self, production_type): # return W in ms
        if production_type == 'LNKD_SSD':
            u = np.random.uniform(0, 1, 1)
            if u < .9122:
                # generate Pareto(xm=.235, alpha=10)
                return (np.random.pareto(10) + 1) * .235
            else:
                # generate exponential(lambda=1.66)
                return np.random.exponential(1.0/1.66)

        if production_type == 'LNKD_DISK':
            u = np.random.uniform(0, 1, 1)
            if u < .38:
                # generate Pareto(xm=1.05, alpha=1.51)
                return (np.random.pareto(1.51) + 1) * 1.05
            else:
                # generate exponential(lambda=.183)
                return np.random.exponential(1.0 / .183)

        if production_type == 'YMMR':
            u = np.random.uniform(0, 1)
            if u < .939:
                return (np.random.pareto(3.35) + 1) * 3
            else:
                # generate exponential(lambda=.183)
                return np.random.exponential(1.0 / .0028)

    def nextA(self, production_type):
        if production_type in ['LNKD_SSD', 'LNKD_DISK']:
            u = np.random.uniform(0, 1)
            if u < .9122:
                # generate Pareto(xm=.235, alpha=10)
                return (np.random.pareto(10) + 1) * .235
            else:
                # generate exponential(lambda=1.66)
                return np.random.exponential(1.0 / 1.66)

        if production_type == 'YMMR':
            u = np.random.uniform(0, 1, 1)
            if u < .982:
                return (np.random.pareto(3.8) + 1) * 1.5
            else:
                # generate exponential(lambda=.183)
                return np.random.exponential(1.0 / .0217)

    def nextR(self, production_type):
        if production_type in ['LNKD_SSD', 'LNKD_DISK']:
            u = np.random.uniform(0, 1, 1)
            if u < .9122:
                # generate Pareto(xm=.235, alpha=10)
                return (np.random.pareto(10) + 1) * .235
            else:
                # generate exponential(lambda=1.66)
                return np.random.exponential(1.0 / 1.66)

        if production_type == 'YMMR':
            u = np.random.uniform(0, 1, 1)
            if u < .982:
                return (np.random.pareto(3.8) + 1) * 1.5
            else:
                # generate exponential(lambda=.183)
                return np.random.exponential(1.0 / .0217)

    def nextS(self, production_type):
        if production_type in ['LNKD_SSD', 'LNKD_DISK']:
            u = np.random.uniform(0, 1, 1)
            if u < .9122:
                # generate Pareto(xm=.235, alpha=10)
                return (np.random.pareto(10) + 1) * .235
            else:
                # generate exponential(lambda=1.66)
                return np.random.exponential(1.0 / 1.66)

        if production_type == 'YMMR':
            u = np.random.uniform(0, 1, 1)
            if u < .982:
                return (np.random.pareto(3.8) + 1) * 1.5
            else:
                # generate exponential(lambda=.183)
                return np.random.exponential(1.0 / .0217)



class DistributedStorageSystem:
    # def __init__(self):

    # N = number of replicas
    # R = read quorum
    # W = write quorum
    # TC = freshness interval
    # TA = read latency deadline
    # iterations = number of iterations to get estimates of PIC, and PUA
    # read_delay = amount by which read invocation is delayed
    # write_delay = amount by which write response is delayed
    def Compute_PIC_PUA_Given_TC_TA(self, N, R, W, TC, TA, iterations, read_delay = 0, write_delay = 0):
        consistent_trials = 0
        latency_trials = 0
        wars = WARS()
        for i in range(iterations):
            Ws = zeros(N)
            As = zeros(N)
            Rs = zeros(N)
            Ss = zeros(N)
            write_latencies = zeros(N)
            read_latencies = zeros(N)

            for replica in range(N):
                #Ws.append(wars.nextW())
                Ws[replica] = wars.nextW('LNKD_DISK')

                #As.append(wars.nextA())
                As[replica] = wars.nextA('LNKD_DISK')

                write_latencies[replica] = Ws[replica] + As[replica] + write_delay

                #Rs.append(wars.nextR())
                #Ss.append(wars.nextS())
                Rs[replica] = wars.nextR('LNKD_DISK')
                Ss[replica] = wars.nextS('LNKD_DISK')

                read_latencies[replica] = Rs[replica] + Ss[replica] + read_delay
            #print(write_latencies)

            #sorted_write_latencies = write_latencies.sort()
            #print (sorted_write_latencies)
            #write_finish = sorted_write_latencies[W]
            write_finish = hpq.nsmallest(W, write_latencies)[0]

            #print(write_finish)

            #sorted_read_latencies = read_latencies.sort()
            #read_finish = sorted_read_latencies[R]

            read_finish = hpq.nsmallest(R, read_latencies)[0]

            #print(read_finish)

            if read_finish < TA:
                latency_trials = latency_trials + 1

            reply_replicas = []

            for replica in range(N):
                if read_latencies[replica] <= read_finish:
                    reply_replicas.append(replica)

            for replica in reply_replicas:
                if write_finish + Rs[replica] + TC >= Ws[replica]:
                    consistent_trials = consistent_trials + 1
                    break

        PIC = consistent_trials / iterations

        PUA = latency_trials / iterations

        # return  PIC, PUA

        return consistent_trials, latency_trials # return integer frequencies instead of real valued probabilities

    def tabularQLearning(self, granularity, N):
        # instead of this gym environemnt, we need a custom environment simulating the consistency behavior of a black box storage system
        # env = gym.make('FrozenLake-v0')
        #Initialize table with all zeros

        # granularity = 100 # granularity of state space (C=pic, A=pua) and action space

        Q = np.zeros([granularity, granularity]) # 2 D state space, C, A, 2D action space read and write delay

        # Set learning parameters
        lr = .8
        y = .95
        num_episodes = 200
        #create lists to contain total rewards and steps per episode
        #jList = []
        rList = []
        for i in range(num_episodes):
            #Reset environment and get first new observation
            #s = env.reset()
            sc, sa = self.Compute_PIC_PUA_Given_TC_TA(N, 1, 1, 0.1, .5, granularity) # initial state
            #sc = int(round(scr))
            #sa = int(round(sar))
            # sc = int_(floor(scr * 100))
            # sa = int_(floor(sar * 100))

            rAll = 0
            d = False
            j = 0
            #The Q-Table learning algorithm
            while j < 9:
                j+=1

                #Choose an action by greedily (with noise) picking from Q table
                ac = np.argmax(Q[sc, :] + np.random.randn(1, granularity)*(1./(i+1)))

                # avoid noise for now
                #print(sc, sa)
                #ac = np.argmax(Q[sc:])
                if (ac >= granularity):
                    continue
                #print(ac)
                #if (Q[sc,0] > Q[sc,1]):
                 #   ac = 0
                #else:
                 #   ac = 1

                #Get new state and reward from environment
                #s1,r,d,_ = env.step(a)
                #if (ac == 0):
                sc1, sa1 = self.Compute_PIC_PUA_Given_TC_TA(N, 1, 1, 0.1, 0.5, granularity, ac/granularity)
                #else:
                 #   sc1, sa1 = self.Compute_PIC_PUA_Given_TC_TA(3, 1, 1, 0.1, 0.5, granularity, -1)
                #r = sc1 + sa1
                r = sc1
                #sc1 = int_(floor(sc1 * granularity))
                #sa1 = int_(floor(sa1 * granularity))
                #print(sc1,sa1)

                #Update Q-Table with new knowledge
                #print (sc, ac, Q[sc,ac])
                #print (lr*(r + y*np.max(Q[sc1,:]) - Q[sc, ac]))

                Q[sc, ac] = Q[sc, ac] + lr*(r + y*np.max(Q[sc1,:]) - Q[sc, ac])
                rAll += r
                #print(rAll)
                sc = sc1
                sa = sa1
                #if d == True:
                 #   break
                #jList.append(j)
        rList.append(rAll)

        #print ("Score over time: " +  str(sum(rList)/num_episodes))

        #print("Final Q-Table Values")
        #print(Q)
        #print(np.count_nonzero(Q))

        # traverse learned policy

        print ("traversing learned policy")
        # initial state
        sc, sa = self.Compute_PIC_PUA_Given_TC_TA(N, 1, 1, 0.1, .5, granularity)  # initial state

        C = np.zeros(granularity)
        A = np.zeros(granularity)
        T = np.zeros(granularity)

        for i in range(granularity):
            ac = np.argmax(Q[sc, :])
            print(sc, sa, ac, Q[sc, ac])
            sc, sa = self.Compute_PIC_PUA_Given_TC_TA(N, 1, 1, 0.1, 0.5, granularity, ac / granularity)
            C[i] = sc / granularity
            A[i] = sa / granularity
            T[i] = i

        #plt.plot(C)


        pyplot.plot(T, C, 'C')
        pyplot.plot(T, A)
        pyplot.plot(T, C + A)
        pyplot.show()


if __name__ == "__main__":


    #wars = WARS('TS')
    #print(wars.nextW())
    store = DistributedStorageSystem()
    #print(store.Compute_PIC_PUA_Given_TC_TA(3, 1, 1, 0.1, .5, 1000)) #
    store.tabularQLearning(300, 5)
