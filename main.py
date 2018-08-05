from __future__ import division
from numpy import *

import numpy as np
import heapq as hpq
import gym

# get next state (C,A) and reward from environment
# C = (pic: probablity of staleness)
# A = (ta: latency of operations)
# def get_C_A_BlackboxStorage():

# WARS model for Dynamo storage for read and write latencies (http://www.bailis.org/papers/pbs-vldbj2014.pdf)
class WARS:
    production_type = 'LNKD_SSD'
    def __init__(self, production_type):
        production_type = self.production_type # LNKD-SSD, LNKD-DISK, YMMR
    def nextW(self): # return W in ms
        if self.production_type == 'LNKD_SSD':
            u = np.random.uniform(0, 1, 1)
            if u < .9122:
                # generate Pareto(xm=.235, alpha=10)
                return (np.random.pareto(10) + 1) * .235
            else:
                # generate exponential(lambda=1.66)
                return np.random.exponential(1.0/1.66)

        if self.production_type == 'LNKD_DISK':
            u = np.random.uniform(0, 1, 1)
            if u < .38:
                # generate Pareto(xm=1.05, alpha=1.51)
                return (np.random.pareto(1.51) + 1) * 1.05
            else:
                # generate exponential(lambda=.183)
                return np.random.exponential(1.0 / .183)

        if self.production_type == 'YMMR':
            u = np.random.uniform(0, 1)
            if u < .939:
                return (np.random.pareto(3.35) + 1) * 3
            else:
                # generate exponential(lambda=.183)
                return np.random.exponential(1.0 / .0028)

    def nextA(self):
        if self.production_type in ['LNKD_SSD', 'LNKD_DISK']:
            u = np.random.uniform(0, 1)
            if u < .9122:
                # generate Pareto(xm=.235, alpha=10)
                return (np.random.pareto(10) + 1) * .235
            else:
                # generate exponential(lambda=1.66)
                return np.random.exponential(1.0 / 1.66)

        if self.production_type == 'YMMR':
            u = np.random.uniform(0, 1, 1)
            if u < .982:
                return (np.random.pareto(3.8) + 1) * 1.5
            else:
                # generate exponential(lambda=.183)
                return np.random.exponential(1.0 / .0217)

    def nextR(self):
        if self.production_type in ['LNKD_SSD', 'LNKD_DISK']:
            u = np.random.uniform(0, 1, 1)
            if u < .9122:
                # generate Pareto(xm=.235, alpha=10)
                return (np.random.pareto(10) + 1) * .235
            else:
                # generate exponential(lambda=1.66)
                return np.random.exponential(1.0 / 1.66)

        if self.production_type == 'YMMR':
            u = np.random.uniform(0, 1, 1)
            if u < .982:
                return (np.random.pareto(3.8) + 1) * 1.5
            else:
                # generate exponential(lambda=.183)
                return np.random.exponential(1.0 / .0217)

    def nextS(self):
        if self.production_type in ['LNKD_SSD', 'LNKD_DISK']:
            u = np.random.uniform(0, 1, 1)
            if u < .9122:
                # generate Pareto(xm=.235, alpha=10)
                return (np.random.pareto(10) + 1) * .235
            else:
                # generate exponential(lambda=1.66)
                return np.random.exponential(1.0 / 1.66)

        if self.production_type == 'YMMR':
            u = np.random.uniform(0, 1, 1)
            if u < .982:
                return (np.random.pareto(3.8) + 1) * 1.5
            else:
                # generate exponential(lambda=.183)
                return np.random.exponential(1.0 / .0217)



class DistributedStorageSystem:
    # def __init__(self):

    def Compute_PIC_PUA_Given_TC_TA(self, N, R, W, TC, TA, iterations):
        consistent_trials = 0
        latency_trails = 0
        wars = WARS('LNKD_SSD')
        for i in range(iterations):
            Ws = zeros(N)
            As = zeros(N)
            Rs = zeros(N)
            Ss = zeros(N)
            write_latencies = zeros(N)
            read_latencies = zeros(N)

            for replica in range(N):
                #Ws.append(wars.nextW())
                Ws[replica] = wars.nextW()

                #As.append(wars.nextA())
                As[replica] = wars.nextA()

                write_latencies[replica] = Ws[replica] + As[replica]

                #Rs.append(wars.nextR())
                #Ss.append(wars.nextS())
                Rs[replica] = wars.nextR()
                Ss[replica] = wars.nextS()

                read_latencies[replica] = Rs[replica] + Ss[replica]
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
                latency_trails = latency_trails + 1

            reply_replicas = []

            for replica in range(N):
                if read_latencies[replica] <= read_finish:
                    reply_replicas.append(replica)

            for replica in reply_replicas:
                if write_finish + Rs[replica] + TC >= Ws[replica]:
                    consistent_trials = consistent_trials + 1
                    break

        PIC = consistent_trials / iterations

        PUA = latency_trails / iterations

        return  PIC, PUA

    def tabularQLearning(self):
        # instead of this gym environemnt, we need a custom environment simulating the consistency behavior of a black box storage system
        env = gym.make('FrozenLake-v0')
        #Initialize table with all zeros
        Q = np.zeros([env.observation_space.n,env.action_space.n])

        # Set learning parameters
        lr = .8
        y = .95
        num_episodes = 2000
        #create lists to contain total rewards and steps per episode
        #jList = []
        rList = []
        for i in range(num_episodes):
            #Reset environment and get first new observation
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            #The Q-Table learning algorithm
            while j < 99:
                j+=1
                #Choose an action by greedily (with noise) picking from Q table
                a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
                #Get new state and reward from environment
                s1,r,d,_ = env.step(a)
                #Update Q-Table with new knowledge
                Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
                rAll += r
                s = s1
                if d == True:
                    break
                #jList.append(j)
        rList.append(rAll)

        print ("Score over time: " +  str(sum(rList)/num_episodes))

        print("Final Q-Table Values")
        print(Q)

if __name__ == "__main__":


    #wars = WARS('TS')
    #print(wars.nextW())
    store = DistributedStorageSystem()
    print(store.Compute_PIC_PUA_Given_TC_TA(5, 1, 1, 0.001, .5, 1000))
    #store.tabularQLearning()
