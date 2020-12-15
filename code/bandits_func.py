from scipy.stats import bernoulli
import numpy as np

#класс стохастических бандитов с бернулиевским распределением выигрышей

class BernoulliBandit:
  #принимает на вход список из K>=2 чисел из [0,1]
    def __init__(self, means):
        self.means = means
        self.sum_regret = 0
        new_means = np.array(means)
        self.best_arm = new_means.argmax()
    
  #возвращает число ручек
    def K(self):
        return len(self.means)

  #принимает параметр 0 <= a <= K-1 и возвращает реализацию случайной величины 
  #X c P[X=1] равной среднему значению выигрыша ручки a+1
    def pull(self, a):
        reward_a = bernoulli.rvs(p=self.means[a])
        self.sum_regret = self.sum_regret + bernoulli.rvs(p=self.means[self.best_arm]) - reward_a
        return reward_a

  #возвращает текущее значение регрета
    def regret(self):
        return self.sum_regret
        
#Алгоритм "Следуй за лидером": сыграть каждое действие 1 раз, затем играть действие с макисмальным средним выигрышем.        
#Вход - бандит(объект класса BernoulliBandit), общее количество игр. 
def FollowTheLeader(bandit, T):
    K = bandit.K()
    mean_rewards = np.zeros(K)
    num_pulling = np.ones(K)
    for t in range(T):
        if t < K:
            mean_rewards[t] = bandit.pull(t)
        if t >= K:
            best_arm = mean_rewards.argmax()
            current_reward = bandit.pull(best_arm)
            mean_rewards[best_arm] = (num_pulling[best_arm] * mean_rewards[best_arm] + current_reward) / (num_pulling[best_arm] + 1)
            num_pulling[best_arm] += 1

#Алгоритм "Explore First": сыграть каждое действие k раз, затем играть действие с максимальным средним выигрышем.
#Вход - бандит(объект класса BernoulliBandit), общее количество игр, число k.
           
def ExploreFirst(bandit, T, k):
  #k-число раз, сколько раз мы играем каждую ручку до exploitation фазы
    K = bandit.K()
    mean_rewards = np.zeros(K)
    num_pulling = np.zeros(K)
    #k = round((np.log(T) ** (1/3)) * T ** (2/3))
    for i in range(K):
        for _ in range(k):
            mean_rewards[i] += bandit.pull(i)
            num_pulling[i] += 1
    mean_rewards = np.array([j / k for j in mean_rewards])
    T -= k*K 
    for t in range(T):
        best_arm = mean_rewards.argmax()
        current_reward = bandit.pull(best_arm)
        mean_rewards[best_arm] = (num_pulling[best_arm] * mean_rewards[best_arm] + current_reward) / (num_pulling[best_arm] + 1)
        num_pulling[best_arm] += 1

#Алгоритм "EGreedy": с вероятность E играем случайную ручку, с вероятность 1 - E играем ручку с максимальным средним выигрышем. 
#Вход - бандит(объект класса BernoulliBandit), общее количество игр, значение E.
        
def EGreedy(bandit, T, e):
    K = bandit.K()
    mean_rewards = np.zeros(K)
    num_pulling = np.zeros(K)
    for t in range(T):
        coin_result = bernoulli.rvs(p=e)
        if coin_result == 1:
            arm = np.random.choice([j for j in range(K)])
            reward_arm = bandit.pull(arm)
            mean_rewards[arm] = (mean_rewards[arm] * num_pulling[arm] + reward_arm) / (num_pulling[arm] + 1)
            num_pulling[arm] += 1
        else:
            best_arm = mean_rewards.argmax()
            current_reward = bandit.pull(best_arm)
            mean_rewards[best_arm] = (num_pulling[best_arm] * mean_rewards[best_arm] + current_reward) / (num_pulling[best_arm] + 1)
            num_pulling[best_arm] += 1
 
#Алгоритм "SuccessiveElimination": играем случайную активную ручку; изначально все ручки активные, если же нижнее значение одной ручки выше верхнего значения другой, то вторая ручка исключается из списка активных ручек. 
#Вход - бандит(объект класса BernoulliBandit), общее количество игр. 
            
def SuccessiveElimination(bandit, T):
    K = bandit.K()
    mean_rewards = np.zeros(K)
    num_pulling = np.ones(K)
    upper_bounds = np.zeros(K)
    bottom_bounds = np.zeros(K)
    active_arms = np.array(range(K))
    for i in range(K):
        mean_rewards[i] = bandit.pull(i)
        radius = 0.8 * (2 * np.log(T) / num_pulling[i]) ** 0.5
        upper_bounds[i] = mean_rewards[i] + radius
        bottom_bounds[i] = mean_rewards[i] - radius
    counter = T-K
    while counter > 0:
        max_bottom_bound = -10000
        for i in active_arms:
            if bottom_bounds[i] > max_bottom_bound:
                max_bottom_bound = bottom_bounds[i]
        new_active_arms = []
        for i in active_arms:
            if upper_bounds[i] > max_bottom_bound:
                new_active_arms.append(i)
        active_arms = np.array(new_active_arms)
        for i in active_arms:
            current_reward = bandit.pull(i)
            mean_rewards[i] = (num_pulling[i] * mean_rewards[i] + current_reward) / (num_pulling[i] + 1)
            num_pulling[i] += 1
            radius = 0.8 * (2 * np.log(T) / num_pulling[i]) ** 0.5
            upper_bounds[i] = mean_rewards[i] + radius
            bottom_bounds[i] = mean_rewards[i] - radius
        counter -= len(active_arms)

#Алгоритм "UCB": играем каждую ручку по 1 разу, затем играем ручку с максимальным верхним значениям.
#Вход - бандит(объект класса BernoulliBandit), общее количество игр.
        
def UCB(bandit, T):
    K = bandit.K()
    mean_rewards = np.zeros(K)
    num_pulling = np.ones(K)
    upper_bounds = np.zeros(K)
    for i in range(K):
        mean_rewards[i] = bandit.pull(i)
        upper_bounds[i] = mean_rewards[i] + (2 * np.log(T) / num_pulling[i]) ** 0.5
    for i in range(T-K):
        best_arm = upper_bounds.argmax()
        current_reward = bandit.pull(best_arm)
        mean_rewards[best_arm] = (num_pulling[best_arm] * mean_rewards[best_arm] + current_reward) / (num_pulling[best_arm] + 1)
        num_pulling[best_arm] += 1
        upper_bounds[best_arm] = mean_rewards[best_arm] + 0.8 * (2 * np.log(T) / num_pulling[best_arm]) ** 0.5
