import time


class MultiArmedBandit:
    '''
    This multi-armed bandit implementation is based on the following sources.

    Sources:

    (1)
    Title: Chapter 9: Applications to Computing, 9.8: Multi-Armed Bandits 
    (From “Probability & Statistics with Applications to Computing” by Alex
    Tsun)
    Author: Alex Tsun
    URL: https://web.stanford.edu/class/archive/cs/cs109/cs109.1218/files/student_drive/9.8.pdf
    Date accessed: November 20, 2023

    (2)
    Title: Multi-arm Bandits, The simplest reinforcement learning problem
    Author: Doina Precup
    URL: https://www.cs.mcgill.ca/~dprecup/courses/RL/Lectures/2-bandits-2019.pdf
    Date published: 2019
    Date accessed: November 20, 2023
    '''
    def __init__(self, n_arms, init_pulls):
        self.n_arms = n_arms
        self.init_pulls = init_pulls
        self.arm_pulls = [0] * self.n_arms
        self.est_exp_reward = [0] * self.n_arms
        self.run(self.init_pulls)

    def choose_arm(self):
        '''
        Selects an arm of the multi-armed bandit to pull
        '''
        raise NotImplementedError(
            'Implement a function to select an arm of the multi-arm bandit')

    def pull_arm(self, arm):
        '''
        Pulls the given arm of the multi-armed bandit and receives an outcome
        '''
        raise NotImplementedError(
            'Implement a function to pull a given arm of the multi-arm bandit '
            + 'and produce some outcome')

    def reward(self, arm, outcome):
        '''
        Calculates the reward of a given outcome
        '''
        raise NotImplementedError(
            'Implement a function to calculate the reward of a given outcome')
    
    def new_exp_reward(self, arm, reward):
        '''
        Calculated the new estimated expected reward of pulling a given arm of
        the multi-arm bandit
        '''
        raise NotImplementedError(
            'Implement a function to calculate the new estimated expected '
            + 'reward of pulling a given arm of the multi-arm bandit')
    
    def process_result(self, arm, outcome, reward):
        '''
        Performs additional processing on the result, should the use case
        require it

        Default to nothing, can be overridden should the use case require it
        '''
        pass

    def _print_results(self):
        print(f'Arm pulls: {self.arm_pulls}')
        print('Estimated expected reward for each arm: '
                + f'{self.est_exp_reward}\n')

    def run(self, n_iters=1e6, print_iters=100, time_limit=3600):
        start_time = time.time()
        for iter in range(n_iters):
            # pull an arm to get some outcome
            arm = self.choose_arm()
            outcome = self.pull_arm(arm)
            self.arm_pulls[arm] += 1

            # calculate reward of the outcome, updated expected reward for
            # pulled arm
            reward = self.reward(arm, outcome)
            self.est_exp_reward[arm] = self.new_exp_reward(arm, reward)

            # process the result of the arm pull
            self.process_result(arm, outcome, reward)

            if (iter + 1) % print_iters == 0:
                print(f'Iteration: {iter + 1}')
                self._print_results()

            # check if time limit has been exceeded
            time_elapsed = time.time() - start_time
            if time_elapsed >= time_limit:
                print('Time limit reached, terminating algorithm')
                self._print_results()
                break
        print('Number of iterations reached, terminating algorithm')
        self._print_results()
