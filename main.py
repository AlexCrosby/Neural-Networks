from network import Network
import gym
import numpy as np


def run_population(pop):
    total = 0
    for p in pop:
        #   env.render()
        observation = env.reset()
        inputs = np.array([observation])
        net.write_to_net(p)
        action = (net.predict(inputs))
        score = 0
        count = 0
        while True:
            observation, reward, done, info = env.step(action)
            score += reward
            inputs = np.array([observation])
            action = (net.predict(inputs))
            count += 1
            if done or count > 400:
                p.score = score
                total += score
                break

    print(total / (best ** 2))


def gen_pop(pop):
    new_population = []
    pop.sort(reverse=True)
    pop = pop[:best]
    for p in pop:
        for q in pop:
            new_population.append(p.cross(q))
    return new_population


def play_best(p):
    net.write_to_net(p)
    observation = env.reset()
    inputs = np.array([observation])
    action = (net.predict(inputs))
    while True:
        env.render()
        observation, reward, done, info = env.step(action)
        inputs = np.array([observation])
        action = (net.predict(inputs))
        if done:
            break


if __name__ == '__main__':
    best = 10
    population = []
    env = gym.make('LunarLanderContinuous-v2')

    net = Network()
    generation = 1

    for i in range(best ** 2):
        population.append(net.random_genome())
    while True:
        run_population(population)
        population = gen_pop(population)
        generation += 1
    # action = env.action_space.sample()
    # print(env.action_space)
