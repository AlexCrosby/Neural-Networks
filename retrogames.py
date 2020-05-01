import retro
import cv2
import numpy as np
from network import Network

cutout = 20


def process_img(obs):
    obs = cv2.resize(obs, (int(256 / 8), int(224 / 8)))
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('color_img.jpg', obs)
    # cv2.imshow("image", obs)
    # cv2.waitKey()
    # obs = np.ndarray.flatten(obs)
    obs = np.array(obs)

    obs = obs.reshape([1, 28, 32, 1])
    obs = net.predict(obs)

    action = [False] * 12
    obs = obs.tolist()
    j = obs.index(max(obs))
    action[j] = True
    return action


def run_population(pop, cutout):
    total = 0
    for p in pop:

        obs = env.reset()
        obs = process_img(obs)
        net.write_to_net(p)

        score = 0
        count = 0
        while True:
            env.render()
            obs, reward, done, info = env.step(obs)
            obs = process_img(obs)
            score += reward
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
    cutout = 400
    best = 4
    population = []
    env = retro.make(game='DonkeyKongCountry-Snes')

    net = Network()
    generation = 1

    for i in range(best ** 2):
        population.append(net.random_genome())
    while True:
        run_population(population, cutout)
        cutout *= 2
        population = gen_pop(population)
        generation += 1
    # action = env.action_space.sample()
    # print(env.action_space)
