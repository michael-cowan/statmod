import abc
import random
import copy
import numpy as np
import functools
import matplotlib.pyplot as plt
import time

"""Example Genetic Algorithm (GA) that solves a goal string"""
CENTER = 30


@functools.total_ordering
class BaseChromosome(abc.ABC):
    """
    Abstract Chromosome class
    Attributes to define in __init__:
    - genome (obj): the genome of the chromosome
    - call self.calc_score()

    Methods to define:
    - calc_score: sets self.score based on self.genome
    - replicate: initializes a new chromosome with a random genome

    Optional methods to define:
    - mate: mates genome with a second chromsome to produce
            two children chromosomes
    - mutate: randomly perturbs the genome of the chromosome (in place)
    """
    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        self._score = score

    @property
    def genome(self):
        return self._genome

    @genome.setter
    def genome(self, genome):
        """Always calc score when genome is set"""
        self._genome = genome
        self.calc_score()

    def __getitem__(self, i):
        return self.genome[i]

    def __len__(self):
        return len(self._genome)

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def __str__(self):
        return str(self.genome)

    def single_point_mate(self, chrom2):
        """
        Single point mating routine
        """
        i = random.randrange(0, len(self))
        new1, new2 = self[:i] + chrom2[i:], chrom2[:i] + self[i:]
        return [self.replicate(genome=new1),
                self.replicate(genome=new2)]

    def k_point_mate(self, chrom2, k=5):
        """
        K-point mating algorithm
        """
        points = sorted(random.sample(range(len(self)), k=k))
        points.append(len(self))

        # build new children
        new1, new2 = [None] * len(self), [None] * len(self)
        prev = 0
        parents = [self.genome.copy(), chrom2.genome.copy()]
        for p in points:
            new1[prev:p] = parents[0][prev:p]
            new2[prev:p] = parents[1][prev:p]
            # reverse parents list
            parents.reverse()

        return [self.replicate(genome=new1),
                self.replicate(genome=new2)]

    def mate(self, chrom2):
        """
        Default mating routine
        - Conserves gene counts
        """
        new1, new2 = [None] * len(self), [None] * len(self)
        i = random.randrange(0, len(self))
        new1[:i] = self[:i]
        new2[i:] = self[i:]
        for c in chrom2:
            if c not in new1:
                new1[new1.index(None)] = c
            else:
                new2[new2.index(None)] = c
        return [self.replicate(genome=new1),
                self.replicate(genome=new2)]

    def mutate(self):
        """
        Default mutatation routine
        - randomly swaps two genes
        """
        for n in range(max(len(self) // 50, 1)):
            i, j = random.sample(range(len(self)), 2)
            self.genome[i], self.genome[j] = self.genome[j], self.genome[i]

    @abc.abstractmethod
    def calc_score(self):
        """Calculates the score of the Chromosome"""

    @abc.abstractmethod
    def replicate(self):
        """Creates a new Chromosome with random genome"""

    def copy(self):
        return copy.deepcopy(self)


class SimpleTestChromosome(BaseChromosome):
    def __init__(self, length=5, genome=None):
        if genome is None:
            self.genome = list(range(length))
            random.shuffle(self.genome)
        else:
            self.genome = genome

    def calc_score(self):
        self.score = sum([abs(self.genome[i] - i)**2
                          for i in range(len(self.genome))])

    def replicate(self, **kwargs):
        return SimpleTestChromosome(length=len(self), **kwargs)


class StringTestChromosome(BaseChromosome):
    def __init__(self, compare, genome=None,
                 textlim=(32, 126)):
        self.compare = compare
        self.text_length = len(self.compare)

        self.textlim = sorted(textlim)
        self.min_textlim, self.max_textlim = sorted(textlim)
        self.max_diff = self.max_textlim - self.min_textlim

        # set text string or make random string of given text_length
        if genome is None:
            self.genome = ''.join([chr(random.randint(*textlim))
                                   for i in range(self.text_length)
                                   ])
        else:
            assert len(genome) == self.text_length
            self.genome = genome

    def calc_score(self):
        cost = 0
        for let, com in zip(self.genome, self.compare):
            val = abs(ord(let) - ord(com))
            cost += min(val, self.max_diff - val)
        self.score = cost

    def mate(self, chrom):
        children = ['', '']
        for i in range(len(self)):
            c = 0 if random.random() < 0.5 else 1
            children[c] += self[i]
            children[c - 1] += chrom[i]
        return [self.replicate(genome=t) for t in children]

    def mutate(self):
        i = random.randrange(len(self.genome))
        orig = ord(self.genome[i])
        jump = 1 if random.random() > 0.5 else -1

        jump *= random.randint(1, 6)

        newnum = orig + jump

        if newnum > self.max_textlim:
            n = self.min_textlim - 1 + (newnum % self.max_textlim)
            newnum = max(n, self.min_textlim)
        elif newnum < self.min_textlim:
            n = self.max_textlim + 1 - (self.min_textlim % newnum)
            newnum = min(n, self.max_textlim)

        newlett = chr(newnum)
        self.genome = self.genome[:i] + newlett + self.genome[i+1:]
        return self

    def replicate(self, **kwargs):
        """Creates a new chromosome with a random genome"""
        return StringTestChromosome(self.compare, textlim=self.textlim,
                                    **kwargs)


class Population(object):
    def __init__(self, chromosome, popsize=10, mute_pct=0.2, max_best=False,
                 random=False, prog_print_genome=True):
        """
        Generic Population Class that can simulate Genetic Algorithms
        - Manages a population of Chromosomes
        - call self.run to run a GA simulation
        - if random=True, self.run will conduct a random search

        Args:
        - chromosome (Chromosome): Chromosome instance which will be used to
                                   replicate more chromosomes during GA sim

        KArgs:
        - popsize (int): size of population
                         (Default: 50)
        - mute_pct (float): percentage of population to mutate each generation
                            (Default: 0.8)
        - random (bool): if True, self.run does a random search and
                         does not perform a GA optimization
                         (Default: False)
        - prog_print_genome (bool): if True, prints genome as part of
                                        progress line while running
                                        (Default: True)
        """
        self.chromosome = chromosome
        self.popsize = popsize

        # determine number of chromos that will be mutated each generation
        # popsize - 1 is max number of chromo objects that can be mutated
        self.n_mute = min(int(popsize * mute_pct), popsize - 1)
        self.mute_pct = self.n_mute / self.popsize

        # keep track of how many times the sim has been continued
        self.continued = 0

        self.random = random

        # determines whether best score is maximum (True) or minimum (False)
        self.max_best = max_best

        # population - list of chromosomes
        self._pop = []
        self.initialize_gen1()

        self.prog_print_genome = prog_print_genome

    def __getitem__(self, position):
        return self._pop[position]

    def __len__(self):
        return len(self._pop)

    def __repr__(self):
        return ('Population(%i Chromosome%s)'
                % (len(self), ['s', ''][len(self) == 1]))

    def __str__(self):
        return self.summ_results()

    def initialize_gen1(self):
        """
        Sets up Pop for a new GA simulation (generation 0)
        """
        self.initialize_pop()
        self.sort_pop()
        self.orig_best = self[0].score
        self.current_best = self.orig_best

        # track statistics
        self.stats = []
        self.update_stats()

        # track runtime
        self.runtime = 0

        # track generations
        self.generation = 1

        # keep track of whether a sim has been run
        self.has_run = False

    def initialize_pop(self):
        """
        Initialize population of nanoparticle Chromosome objects
        """
        # if random and stepping to next generation, keep current best
        if self.random and self._pop:
            self._pop = [self[0]]

        # create random structures for remaining popsize
        self._pop += [self.chromosome.replicate()
                      for i in range(self.popsize - len(self))]

    def roulette_mate(self):
        """
        Roulette Wheel selection algorithm that chooses mates
        based on their fitness
        - probability of chromosome <i> being selected:
          P_i = (fitness_i / sum of all fitnesses)
        - fitter chromosomes more likely to be selected, but not guaranteed
          to help mitigate lack of diversity in population
        """
        scores = np.array([-abs(p.score) for p in self])
        fitness = scores + scores.min()
        totfit = fitness.sum()
        probs = np.zeros(self.popsize)
        for i in range(self.popsize):
            probs[i:] += abs(fitness[i] / totfit)

        mates = []
        for n in range(self.popsize // 2):
            m = [self[np.where(probs > random.random())[0][0]]
                 for z in range(2)]
            mates += m[0].mate(m[1])

        # keep the previous minimum
        self._pop = [self[0]] + mates
        self._pop += [self.chromosome.replicate()
                      for z in range(self.popsize - len(self))]
        self._pop = self[:self.popsize]

    def step(self):
        """
        Wrapper method that takes GA to next generation
        - mates the population
        - mutates the population
        - calculates statistics of new population
        - resorts population and increments self.generation
        """
        if self.random:
            self.initialize_pop()
        else:
            # MATE
            self.roulette_mate()

            # MUTATE - does not mutate most fit nanoparticle
            for j in range(self.n_mute):
                self[random.randrange(1, self.popsize)
                     ].mutate()

        self.sort_pop()
        self.update_stats()
        self.current_best = self[0].score
        self.print_status(prog_print_genome=self.prog_print_genome)

        # increment generation
        self.generation += 1

    def print_status(self, end='\r', prog_print_genome=True):
        """
        Prints info on current generation of GA

        KArgs:
        - end (str): what to end the print statement with
                     - allows generations to overwrite on same line
                       during simulation
                     (Default: '\r')
        - prog_print_genome (bool): if True, prints genome as part of
                                        progress line while running
                                        (Default: True)
        """
        if prog_print_genome or self.prog_print_genome:
            val = f'{self[0].genome}\t'
        else:
            val = f'{str(self[0])}\t'
        val += f'{self[0].score}\t{self.generation:05d}'

        # format of string to be written to console during sim
        # update_str = ' SCORE: %.5f    %05i'
        # val = update_str % (self.stats[-1][0], self.generation)
        print(val.center(CENTER), end=end)

    def run(self, max_gens=-1, max_nochange=50, min_gens=-1):
        """
        Runs a GA simulation

        Kargs:
        - max_gens (int): maximum generations the GA will run
                          -1: the GA only stops based on <max_nochange>
                          (Default: -1)
        - max_nochange (int): specifies max number of generations GA will run
                              without finding a better NP
                              -1: GA will run to <max_gens>
                              (default: 50)
        - min_gens (int): minimum generations that the GA runs before checking
                          the max_nochange criteria
                          -1: no minimum
                          (Default: -1)

        Raises:
        - TypeError: can only call run for first GA sim
        - ValueError: max_gens and max_nochange cannot both equal -1
        """
        if self.has_run:
            raise TypeError("Simulation has already run. "
                            "Please use continue method.")
        elif max_gens == max_nochange == -1:
            raise ValueError("max_gens and max_nochange cannot both be "
                             "turned off (equal: -1)")

        self.max_gens = max_gens

        # GA will not continue if <max_nochange> generations are
        # taken without a change in minimum CE
        nochange = 0

        start = time.time()

        while self.generation != self.max_gens:
            if self[0].score == 0:
                break

            # step to next generation
            self.step()

            # track if there was a change (if min_gens has been passed)
            if (max_nochange and min_gens and
                    self.generation > min_gens):
                if self.stats[-1][0] == self.stats[-2][0]:
                    nochange += 1
                else:
                    nochange = 0

                # if no change has been made after <maxnochange>, stop GA
                if nochange == max_nochange:
                    break

        # print status of final generation
        self.print_status(end='\n',
                          prog_print_genome=self.prog_print_genome)

        # set max_gens to actual generations simulated
        self.max_gens = self.generation

        # capture runtime in seconds
        self.runtime += time.time() - start

        # convert stats to an array
        self.stats = np.array(self.stats)

        self.has_run = True

    def continue_run(self, max_gens=-1, max_nochange=50, min_gens=-1):
        """
        Used to continue GA sim from where it left off

        Kargs:
        max_gens (int): maximum generations the GA will run
                        -1: the GA only stops based on <max_nochange>
                        (Default: -1)
        max_nochange (int): specifies max number of generations GA will run
                            without finding a better NP
                            -1: GA will run to <max_gens>
                            (default: 50)
        - min_gens (int): minimum generations that the GA runs before checking
                          the max_nochange criteria
                          -1: no minimum
                          (Default: -1)
        """
        self.has_run = False
        self.stats = list(self.stats)
        self.run(max_gens=max_gens, max_nochange=max_nochange,
                 min_gens=min_gens)
        self.continued += 1

    def sort_pop(self):
        """
        Sorts population based on cohesive energy
        - lowest cohesive energy = most fit = first in list
        """
        self._pop = sorted(self,
                           reverse=self.max_best)

    def update_stats(self):
        """
        - Adds statistics of current generation to self.stats
        """
        self.sort_pop()
        s = np.array([i.score for i in self])
        self.stats.append([s[0],     # best score
                          s.mean(),  # mean score
                          s.std()])  # STD score

    def is_new_best(self):
        """
        Returns True if GA sim found new best score
        - based on stats matrix that is populated during sim
        """
        return self.orig_best != self.current_best

    def save(self, path):
        """
        Saves the Pop instance as a pickle

        Args:
        - path (str): path to save pickle file
                      - can include filename
        """
        # if path doesn't include a filename, make one
        if not path.endswith('.pickle'):
            fname = 'GA_sim.pickle'
            path = os.path.join(path, fname)

        # pickle self
        with open(path, 'wb') as fidw:
            pickle.dump(self, fidw, protocol=pickle.HIGHEST_PROTOCOL)

    def summ_results(self):
        """
        Creates string listing GA simulation stats and results info

        Returns:
        - (str): result string
        """
        # summary string to be printed
        cen = 30
        rjust = 15
        summ = []

        # population details
        summ.append(' Population Details '.center(cen, '-'))

        mute = 'Mute Pct:'.rjust(rjust) + ' {0:.0%}'.format(self.mute_pct)
        summ.append(mute)

        ps = 'Popsize:'.rjust(rjust) + ' %i' % self.popsize
        summ.append(ps)

        summ.append(''.center(cen, '-'))

        if self.has_run:
            summ.append(' GA Sim Results '.center(cen, '-'))

            best = 'Best:'.rjust(rjust) + ' %.5f' % self.stats[-1, 0]
            summ.append(best)

            mean = 'Mean:'.rjust(rjust) + ' %.3f' % self.stats[-1, 1]
            summ.append(mean)

            std = 'STD:'.rjust(rjust) + ' %.3f' % self.stats[-1, 2]
            summ.append(std)

            ngen = 'nGens:'.rjust(rjust) + ' %i' % self.max_gens
            summ.append(ngen)

            rt = 'Runtime:'.rjust(rjust) + ' %.2f s' % self.runtime
            summ.append(rt)

            summ.append(''.center(cen, '-'))

        return '\n'.join(summ)

    def plot_results(self, savepath=None, ax=None, title=None):
        """
        Method to create a plot of GA simulation
        - plots average, std deviation, and best score
          of the population at each step

        KArgs:
        - savepath (str): path and file name to save the figure
                          - if None, figure is not saved
                          (default: None)
        - ax (matplotlib.axis): if given, results will be plotted on axis
                                (Default: None)
        - title (str): title of plot if given
                       (Default: None)

        Returns:
        - (matplotlib.figure.Figure),
        - (matplotlib.axes._subplots.AxesSubplot): fig and ax objs
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 9))
            color = 'navy'
            fillcolor = 'lightblue'
            secondary = False
        else:
            fig = ax.figure
            color = 'red'
            fillcolor = 'pink'
            secondary = True

        # number of steps GA took
        steps = range(len(self.stats))

        # minimum, average, and std deviation scores of population at each step
        best = self.stats[:, 0]
        mean = self.stats[:, 1]
        std = self.stats[:, 2]

        # plot data - only give labels if not secondary

        # plot best score as a dotted line
        ax.plot(best, ':', color=color, label=['BEST', None][secondary])
        # light blue fill of one std deviation
        ax.fill_between(range(len(self.stats)), mean + std, mean - std,
                        color=fillcolor, label=['STD', None][secondary])

        # plot mean as a solid line and minimum as a dotted line
        ax.plot(mean, color=color, label=['MEAN', None][secondary])

        # create legend and title if necessary
        if not secondary:
            fs = 14
            ax.legend(ncol=3, fontsize=fs - 4, frameon=False)
            ax.set_ylabel('Score', fontsize=fs)
            ax.set_xlabel('Generation', fontsize=fs)

            if title is not None:
                ax.set_title(title)

            fig.tight_layout()

        # save figure if <savepath> was specified
        if savepath:
            fig.savefig(savepath)

        return fig, ax


def stats(goal='<test-text>---TESTinG?---|12343210|'):
    c = StringTestChromosome(goal)
    pop = Population(c, popsize=5)
    pop.run(max_nochange=0)
    print(pop)
    fig, ax = pop.plot_results()
    ax.set_title(f'{pop[0]}\n{goal}')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    stats()

    # # genome length
    # length = 10

    # # population size
    # popsize = 10

    # stc = SimpleTestChromosome(length=length)
    # pop = Population(stc, popsize=popsize, mute_pct=0.2)
    # pop.run(max_nochange=5000, max_gens=-1)
    # pop.plot_results()
    # plt.show()
