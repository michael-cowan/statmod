import random as r
import matplotlib.pyplot as plt
import time

"""Example Genetic Algorithm (GA) that solves a goal string"""


class Chromosome(object):
    def __init__(self, text=None, text_length=None, textlim=(32, 126)):
        self.textlim = sorted(textlim)
        self.min_textlim, self.max_textlim = sorted(textlim)
        self.max_diff = self.max_textlim - self.min_textlim

        # set text string or make random string of given text_length
        assert text or text_length
        if not text:
            self.text = ''.join([chr(r.randint(*textlim))
                                 for i
                                 in xrange(text_length)
                                 ])
        else:
            self.text = text
        self.cost = 1e6

    def calc_cost(self, compare):
        assert len(compare) == len(self.text), "Compare to diff length str"
        cost = 0
        for let, com in zip(self.text, compare):
            val = abs(ord(let) - ord(com))
            cost += min(val, self.max_diff + 1 - val)
        self.cost = cost

    def mate(self, chrom):
        children = ['', '']
        for i in xrange(len(self.text)):
            c = 0 if r.random() < 0.5 else 1
            children[c] += self.text[i]
            children[c - 1] += chrom.text[i]
        return map(lambda t: Chromosome(t, textlim=self.textlim),
                   children)

    def mutate(self, prob):
        if r.random() < prob:
            i = r.randrange(len(self.text))
            orig = ord(self.text[i])
            jump = 1 if r.random() > 0.5 else -1

            jump *= r.randint(1, 6)

            newnum = orig + jump

            if newnum > self.max_textlim:
                n = self.min_textlim - 1 + (newnum % self.max_textlim)
                newnum = max(n, self.min_textlim)
            elif newnum < self.min_textlim:
                n = self.max_textlim + 1 - (self.min_textlim % newnum)
                newnum = min(n, self.max_textlim)

            newlett = chr(newnum)
            self.text = self.text[:i] + newlett + self.text[i+1:]
        return self


class Population(object):
    def __init__(self,
                 goal='Hello, World!',
                 size=100,
                 mutation_prob=0.3,
                 pct2kill=0.1
                 ):

        # keep track of n generations and gen stats
        self.diff = []
        self.generation = 0
        self.stats = []
        self.goal = goal
        self.length = len(self.goal)
        self.size = size
        self.mutation_prob = mutation_prob
        self.pct2kill = pct2kill
        self.members = []

        # get ascii number range based on goal
        ords = map(ord, list(self.goal))
        minmax = (min(ords), max(ords))

        for i in xrange(self.size):
            c = Chromosome(text_length=self.length, textlim=minmax)
            c.calc_cost(self.goal)
            self.members.append(c)

        self.gen_stats()

    def next_gen(self):
        self.generation += 1

        # sort the members by cost
        self.members = sorted(self.members, key=lambda i: i.cost)

        # remove the weakest
        rem_n = int(self.pct2kill * len(self.members))

        # ensure at least one gets removed
        if not rem_n:
            rem_n = 1

        self.members = self.members[:-rem_n]

        # top of remaining members mate to create new children
        used_pairs = []
        children = []
        cut = len(self.members)

        while len(children) < rem_n:
            m1 = r.randrange(cut)
            m2 = r.randrange(cut)
            while m1 == m2 or (m1, m2) in used_pairs:
                m1 = r.randrange(cut)
                m2 = r.randrange(cut)
            used_pairs.append((m1, m2))
            children += self.members[m1].mate(self.members[m2])

        self.members += children[:rem_n]

        # mutation
        self.members = map(lambda m: m.mutate(self.mutation_prob),
                           self.members)

        # calculate new costs
        for m in self.members:
            m.calc_cost(self.goal)

        # calc generation stats
        self.stats.append(self.gen_stats())

    def cutoff(self):
        top = min(self.members, key=lambda i: i.cost)
        return bool(top.cost <= 0)

    def gen_stats(self):
        self.members = sorted(self.members, key=lambda i: i.cost)
        # top = self.members[0].text
        low = self.members[0].cost
        # high = self.members[-1].cost
        # mean = sum([j.cost for j in self.members]) / float(self.size)
        self.diff.append(low)
        return [self.generation, low]

    def get_text(self):
        return [i.text for i in self.members]

    def run(self):
        start = time.time()
        while not self.cutoff():
            self.next_gen()
            print('\r%s' % self.members[0].text),
        return time.time() - start


def stats(goal='abcdefghijklmnopqrstuvwxyz0123456789|',
          ntimes=10):
    fig, ax = plt.subplots(figsize=(9, 7))
    print('\n'*100)
    tot = 0
    for i in xrange(ntimes):
        a = Population(goal=goal, size=50,
                       mutation_prob=0.05,
                       pct2kill=0.2)
        tot += a.run()
        ax.plot(range(a.generation + 1), a.diff)
    avg = tot / float(ntimes)
    ax.set_title('Average time: %.3f s (%i runs)' % (avg, ntimes))
    ax.set_xlabel('Generation')
    ax.set_ylabel('Minimum Cost')
    fig.text(0.4, 0.3, goal)
    fig.tight_layout()
    fig.show()
    return fig, ax

if __name__ == '__main__':
    stats()
