import math
import numpy as np
import matplotlib.pyplot as plt

TYPE_0 = 0
TYPE_1 = 1
A = 10

# With constraint: x_i in [-5, 5]
def fitness_0(n, x):
    result = 0.0
    for i in range(n):
        result += x[i] * x[i]

    return result

# With constraint: x_i in [-5, 5]
def fitness_1(A, n, x):
    result = A * n
    for i in range(n):
        result += (x[i] * x[i] - A * math.cos(2 * math.pi * x[i]))

    return result


# fourmis
class fourmi:
    def __init__(self):
        self.x = []  # coor
        self.s_i = [] # site chasse
        self.NbEchec = 0 # nombre d'echec
        self.Plocal = 100 # P local
        self.local = 0.01 # r
        self.site = 0.1 # site

    def __eq__(self, other):
        self.x = other.x
        self.s_i = other.s_i
        self.NbEchec = other.NbEchec
        self.Plocal = other.Plocal
        self.local = other.local
        self.site = other.site

    def __repr__(self):
        return repr((self.s_i, self.x, self.NbEchec, self.Plocal, self.local, self.site))

def explore_init(nid, site, a_i, N):
    for i in range(N):
        temp = nid[i] + np.random.uniform(0-site, site)
        if temp > 5:
            temp = temp - 10
        elif temp < -5:
            temp = temp + 10
        a_i.s_i.append(temp)


def explore(s_i, local, a_i, N):
    result = []
    for i in range(N):
        a_i.x[i] = s_i[i] + np.random.uniform(0-local, local)
        if a_i.x[i] > 5:
            a_i.x[i] = a_i.x[i] - 10
        elif a_i.x[i] < -5:
            a_i.x[i] = a_i.x[i] + 10
        result.append(a_i.x[i])
    return result

# open a file
path = "tp.txt"
f = open(path, "w")


def api(N, type_function, Neval, ub, lb, numFourmi):

    # initiation of nid
    nid = []
    for i in range(N):
        nid.append(np.random.uniform(ub, lb))

    # init of varaibles
    best = 65536
    bestPos = []
    records = []
    t = 1
    Tmax = Neval
    Tdepl = 20
    a = []
    for i in range(numFourmi):
        a_i = fourmi()
        x = []
        for j in range(N):
            x.append(np.random.uniform(ub, lb))
        a_i.x = x
        a.append(a_i)
    i = 0

    # algo API
    while t <= Tmax:

        # pour chaque fourmi
        while i < numFourmi:

            # si le fourmi n'a pas de site de chasse
            if len(a[i].s_i) == 0:
                explore_init(nid, a[i].site, a[i], N)
                a[i].NbEchec = 0
            # sinon
            else:
                # explorer autour de la site de chasse
                p = explore(a[i].s_i, a[i].local, a[i], N)

                # calculer la valeur d'evaluation
                pEval = sEval = 0
                if type_function == TYPE_0:
                    pEval = fitness_0(N, p)
                    sEval = fitness_0(N, a[i].s_i)
                elif type_function == TYPE_1:
                    pEval = fitness_1(A, N, p)
                    sEval = fitness_1(A, N, a[i].s_i)

                # si mieux que la site de chasse, mettre a jour la site
                if pEval < sEval:
                    a[i].s_i = p
                    a[i].NbEchec = 0

                # sinon, nombre d'echec augmente par 1
                else:
                    a[i].NbEchec = a[i].NbEchec + 1
                    if a[i].NbEchec > a[i].Plocal:
                        a[i].s_i = []
                        a[i].NbEchec = 0
                # ecrire les coordonnes de la fourmi dans le fichier
                if len(a[i].x) == 2:
                    f.write(str(a[i].x[0]) + " " + str(a[i].x[1]) + " " + str(pEval / 50) + "\n")


            i = i+1

        # apres certaines iteration, re-situer le nid dans la meilleur site de chasse
        if t % Tdepl == 0:
            for k in range(numFourmi):
                if type_function == TYPE_0 and len(a[k].s_i) != 0:
                    pRes = fitness_0(N, a[k].s_i)
                elif type_function == TYPE_1 and len(a[k].s_i) != 0:
                    pRes = fitness_1(A, N, a[k].s_i)
                if best > pRes:
                    best = pRes
                    bestPos = a[k].s_i
            nid = bestPos
            print(best)
            print(bestPos)
            records.append(best)


            # liberer le memoire de la fourmi
            for m in range(numFourmi):
                a[m].s_i = []

        t = t+1
        i = 0



    return best, bestPos, records

if __name__ == '__main__':
    '''best, bestPos = api(2, 0, 5000, -5, 5, 50)
    print("best point " + str(bestPos))
    print("with value of " + str(best))'''

    results = []
    i = 0
    while (i < 1):
        best, bestPos, records = api(2, TYPE_0, 20000, -5, 5, 100)
        print("End of iteration " + str(i))
        print("best individual : " + str(bestPos))
        print("best fitness is : " + str(best))
        results.append(best)
        i = i + 1
    moyen = np.mean(results)
    ecart = np.std(results)
    print("La moyenne est: " + str(moyen))
    print("L'ecart type est: " + str(ecart))

    '''best, bestPos, records = api(30, TYPE_1, 20000, -5, 5, 100)
    x = range(len(records))
    plt.figure()
    plt.plot(x, records)
    plt.show()'''

    f.close()

