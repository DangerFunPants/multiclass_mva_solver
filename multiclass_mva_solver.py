
from collections import defaultdict
import pprint as pp

def multiclass_mva_solver(C, D, K, Z, ns):
    def all_zeros(ns):
        return len([v for v in ns if v != 0]) == 0

    if all_zeros(ns):
        return ({k: 0 for k in range(K)}, {}, {})
    else:
        R = defaultdict(dict)
        for c in range(C):
            if ns[c] == 0:
                R[c] = {k: 0 for k in range(K)}
            else:
                new_vector = list(ns)
                new_vector[c] -= 1
                Q, _, _ = multiclass_mva_solver(C, D, K, Z, new_vector)
                for k in range(K):
                    R[c][k] = D[c][k] * (1 + Q[k])

        X = {}
        for c in range(C):
            if ns[c] == 0:
                X[c] = 0
            else:
                X[c] = ns[c] / (Z[c] + sum(R[c].values()))
        
        Q = {}
        for k in range(K):
            Q[k] = sum([X[c] * R[c][k] for c in range(C)])

        return Q, X, R

def main():
    ns = [3, 1]
    D = { 0: {0: 0.105, 1: 0.180, 2: 0.000}
        , 1: {0: 0.375, 1: 0.480, 2: 0.240}
        }
    Z = { 0: 0.0
        , 1: 0.0
        }
    K = 3
    C = 2
    res = multiclass_mva_solver(C, D, K, Z, ns)
    pp.pprint(res)

if __name__ == "__main__":
    main()
