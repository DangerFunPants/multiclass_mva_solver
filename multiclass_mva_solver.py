
from collections import defaultdict
import pprint as pp
import copy

class Model_Spec:
    def __init__(self):
        # population vector
        self._ns = {}
        # demands :: c -> k -> \mathbb{R}
        # c \in Clients, k \in Servers
        self._demands = {}
        self._server_list = []
        # Think times (infinite server resources)
        # think :: c -> \mathbb{R}
        self._think = {}

    def add_server(self, name):
        self._server_list.append(name)
        for c_ident in self._demands.keys():
            # initialize demands to zero for new servers.
            self._demands[c_ident][name] = 0.0

    def add_customer_class(self, name, count, demands, think):
        """
        Demands must specify demand for a subset of the 
        servers in self._demands
        """
        self._ns[name] = count
        self._demands[name] = {}
        for server in self._server_list:
            self._demands[name][server] = demands.get(server, 0.0)
        self._think[name] = think
    
    def __str__(self):
        s = ( "Population Vector:\n"
            + pp.pformat(self._ns) + "\n"
            + "Demands\n"
            + pp.pformat(self._demands) + "\n"
            + "Think times: \n"
            + pp.pformat(self._think)
            )
        return s

    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        if key == 0:
            return len(self._ns)
        elif key == 1:
            return copy.deepcopy(self._demands)
        elif key == 2:
            return len(self._server_list)
        elif key == 3:
            return copy.deepcopy(self._think)
        elif key == 4:
            return copy.deepcopy(self._ns)
        else:
            raise IndexError("Attempted to unpack Model_Spec into more than four params")

class Model_Solution:
    def __init__(self, throughput, response_time, queue_lengths):
        self._throughputs = throughput
        self._response_times = response_time 
        self._queue_lengths = queue_lengths

    def throughput(self, class_name):
        return self._throughputs[class_name]

    def response_time(self, class_name):
        return sum(self._response_times[class_name])

    def detailed_response_time(self, class_name):
        return self._response_times[class_name]

    def queue_lengths(self, class_name):
        return self._queue_lengths[class_name]

    def idle_time(self, class_name):
        x = 1 / (sum(self._response_times[class_name]) + self._think[class_name])
        r = sum(self._response_times[class_name])
        u = r * x
        return (r / u) - r

    def __str__(self):
        s = ( "Throughput: \n"
            + pp.pformat(self._throughputs) + "\n"
            + "Response Times: \n"
            + pp.pformat(self._response_times)
            )
        return s

    def __repr__(self):
        return str(self)

def build_model(C, D, K, Z, ns):
    model = Model_Spec()
    for k in range(K):
        model.add_server(k)

    for c in range(C):
        model.add_customer_class(c, ns[c], D[c], Z[c])
    return model


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
                new_vector = copy.deepcopy(ns)
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
                print("X[%d]: %f" % (c, X[c]))
        
        Q = {}
        for k in range(K):
            Q[k] = sum([X[c] * R[c][k] for c in range(C)])

        return Q, X, R

def solve_mva_model(model):
    Q, X, R = multiclass_mva_solver(*model)
    return Model_Solution(X, R, Q)

def some_model():
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

def throughput(model_output):
    Q, X, R = model_output
    return X

def response_time(model_output):
    Q, X, R = model_output
    # R :: class -> resource -> response time
    return {c_i: sum(response_times.values()) 
                for c_i, response_times in R.items()}

def detailed_response_time(model_output):
    Q, X, R = model_output
    return R

def method_of_layers():
    # Initialize assuming no device contention
    # R_{g,k} = V_{g,k} * S_{g,k} for all G \in G_l, k \in K
    R = { 0: { 0: 0.0, 1: 0.0 } 
        , 1: { 0: 0.2, 1: 0.0 }
        , 2: { 0: 0.0, 1: 0.1 }
        }

def sw_model2():
    ns = [3]
    D = {0: {0: 0.0, 1: 0.0, 2: 0.3}
        } 
    Z = {0: 1.0}
    K = 3
    C = 1
    return (C, D, K, Z, ns)

# Clients:
# 0: Web sw process

# Servers:
# 0: Web cpu
# 1: db cpu
# 2: db sw process
def sw_model1():
    ns = [1]
    D = { 0: {0: 0.0, 1: 0.0, 2: 0.1}
        }
    Z = {0: 0.225468164794007486} 
    K = 3
    C = 1
    return (C, D, K, Z, ns)

# Clients:
# 0: Web sw process
# 1: Db sw process
# 
# Servers:
# 0: web cpu
# 1: db cpu
def hw_model():
    # Not sure what the number of customers should be?
    # Since the question states that everything executes synchronously 
    # I would assume one but that could most certianly be incorrect. 
    # 0 is web, 1 is db
    ns = [1, 1] 
    D = { 0: {0: 0.2, 1: 0.0}
        , 1: {0: 0.0, 1: 0.1}
        }
    Z = { 0: 0.125468164794007486
        , 1: 0.22546816479400753
        }
    K = 2
    C = 2
    return (C, D, K, Z, ns)

def main():
    # lqm()
    model = build_model(*hw_model())
    print(model)
    print(solve_mva_model(model))
    # print(multiclass_mva_solver(*hw_model()))

if __name__ == "__main__":
    main()
