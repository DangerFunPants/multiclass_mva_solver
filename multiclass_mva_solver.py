
from collections import defaultdict
import pprint as pp
import copy

class Model_Spec:
    def __init__(self):
        # population vector
        self._ns = {}
        # demands :: c -> k -> \mathbb{R}
        # c \in Clients, k \in Servers
        # Demands for queueing and delay resources. Could namespace these two but whatevs
        self._demands = {}

        # Need to differentiate between dealy and queueing centers
        self._queueing_centers = set()
        self._delay_centers = set()

        # Treat think independant of delay resources.
        # think :: c -> \mathbb{R}
        self._think = {}

    # deprecated
    def add_server(self, name):
        for c_ident in self._demands.keys():
            # initialize demands to zero for new resources.
            self._demands[c_ident][name] = 0.0

    def add_queueing_center(self, name):
        self._queueing_centers.add(name)

    def add_delay_center(self, name):
        self._delay_centers.add(name)

    def add_customer_class(self, name, count, demands, think):
        """
        Demands must specify demand for a subset of the 
        servers in self._demands
        """
        self._ns[name] = count
        self._demands[name] = {}
        for server in self._queueing_centers.union(self._delay_centers):
            self._demands[name][server] = demands.get(server, 0.0)
        self._think[name] = think

    def add_think_time(self, name, time):
        self._think[name] += time
    
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
            return len(self._queueing_centers)
        elif key == 3:
            return copy.deepcopy(self._think)
        elif key == 4:
            return copy.deepcopy(self._ns)
        elif key == 5:
            return self._queueing_centers
        elif key == 6:
            return self._delay_centers
        else:
            raise IndexError("Attempted to unpack Model_Spec into more than four params")

    def get_demands(self):
        return self._demands

    def get_queueing_centers(self):
        return self._queueing_centers

    demands= property(get_demands)
    queueing_centers = property(get_queueing_centers)

    def get_infinite_server_time(self, class_name):
        return sum([self._demands[class_name][inf_server]
                    for inf_server in self._delay_centers])

class Model_Solution:
    def __init__(self, throughput, response_time, queue_lengths, think):
        self._throughputs = throughput
        self._response_times = response_time 
        self._queue_lengths = queue_lengths
        self._think = think

    def throughput(self, class_name):
        return self._throughputs[class_name]

    def response_time(self, class_name):
        return sum(self._response_times[class_name])

    def detailed_response_time(self, class_name):
        return self._response_times[class_name]

    def get_response_times(self):
        return self._response_times

    response_times = property(get_response_times)

    def queue_lengths(self, class_name):
        return self._queue_lengths[class_name]

    def idle_time(self, dev_name):
        x = sum([tput for c_name_prime, tput in self._throughputs.items()
            if self._response_times[c_name_prime][dev_name] != 0.0])
        r = sum([dev_dict[dev_name] for c_name, dev_dict in self._response_times.items()
                if dev_dict[dev_name] != 0.0])
        u = r * x
        if u == 0:
            return 0.0
        return (r / u) - r

    def idle_times(self):
        return {c_name: self.idle_time(c_name) for c_name in self.devices()}
    
    def classes(self):
        return next(iter(self._response_times.values())).keys()

    def devices(self):
        return next(iter(self._response_times.values())).keys()

    def __str__(self):
        s = ( "Throughput: \n"
            + pp.pformat(self._throughputs) + "\n"
            + "Response Times: \n"
            + pp.pformat(self._response_times)
            )
        return s

    def __repr__(self):
        return str(self)

def build_model(delay_centers, queueing_centers, demands, think, population_vector):
    model = Model_Spec()
    # for client_name, v in D.items():
    #     for dev_name, demand in v.items():
    #         model.add_server(dev_name)

    for client_name, v in demands.items():
        for dev_name, demand in v.items():
            if dev_name in delay_centers:
                model.add_delay_center(dev_name)
            elif dev_name in queueing_centers:
                model.add_queueing_center(dev_name)
            else:
                print("Error building model. Unknown device name.")

    # for client_name, d in D.items():
    #     model.add_customer_class(client_name, ns[client_name], D[client_name], Z[client_name])
    for client_name, d in demands.items():
        model.add_customer_class(client_name, population_vector[client_name], 
                d, think[client_name])

    return model


def multiclass_mva_solver(model):
    def all_zeros(ns):
        return len([v for v in ns.values() if v != 0]) == 0
    
    # need "with" keyword, that would be cool maybe...
    C, D, K, Z, ns, queueing_centers, delay_centers = model
    # This will most assuredly break...
    customer_classes = list(ns.keys())
    device_names = model.queueing_centers

    if all_zeros(ns):
        return ({k: 0 for k in device_names}, {}, {})
    else:
        R = defaultdict(dict)
        for c in customer_classes:
            if ns[c] == 0:
                R[c] = {k: 0 for k in device_names}
            else:
                new_vector = copy.deepcopy(ns)
                new_vector[c] -= 1
                next_iter_model = build_model(delay_centers, queueing_centers,
                        D, Z, new_vector)
                Q, _, _ = multiclass_mva_solver(next_iter_model)
                for k in device_names:
                    R[c][k] = D[c][k] * (1 + Q[k])

        X = {}
        for c in customer_classes:
            if ns[c] == 0:
                X[c] = 0
            else:
                X[c] = ns[c] / (Z[c] + model.get_infinite_server_time(c) + sum(R[c].values()))
        
        Q = {}
        for k in device_names:
            Q[k] = sum([X[c] * R[c][k] for c in customer_classes])

        return Q, X, R

def solve_mva_model(model):
    Q, X, R = multiclass_mva_solver(model)
    
    return Model_Solution(X, R, Q, model._think)

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

# def sw_model2():
#     ns = [3]
#     D = {0: {0: 0.0, 1: 0.0, 2: 0.3}
#         } 
#     Z = {0: 1.0}
#     K = 3
#     C = 1
#     return (C, D, K, Z, ns)

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

def compute_abs_error(expected, actual):
    numer = abs(expected - actual)
    return numer / expected

def check_error_tolerance(hw_results, sw_models):
    # each software resource will have a response time for a subset
    # of the devices present in the H/W model. For each of these
    # response times, traverse the list of S/W models, find the error
    # and report it.
    for sw_model in sw_models:
        for class_name, demand_dict in sw_model.demands.items():
            for device_name, demand in demand_dict.items():
                if (class_name in hw_results.response_times and
                    device_name in hw_results.response_times[class_name]):
                    hw_time = hw_results.response_times[class_name][device_name]
                    assumed_time = demand
                    print(class_name, device_name)
                    print("Current: ", hw_time)
                    print("Assumed: ", assumed_time)
                    error = compute_abs_error(hw_time, assumed_time)
                    print("Error: ", error)

# For now this method assumes that the sw models appear in order in the list. 
# This is easy to fix, construct a DAG where edges are clients consuming resources at a
# queueing center then perform a topological sort of the DAG. This will partition the DAG
# into layers, each of which corresponds to a single S/W model.
def solve_lqm_sw_models(sw_models):
    results = {}
    for idx in range(len(sw_models)-1):
        model_to_solve = sw_models[idx]
        model_result = solve_mva_model(model_to_solve)
        # Compute idle times for all s/w resources in level idx + 1
        idle_times = model_result.idle_times()
        for class_name, idle_time in idle_times.items():
            results[class_name] = idle_time
            sw_models[idx+1].add_think_time(class_name, idle_time)
        
    return results

def solve_lqm_model(sw_models, hw_model, tolerance):
    """
    sw_models :: [Model_Spec]
    hw_models :: [Model_Spec]
    Note that sw_models contains all (L) software models even though only L-1 models
    are actually solved
    """

    while True:
        sw_proc_idle_times = solve_lqm_sw_models(sw_models)
        for class_name, idle_time in sw_proc_idle_times.items():
            hw_model.add_think_time(class_name, idle_time)
        hw_model_results = solve_mva_model(hw_model)
        check_error_tolerance(hw_model_results, sw_models)
        break

    return hw_model_results

def main():
    def sw_model2():
        ns = {"clients": 3}
        D = {"clients": {"web_server": 0.3}
            } 
        Z = {"clients": 1.0}
        delay_centers = {}
        queueing_centers = {"web_server"}
        return (delay_centers, queueing_centers, D, Z, ns)

    def sw_model1():
        ns = {"web_server": 1}
        D = { "web_server": {"db_server": 0.1, "web_server_cpu": 0.2}
            }
        Z = {"web_server": 0.0}
        delay_centers = {"web_server_cpu"}
        queueing_centers = {"db_server"}
        return (delay_centers, queueing_centers, D, Z, ns)

    def sw_model0():
        ns = {"db_server": 1}
        D = { "db_server": {"db_server_cpu": 0.1}
            }
        Z = {"db_server": 0.0}
        delay_centers = {"db_server_cpu"}
        queueing_centers = set()
        return (delay_centers, queueing_centers, D, Z, ns)

    def hw_model():
        # Not sure what the number of customers should be?
        # Since the question states that everything executes synchronously 
        # I would assume one but that could most certianly be incorrect. 
        # 0 is web, 1 is db
        # ns = [1, 1] 
        ns = { "web_server": 1
             , "db_server": 1
             }
        D = { "web_server": {"web_server_cpu": 0.2, "db_server_cpu": 0.0}
            , "db_server" : {"web_server_cpu": 0.0, "db_server_cpu": 0.1}
            }
        Z = { "web_server": 0.0
            , "db_server": 0.0
            }
        delay_centers = {"web_server", "db_server"}
        queueing_centers = {"web_server_cpu", "db_server_cpu"}
        return (delay_centers, queueing_centers, D, Z, ns)

    sw_models = [ build_model(*sw_model2())
                , build_model(*sw_model1())
                , build_model(*sw_model0())
                ]
    hw_model = build_model(*hw_model())
    model_results = solve_lqm_model(sw_models, hw_model, 0.1)
    print(model_results)

if __name__ == "__main__":
    main()
