
from collections import defaultdict
import pprint as pp

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
            return self.demands
        elif key == 2:
            return len(self._queueing_centers)
        elif key == 3:
            return self._think
        elif key == 4:
            return self._ns
        elif key == 5:
            return self._queueing_centers
        elif key == 6:
            return self._delay_centers
        else:
            raise IndexError("""Attempted to unpack Model_Spec 
                    into more than four params""")

    def get_demands(self):
        return self._demands

    def get_queueing_centers(self):
        return self._queueing_centers

    def get_delay_centers(self):
        return self._delay_centers

    demands= property(get_demands)
    queueing_centers = property(get_queueing_centers)
    delay_centers = property(get_delay_centers)

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

    @property
    def response_times(self):
        return self._response_times

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
        return {c_name: self.idle_time(c_name) for c_name in self.devices}
    
    @property
    def classes(self):
        return next(iter(self._response_times.values())).keys()

    @property
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
    for client_name, v in demands.items():
        for dev_name, demand in v.items():
            if dev_name in delay_centers:
                model.add_delay_center(dev_name)
            elif dev_name in queueing_centers:
                model.add_queueing_center(dev_name)
            else:
                print("Error building model. Unknown device name.")

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
                new_vector = dict(ns)
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
                numer = Z[c] + model.get_infinite_server_time(c)
                numer += sum(R[c].values())
                X[c] = ns[c] / numer
        
        Q = {}
        for k in device_names:
            Q[k] = sum([X[c] * R[c][k] for c in customer_classes])

        return Q, X, R

def solve_mva_model(model):
    Q, X, R = multiclass_mva_solver(model)
    return Model_Solution(X, R, Q, model._think)

def compute_abs_error(expected, actual):
    numer = abs(expected - actual)
    return numer / expected

def check_error_tolerance(hw_results, sw_models):
    abs_errors = []
    for sw_model in sw_models:
        for consumer, ds in sw_model.demands.items():
            for resource, demand in ds.items():
                if resource in sw_model.delay_centers:
                    hw_estimate = hw_results.response_times[consumer][resource]
                    error = compute_abs_error(hw_estimate, demand)
                    abs_errors.append(error)
    return abs_errors

# For now this method assumes that the sw models appear in order in the list. 
# This is easy to fix, construct a DAG where edges are clients consuming resources at a
# queueing center then perform a topological sort of the DAG. This will partition 
# the DAG into layers, each of which corresponds to a single S/W model.
def solve_lqm_sw_models(sw_models):
    results = []
    for idx in range(len(sw_models)-1):
        model_to_solve = sw_models[idx]
        model_result = solve_mva_model(model_to_solve)
        results.append(model_result)
        # Compute idle times for all s/w resources in level idx + 1
        idle_times = model_result.idle_times()
        for class_name, idle_time in idle_times.items():
            sw_models[idx+1].add_think_time(class_name, idle_time)
        
    return results

def solve_lqm_model(sw_models, hw_model, tolerance):
    """
    sw_models   :: [Model_Spec]
    hw_model    :: Model_Spec
    Note that sw_models contains all L software models even though only L-1 models
    are actually solved
    """
    while True:
        sw_model_results = solve_lqm_sw_models(sw_models)
        sw_proc_idle_times = {}
        for model_result in sw_model_results:
            for class_name, idle_time in model_result.idle_times().items():
                sw_proc_idle_times[class_name] = idle_time

        for class_name, idle_time in sw_proc_idle_times.items():
            hw_model.add_think_time(class_name, idle_time)
        hw_model_results = solve_mva_model(hw_model)
        abs_errors = check_error_tolerance(hw_model_results, sw_models)
        if max(abs_errors) < tolerance:
            break

        # Update demand of delay (H/W) resources in S/W modles
        hw_response_times = hw_model_results.response_times
        for sw_model in sw_models:
            for class_name, device_dict in sw_model.demands.items():
                for device_name, demand in device_dict.items():
                    if device_name in sw_model.delay_centers:
                        projected_demand = hw_response_times[class_name][device_name]
                        sw_model[class_name][device_name] = projected_demand


    return (sw_model_results, hw_model_results)

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
    
    sw_model_results, hw_model_results = solve_lqm_model(sw_models, hw_model, 0.1)
    pp.pprint(sw_model_results[0])

if __name__ == "__main__":
    main()
