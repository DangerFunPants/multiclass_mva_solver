
from collections import defaultdict
import pprint as pp
from toposort import toposort

class Model_Spec:
    def __init__(self):
        # population vector
        self._ns = {}
        # demands :: c -> k -> \mathbb{R}
        # c \in Clients, k \in Servers
        # Demands for queueing and delay resources. Could namespace these two but whatevs
        self._demands = {}

        # The names of the resources are imposed by the network itself.
        # self._resources :: r -> Resource for r \in Servers
        self._resources = {}

        # Treat think independant of delay resources.
        # think :: c -> \mathbb{R}
        self._think = {}

    # def add_queueing_center(self, name):
    #     self._queueing_centers.add(name)

    # def add_delay_center(self, name):
    #     self._delay_centers.add(name)
    def add_resource(self, name, resource):
        self._resources[name] = resource

    def add_customer_class(self, name, count, demands, think):
        """
        Demands must specify demand for a subset of the 
        servers in self._demands
        """
        self._ns[name] = count
        self._demands[name] = {}
        for server in self._resources.keys():
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
            return len(self.queueing_centers)
        elif key == 3:
            return self._think
        elif key == 4:
            return self._ns
        elif key == 5:
            return self.queueing_centers
        elif key == 6:
            return self.delay_centers
        else:
            raise IndexError("""Attempted to unpack Model_Spec 
                    into more than seven params""")

    @property
    def demands(self):
        return self._demands

    @property
    def think(self):
        return self._think

    @property
    def queueing_centers(self):
        return [name for name, resc_description in self._resources.items()
                    if resc_description.service_type == "queueing_center"]

    @property
    def delay_centers(self):
        # return self._delay_centers
        return [name for name, resource in self._resources.items()
                    if resource.service_type == "delay_center"]

    def get_infinite_server_time(self, class_name):
        return sum([self._demands[class_name][inf_server]
                    for inf_server in self.delay_centers])

class Resource:
    def __init__(self, service_type, resource_type):
        """
        service_type    : Queueing center, delay_center, load dependent queueing center
        resc_type       : hw or software.
        """
        self._service_type = service_type
        self._resource_type = resource_type

    @property
    def resource_type(self):
        return self._resource_type
    
    @property
    def service_type(self):
        return self._service_type

class Model_Solution:
    def __init__(self, throughput, response_time, queue_lengths, think, demands):
        self._throughputs = throughput
        self._response_times = response_time 
        self._queue_lengths = queue_lengths
        self._think = think
        self._demands = demands

    def throughput(self, class_name):
        return self._throughputs[class_name]

    @property
    def response_times(self):
        return self._response_times

    @property
    def demands(self):
        return self._demands

    def queue_lengths(self, class_name):
        return self._queue_lengths[class_name]

    def idle_time(self, dev_name):
        x = sum([tput for c_name_prime, tput in self._throughputs.items()
            if self._response_times[c_name_prime][dev_name] != 0.0])
        r = sum([dev_dict[dev_name] for c_name, dev_dict in self._response_times.items()
                if dev_name in dev_dict])
        d = sum([dev_dict[dev_name] for c_name, dev_dict in self.demands.items()
                if dev_name in dev_dict])
        u = x * d
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
            + pp.pformat(dict(self._response_times))
            )
        return s

    def __repr__(self):
        return str(self)

def build_model(delay_centers, queueing_centers, demands, think, population_vector):
    model = Model_Spec()
    for client_name, v in demands.items():
        for dev_name, demand in v.items():
            if dev_name in delay_centers:
                resource = Resource("delay_center", "unspecified")
            elif dev_name in queueing_centers:
                resource = Resource("queueing_center", "unspecified")
            else:
                print("Error building model. Unknown device name.")
            model.add_resource(dev_name, resource)

    for client_name, d in demands.items():
        model.add_customer_class(client_name, population_vector[client_name], 
                d, think[client_name])

    return model

def calculate_initial_demand_estimates(sw_sub_models, model_index):
    """
    This method modifies the demands of the models in place.
    """
    if model_index == len(sw_sub_models) - 1:
        return {res_name: sum(demand_dict.values()) 
                for res_name, demand_dict
                in sw_sub_models[model_index].demands.items()}

    sw_res_demand_estimates = calculate_initial_demand_estimates(
            sw_sub_models, model_index+1)

    for res_name, demand_estimate in sw_res_demand_estimates.items():
        for client_name in sw_sub_models[model_index].demands.keys():
            sw_sub_models[model_index].demands[client_name][res_name] = demand_estimate

    return {res_name: sum(demand_dict.values())
            for res_name, demand_dict 
            in sw_sub_models[model_index].demands.items()}

def build_overall_model(sw_resources, hw_resources, demands, think, population_vector):
    model = Model_Spec()
    # Create a graph of all the software resources in the QNM where nodes 
    # are resources and directed edges indicate that one resource on which
    # the edge is incident is receiving service from the resource on which
    # the edge terminates.
    sw_resource_graph = defaultdict(set)
    for sw_resource in sw_resources:
        egress_edges = demands.get(sw_resource, {})
        edges_to_sw_resources = [dest_node for dest_node 
                                    in egress_edges.keys()
                                    if dest_node in sw_resources]
        for dest_node in edges_to_sw_resources:
            sw_resource_graph[sw_resource].add(dest_node)

    sort_result = list(toposort(sw_resource_graph))[::-1]
    assert(len(sort_result) > 1) # emphasis on layered.

    # Now construct software submodels in accordance with the partitions
    sw_sub_models = []
    for idx in range(len(sort_result)-1):
        sw_clients = sort_result[idx]
        sw_servers = sort_result[idx+1]
        ns = {c_name: population_vector[c_name] for c_name in sw_clients}
        sub_model_think = {c_name: think[c_name] for c_name in sw_clients}
        sub_model_demands = {c_name: demands[c_name] for c_name in sw_clients}
        all_resources = {res_name 
                            for res_dict in sub_model_demands.values()
                            for res_name in res_dict.keys()}
        queueing_centers = sw_servers
        delay_centers = all_resources.difference(queueing_centers)
        sw_sub_models.append(build_model(delay_centers, queueing_centers, 
            sub_model_demands, sub_model_think, ns))

    # Last submodel has no outgoing edges
    sw_clients = sort_result[-1]
    ns = {c_name: population_vector[c_name] for c_name in sw_clients} 
    sub_model_think = {c_name: think[c_name] for c_name in sw_clients}
    sub_model_demands = {c_name: demands[c_name] for c_name in sw_clients}
    delay_centers = {res_name
                        for res_dict in sub_model_demands.values()
                        for res_name in res_dict.keys()}
    queueing_centers = set()
    sw_sub_models.append(build_model(delay_centers, queueing_centers,
        sub_model_demands, sub_model_think, ns))

    calculate_initial_demand_estimates(sw_sub_models, 0)

    # Now construct the hardware submodel
    hw_model_demands = defaultdict(dict)
    for client_name, demand_dict in demands.items():
        for server_name, demand in demand_dict.items():
            if server_name in hw_resources: 
                hw_model_demands[client_name][server_name] = demand
    ns = {client_name: population_vector[client_name] 
            for client_name in hw_model_demands.keys()}
    hw_model_think = {client_name: think[client_name]
                        for client_name in hw_model_demands.keys()}
    queueing_centers = hw_resources
    delay_centers = set()
    hw_model = build_model(delay_centers, queueing_centers, 
            hw_model_demands, hw_model_think, ns)
    return (sw_sub_models, hw_model)

def multiclass_mva_solver(model):
    def all_zeros(ns):
        return len([v for v in ns.values() if v != 0]) == 0
    
    # need "with" keyword, that would be cool maybe...
    C, D, K, Z, ns, queueing_centers, delay_centers = model
    # This will most assuredly break...
    customer_classes = list(ns.keys())
    device_names = queueing_centers

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
    return Model_Solution(X, R, Q, model.think, model.demands)

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

def solve_lqm_sw_models(sw_models):
    """
    This method assumes that sw_models is ordered topologically 
    based on edges between software resources.
    """
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
    def lqm_model():
        population_vector = { "clients": 3
                            , "web_server": 1
                            , "db_server": 1
                            }
        demands = { "clients": {"web_server": 0.0}
                  , "web_server": {"web_server_cpu": 0.2, "db_server": 0.0}
                  , "db_server": {"db_server_cpu": 0.1}
                  }
        think = { "clients": 1.0
                , "web_server": 0.0
                , "db_server": 0.0
                }
        hw_resources = {"web_server_cpu", "db_server_cpu"}
        sw_resources = {"clients", "web_server", "db_server"}
        return (sw_resources, hw_resources, demands, think, population_vector)

    sw_models, hw_model = build_overall_model(*lqm_model())
    sw_model_results, hw_model_results = solve_lqm_model(sw_models, hw_model, 0.1)
    pp.pprint(sw_model_results[0])

if __name__ == "__main__":
    main()
