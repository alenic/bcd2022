'''
Layers monitor
'''
# edited by Alessandro Nicolosi - https://github.com/alenic



class ForwardMonitor:
    def __init__(self, model, verbose=False):
        self.model = model
        self.verbose = verbose

        self.layer = {}
        self.handle = {}

    def add_layer(self, name, alias=None, htype="output", detach=False):
        self.layer_reached = False
        root_name = ""
        self._explore_and_hook(
            self.model, root_name, name, layer_alias=alias, htype=htype, detach=detach
        )
        if not self.layer_reached:
            raise ValueError("WARNIG! No " + name + " founded!")
        
    def get_layer(self, layer_name):
        if layer_name not in self.layer:
            raise ValueError(f"{layer_name} does not exits")
        return self.layer[layer_name]

    def remove_layer(self, layer_name):
        self.handle[layer_name].remove()
        del self.handle[layer_name]
        del self.layer[layer_name]

    def _register_forward_hook(self, module, name, htype="output", detach=False):
        def hook_output(model, input_, output):
            if detach:
                self.layer[name] = output.detach()
            else:
                self.layer[name] = output

        def hook_input(model, input_, output):
            if detach:
                self.layer[name] = input_[0].detach()
            else:
                self.layer[name] = input_[0]

        if htype == "input":
            self.handle[name] = module.register_forward_hook(hook_input)
        elif htype == "output":
            self.handle[name] = module.register_forward_hook(hook_output)
        else:
            raise ValueError("htype must be 'input' or 'output")

    def _explore_and_hook(
        self, model, name, layer_name, layer_alias=None, htype="output", detach=False
    ):
        for module_name, module in model.named_children():
            if self.layer_reached:
                break

            complete_name = name + "." + module_name
            if self.verbose:
                print(f"{complete_name} -> {type(module)}")

            if len(list(module.children())) == 0:
                if complete_name == "." + layer_name:
                    if self.verbose:
                        print(f"layer '{layer_name}' founded!")
                    self.layer_reached = True
                    if layer_alias is None:
                        self._register_forward_hook(module, layer_name, htype, detach)
                    else:
                        self._register_forward_hook(module, layer_alias, htype, detach)
                    break

            self._explore_and_hook(
                module, name + "." + module_name, layer_name, layer_alias, htype, detach
            )

def print_layers(model, name=""):
    for module_name, module in model.named_children():
        complete_name = name + "." + module_name

        print(f"{complete_name} -> {type(module)}")
        print_layers(module, name + "." + module_name)
