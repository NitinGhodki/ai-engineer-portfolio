three progressively complex graphs FLOW 

--> linear
graph TD (LinearState)
    START --> node_a
    node_a --> node_b
    node_b --> node_c
    node_c --> END

--> branching
graph TD (RoutingState)
    START --> classifier
    classifier --> router decides
    router decides --> node_a --> END
    router decides --> node_b --> END

--> joining
graph TD (ValidationState)
    START --> generate
    generate --> validate
    validate --> (if failed) --> generate (loop) --> (if passed) --> END
    validate --> (if passed) --> END