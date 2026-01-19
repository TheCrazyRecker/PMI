using Gen
include("../common/utils.jl")

# These are the actual recorded temperatures from the 23 flights.
actual_challenger_temps = [
    66, 70, 69, 68, 67, 72, 73, 70, 57, 63, 70, 78, 
    67, 53, 67, 75, 70, 81, 76, 79, 75, 76, 58
]

challenger_temps = sample_temps_from_data(actual_challenger_temps, 100)

true_alpha = 0.23
true_beta = -15.0

@gen function challenger_model(temps::Vector{Int})
    # Priors for the physics parameters
    alpha ~ normal(0.0, 1000.0) # Temperature sensitivity (slope)
    beta  ~ normal(0.0, 1000.0) # Baseline bias (intercept)

    # 2. Loop over each flight temperature
    for (i, t) in enumerate(temps)
        
        # 3. The "Simulator" (Logistic Function)
        #    Calculate probability of failure at this specific temperature
        logit_p = alpha * t + beta
        p_failure = 1.0 / (1.0 + exp(-logit_p))
        
        # 4. Observation
        #    We observe if the O-ring failed (1) or not (0)
        {(:failure, i)} ~ bernoulli(p_failure)
    end
end

# Generates synthetic data where we actually know alpha and beta (which we set ourselves)
function generate_challenger_data(inputs)

    constraints = choicemap()
    constraints[:alpha] = true_alpha 
    constraints[:beta]  = true_beta 
    
    (trace, _) = generate(challenger_model, (inputs,), constraints)
    
    data = [trace[(:failure, i)] for i in 1:length(inputs)]
    
    return data
end

observations = generate_challenger_data(challenger_temps)

println("Generated $(length(observations)) flight outcomes based on temperatures.")
println("Temperatures: ", challenger_temps)
println("Failures:     ", observations) # 1 = failure