using Gen

# Actual historic data
challenger_data = [
    (66, 0), (70, 1), (69, 0), (68, 0), (67, 0), (72, 0), (73, 0),
    (70, 0), (57, 1), (63, 1), (70, 1), (78, 0), (67, 0), (53, 1),
    (67, 0), (75, 0), (70, 0), (81, 0), (76, 0), (79, 0), (75, 1),
    (76, 0), (58, 1)
]

@gen function challenger_model(data::Vector{Tuple{Int, Int}})
    # 1. Priors for the physics parameters
    #    Alpha (Sensitivity): Usually positive (colder = worse)
    #    Beta (Bias): The baseline failure rate
    alpha ~ normal(0.0, 1.0) 
    beta  ~ normal(0.0, 5.0)

    # 2. Loop over each flight in history
    for (i, (temp, failed)) in enumerate(data)
        
        # 3. The "Simulator" (Logistic Function)
        #    Calculate probability of failure at this specific temperature
        logit_p = alpha * temp + beta
        p_failure = 1.0 / (1.0 + exp(-logit_p))
        
        # 4. Observation
        #    We verify if the O-ring actually failed
        {(:failure, i)} ~ bernoulli(p_failure)
    end
end

# --- RUN AND INSPECT ---
# We run the model "forward" to see what it predicts with random priors.
# (In Part 2, you will infer the specific alpha/beta that match the data).
(trace, _) = generate(challenger_model, (challenger_data,))

# Extract the random guesses for alpha/beta
rand_alpha = trace[:alpha]
rand_beta  = trace[:beta]

# Calculate the predicted risk at 31°F (The launch day temperature)
# Note: Since we are just sampling from priors, this will be random garbage for now.
# Later, inference will make this accurate.
launch_logit = rand_alpha * 31.0 + rand_beta
launch_risk = 1.0 / (1.0 + exp(-launch_logit))

println("--- CHALLENGER SIMULATION REPORT ---")
println("Simulated Logic for $(length(challenger_data)) historical flights.")
println("Latent Variables (Randomly Sampled from Prior):")
println("  Sensitivity (Alpha): ", round(rand_alpha, digits=3))
println("  Baseline    (Beta) : ", round(rand_beta, digits=3))

println("\nPrediction for Launch Day (31°F):")
println("  Calculated Risk: ", round(launch_risk * 100, digits=1), "%")
println("  (Note: Without inference, this number is random. With inference, it should be near 99%.)")