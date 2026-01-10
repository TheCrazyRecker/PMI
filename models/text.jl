using Gen

@gen function text_model(total_days::Int)
    # 1. Priors for the rates (average texts per day)
    lambda1 ~ gamma(1, 20) # Rate before the switch
    lambda2 ~ gamma(1, 20) # Rate after the swtich

    tau ~ uniform_discrete(1, total_days) # The day we switch rates

    for i in 1:total_days
        # Checks if we have reached the above mentioned day yet, if so, we switch rates, if not, we keep the same rate
        rate = (i < tau) ? lambda1 : lambda2
        
        # We observe the number of texts for day 'i'
        # based on the selected rate.
        {(:count, i)} ~ poisson(rate)
    end
end

# Generates the data
function generate_text_data()
    # Let's pretend the switch happens on day 40
    # Rate before = 10 texts/day, Rate after = 40 texts/day
    constraints = choicemap()
    constraints[:lambda1] = 10.0
    constraints[:lambda2] = 40.0
    constraints[:tau] = 40
    
    num_data_points = 74

    (trace, _) = generate(text_model, (num_data_points,), constraints)
    data = [trace[(:count, i)] for i in 1:num_data_points]

    return data
end

observations = generate_text_data()
println("Generated $(length(observations)) days of text message counts.")
println(observations)