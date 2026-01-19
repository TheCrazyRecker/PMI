using Statistics
using Random
using Distributions

function sample_temps_from_data(temps::Vector{Int}, n_samples::Int;
                                rng::AbstractRNG=Random.default_rng())

    mean_temp = mean(temps)
    var_temp  = var(temps)

    dist = Normal(mean_temp, sqrt(var_temp))
    samples = rand(rng, dist, n_samples)

    samples_int = round.(Int, samples)

    return samples_int
end
