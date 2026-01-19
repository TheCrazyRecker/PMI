using Gen
using LinearAlgebra
using Random
using Plots
using Statistics

include("../../models/challenger.jl")
include("../../common/base_inference.jl")
include("../../common/utils.jl")

observations_vector = generate_challenger_data(challenger_temps)

function vector_to_choicemap(data_vector)
    cm = choicemap()
    for (i, val) in enumerate(data_vector)
        cm[(:failure, i)] = val
    end
    return cm
end

const obs_choicemap = vector_to_choicemap(observations_vector)


@gen function challenger_ridge_proposal(trace::Gen.Trace,
                                       mean_t::Float64,
                                       sigma_alpha::Float64,
                                       sigma_beta_orth::Float64)
    alpha = trace[:alpha]
    beta  = trace[:beta]

    new_alpha = {:alpha} ~ normal(alpha, sigma_alpha)

    compensated_mean = beta - mean_t * (new_alpha - alpha)
    {:beta} ~ normal(compensated_mean, sigma_beta_orth)
end

@gen function challenger_global_rw_proposal(trace::Gen.Trace,
                                           sigma_alpha_big::Float64,
                                           sigma_beta_big::Float64)
    alpha = trace[:alpha]
    beta  = trace[:beta]
    {:alpha} ~ normal(alpha, sigma_alpha_big)
    {:beta}  ~ normal(beta,  sigma_beta_big)
end

function run_custom_inference_challenger(n_samples::Int=5000; burnin::Int=2000)
    println("Running Custom MH (Ridge-aligned + escape burn-in)...")

    init_cm = merge(obs_choicemap, choicemap())
    init_cm[:alpha] = 0.0
    init_cm[:beta]  = 0.0
    trace, _ = generate(challenger_model, (challenger_temps,), init_cm)

    mean_t = mean(Float64.(challenger_temps))

    sigma_alpha = 0.05
    sigma_beta_orth = 1.0
    sigma_alpha_big = 1.0
    sigma_beta_big  = 5.0

    escape_prob = 0.15

    # burn-in phase (not recorded)
    for i in 1:burnin
        if rand() < escape_prob
            trace, _ = metropolis_hastings(trace, challenger_global_rw_proposal, (sigma_alpha_big, sigma_beta_big))
        else
            trace, _ = metropolis_hastings(trace, challenger_ridge_proposal, (mean_t, sigma_alpha, sigma_beta_orth))
        end
    end

    # sampling phase (recorded)
    samples = Vector{Tuple{Float64, Float64}}(undef, n_samples)
    acc = 0
    for i in 1:n_samples
        trace, accepted = metropolis_hastings(trace, challenger_ridge_proposal, (mean_t, sigma_alpha, sigma_beta_orth))
        acc += accepted ? 1 : 0
        samples[i] = (trace[:alpha], trace[:beta])
    end

    println("Custom acceptance (post burn-in): ", acc / n_samples)
    return samples
end



function run_mh_rw_challenger(n_samples::Int)
    println("Running MH (Random-Walk baseline)...")
    traces = mh_inference(challenger_model, (challenger_temps,);
        observations=obs_choicemap,
        latent_addrs=[:alpha, :beta],
        n_samples=n_samples,
        burnin=0,
        thin=1,
        init_constraints=choicemap((:alpha, 0.0), (:beta, 0.0)),
        rw_sigmas=[0.05, 1.0]   # tune if you want
    )
    return traces_to_samples(traces, [:alpha, :beta])
end

function run_hmc_challenger(n_samples::Int)
    println("Running HMC...")

    traces = hmc_inference(challenger_model, (challenger_temps,);
        observations=obs_choicemap,
        latent_addrs=[:alpha, :beta],
        n_samples=n_samples,
        burnin=0,
        thin=1,
        init_constraints=choicemap((:alpha, 0.0), (:beta, 0.0)),
        step_size=0.001,   
        n_steps=500
    )

    return traces_to_samples(traces, [:alpha, :beta])
end


function run_is_challenger(n_samples::Int)
    println("Running Importance Sampling (resampled)...")
    traces, _ = is_inference(challenger_model, (challenger_temps,);
        observations=obs_choicemap,
        n_particles=n_samples,
        resample=true
    )
    return traces_to_samples(traces, [:alpha, :beta])
end

function run_pf_challenger(n_samples::Int)
    println("Running Particle Filter / SMC (resampling by ESS)...")
    obs_order = [(:failure, i) for i in 1:length(challenger_temps)]
    particles, _ = pf_inference(challenger_model, (challenger_temps,);
        observations=obs_choicemap,
        obs_order=obs_order,
        n_particles=n_samples,
        ess_threshold=0.5
    )
    return traces_to_samples(particles, [:alpha, :beta])
end