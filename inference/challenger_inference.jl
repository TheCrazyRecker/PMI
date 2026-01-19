using Gen
using LinearAlgebra
using Random
using Plots
using Statistics

include("../models/challenger.jl")
include("../common/base_inference.jl")
include("../common/utils.jl")

observations_vector = generate_challenger_data(challenger_temps)

function vector_to_choicemap(data_vector)
    cm = choicemap()
    for (i, val) in enumerate(data_vector)
        cm[(:failure, i)] = val
    end
    return cm
end

const obs_choicemap = vector_to_choicemap(observations_vector)

# ==========================================
# 3. CUSTOM INFERENCE (REVISED, FIXED)
# ==========================================

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

# function run_custom_inference_challenger(n_samples::Int=5000)
#     println("Running Custom MH (Ridge-aligned + escape burn-in)...")

#     # --- SAFE INITIALIZATION (no iteration over obs_choicemap) ---
#     init_cm = merge(obs_choicemap, choicemap())
#     init_cm[:alpha] = 0.0
#     init_cm[:beta]  = 0.0

#     trace, _ = generate(challenger_model, (challenger_temps,), init_cm)

#     mean_t = mean(Float64.(challenger_temps))

#     # Ridge move scales
#     sigma_alpha = 0.05
#     sigma_beta_orth = 1.0

#     # Escape move scales (burn-in)
#     sigma_alpha_big = 1.0
#     sigma_beta_big  = 5.0

#     samples = Vector{Tuple{Float64, Float64}}(undef, n_samples)

#     burnin = min(3000, n_samples ÷ 3)
#     escape_prob = 0.15

#     acc = 0
#     for i in 1:n_samples
#         if i <= burnin && rand() < escape_prob
#             trace, accepted = metropolis_hastings(
#                 trace,
#                 challenger_global_rw_proposal,
#                 (sigma_alpha_big, sigma_beta_big)
#             )
#         else
#             trace, accepted = metropolis_hastings(
#                 trace,
#                 challenger_ridge_proposal,
#                 (mean_t, sigma_alpha, sigma_beta_orth)
#             )
#         end

#         acc += accepted ? 1 : 0
#         samples[i] = (trace[:alpha], trace[:beta])
#     end

#     println("Custom overall acceptance rate: ", acc / n_samples)
#     return samples
# end

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
        step_size=0.01,
        n_steps=25
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


# ==========================================
# 4. EXECUTION
# ==========================================

# n_iter = 100
# std_samples = run_standard_inference_challenger(n_iter)
# cst_samples = run_custom_inference_challenger(n_iter)

# std_alpha = [s[1] for s in std_samples]
# std_beta  = [s[2] for s in std_samples]
# cst_alpha = [s[1] for s in cst_samples]
# cst_beta  = [s[2] for s in cst_samples]

# println("Generating plots...")

# p1 = plot(std_alpha, label="Standard MH", color=:red, alpha=0.7,
#     xlabel="Iteration", ylabel="Alpha", title="Trace Plot (Alpha)")
# plot!(p1, cst_alpha, label="Custom", color=:blue, alpha=0.5)
# hline!(p1, [true_alpha], label="True Alpha", color=:green, linestyle=:dash)

# p2 = plot(std_beta, 
#     label="Standard MH", 
#     color=:red, 
#     alpha=0.7,
#     xlabel="Iteration", 
#     ylabel="Beta",
#     title="Trace Plot: Mixing Efficiency (Beta)"
# )
# plot!(p2, cst_beta, 
#     label="Custom", 
#     color=:blue, 
#     alpha=0.5
# )
# hline!(p2, [true_beta], label="True Beta", color=:green, linestyle=:dash)

# combined_plot = plot(p1, p2, layout=(2, 1), size=(800, 1000))

# display(combined_plot)


n_iter = 20000  # increase for nicer traces

# Custom
cst_samples = run_custom_inference_challenger(n_iter)

# Comparators
# mh_samples  = run_mh_rw_challenger(n_iter)
# hmc_samples = run_hmc_challenger(n_iter)
# is_samples  = run_is_challenger(n_iter)
# pf_samples  = run_pf_challenger(n_iter)

# Split into alpha/beta vectors
function split_samples(samples)
    a = [s[1] for s in samples]
    b = [s[2] for s in samples]
    return a, b
end

cst_alpha, cst_beta = split_samples(cst_samples)
# mh_alpha,  mh_beta  = split_samples(mh_samples)
# hmc_alpha, hmc_beta = split_samples(hmc_samples)
# is_alpha,  is_beta  = split_samples(is_samples)
# pf_alpha,  pf_beta  = split_samples(pf_samples)

println("Generating plots...")

p1 = plot(cst_alpha, label="Custom", alpha=0.8,
    xlabel="Iteration / Sample Index", ylabel="Alpha", title="Alpha Trace / Samples")
# plot!(p1, mh_alpha,  label="MH (RW)", alpha=0.6)
# plot!(p1, hmc_alpha, label="HMC", alpha=0.6)
# plot!(p1, is_alpha,  label="IS (resampled)", alpha=0.6)
# plot!(p1, pf_alpha,  label="PF/SMC", alpha=0.6)
hline!(p1, [true_alpha], label="True Alpha", linestyle=:dash)

p2 = plot(cst_beta, label="Custom", alpha=0.8,
    xlabel="Iteration / Sample Index", ylabel="Beta", title="Beta Trace / Samples")
# plot!(p2, mh_beta,  label="MH (RW)", alpha=0.6)
# plot!(p2, hmc_beta, label="HMC", alpha=0.6)
# plot!(p2, is_beta,  label="IS (resampled)", alpha=0.6)
# plot!(p2, pf_beta,  label="PF/SMC", alpha=0.6)
hline!(p2, [true_beta], label="True Beta", linestyle=:dash)

combined_plot = plot(p1, p2, layout=(2, 1), size=(900, 1000))
display(combined_plot)

function running_mean(x)
    s = 0.0
    out = similar(x, Float64)
    for i in eachindex(x)
        s += x[i]
        out[i] = s / i
    end
    out
end

mean_alpha = plot(running_mean([s[1] for s in cst_samples]), title="Running mean of alpha", label="mean(alpha)")
hline!(mean_alpha, [true_alpha], linestyle=:dash, label="true")

mean_beta = plot(running_mean([s[2] for s in cst_samples]), title="Running mean of beta", label="mean(alpha)")
hline!(mean_beta, [true_beta], linestyle=:dash, label="true")

combined_means_plot = plot(mean_alpha, mean_beta, layout=(2, 1), size=(900, 1000))
display(combined_means_plot)