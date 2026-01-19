using Gen
using Random
using Plots
using Statistics

include("../../models/text.jl")
include("../../common/base_inference.jl")
include("../../common/utils.jl")

function counts_to_choicemap(counts::Vector{Int})
    cm = choicemap()
    for (i, val) in enumerate(counts)
        cm[(:count, i)] = val
    end
    return cm
end

const obs_choicemap = counts_to_choicemap(observations)



#This calculates two metrics: number of days of one segment and the sum of its counts. It does this for both segments
function segment_stats(counts::Vector{Int}, tau::Int)
    n1 = tau - 1
    S1 = (n1 == 0) ? 0 : sum(@view counts[1:n1])
    n2 = length(counts) - n1
    S2 = sum(counts) - S1
    return n1, S1, n2, S2
end


function gibbs_lambdas(counts::Vector{Int}, tau::Int; a=1.0, θ=20.0, rng=Random.default_rng())
    n1, S1, n2, S2 = segment_stats(counts, tau)
    β = 1 / θ
    λ1 = rand(rng, Gamma(a + S1, 1 / (β + n1)))
    λ2 = rand(rng, Gamma(a + S2, 1 / (β + n2)))
    return λ1, λ2
end

# MH: propose tau -> tau +/- 1 and accept with boundary-only likelihood ratio
function mh_tau_step(counts::Vector{Int}, tau::Int, λ1::Float64, λ2::Float64; rng=Random.default_rng())
    N = length(counts)
    if N == 1
        return tau, false
    end

    # propose +/- 1, reflecting at boundaries
    step = rand(rng, Bool) ? 1 : -1
    tau_prop = tau + step
    if tau_prop < 1
        tau_prop = 2
    elseif tau_prop > N
        tau_prop = N - 1
    end

    # Only ONE day's assignment changes when tau moves by 1:
    # - if tau_prop = tau + 1: day tau switches from segment2 -> segment1
    # - if tau_prop = tau - 1: day (tau-1) switches from segment1 -> segment2
    if tau_prop == tau + 1
        i = tau
        logα = logpdf(Poisson(λ1), counts[i]) - logpdf(Poisson(λ2), counts[i])
    else
        i = tau - 1
        logα = logpdf(Poisson(λ2), counts[i]) - logpdf(Poisson(λ1), counts[i])
    end

    if log(rand(rng)) < logα
        return tau_prop, true
    else
        return tau, false
    end
end

function custom_text_inference_mh_gibbs(counts::Vector{Int};
    niter=20_000, burnin=5_000, thin=10, rng=Random.default_rng()
)
    N = length(counts)
    tau = rand(rng, 1:N)
    λ1, λ2 = gibbs_lambdas(counts, tau; rng=rng)

    nsaved = max(0, (niter - burnin) ÷ thin)
    taus = Vector{Int}(undef, nsaved)
    λ1s  = Vector{Float64}(undef, nsaved)
    λ2s  = Vector{Float64}(undef, nsaved)

    acc = 0
    saved = 0

    for it in 1:niter
        # 1) Gibbs update for continuous rates
        λ1, λ2 = gibbs_lambdas(counts, tau; rng=rng)

        # 2) MH update for discrete changepoint tau
        tau, accepted = mh_tau_step(counts, tau, λ1, λ2; rng=rng)
        acc += accepted ? 1 : 0

        # Save samples
        if it > burnin && ((it - burnin) % thin == 0)
            saved += 1
            taus[saved] = tau
            λ1s[saved] = λ1
            λ2s[saved] = λ2
        end
    end

    return (taus=taus, lambda1=λ1s, lambda2=λ2s, acceptance_rate=acc / niter)
end

N = length(observations)

function run_hmc_text(n_samples::Int)
    traces = hmc_inference(text_model, (N,);
        observations=obs_choicemap,
        latent_addrs=[:lambda1, :lambda2],   # HMC only on continuous vars
        n_samples=n_samples,
        burnin=0,
        thin=1,
        init_constraints=choicemap((:lambda1, 10.0), (:lambda2, 10.0), (:tau, Int(clamp(round(N/2), 1, N)))),
        step_size=0.01,
        n_steps=50
    )
    return traces
end

function run_is_text(n_samples::Int)
    traces, _ = is_inference(text_model, (N,);
        observations=obs_choicemap,
        n_particles=n_samples,
        resample=true
    )
    return traces
end

function run_pf_text(n_samples::Int)
    obs_order = [(:count, i) for i in 1:N]
    particles, _ = pf_inference(text_model, (N,);
        observations=obs_choicemap,
        obs_order=obs_order,
        n_particles=n_samples,
        ess_threshold=0.5
    )
    return particles
end

