# generic_inference.jl
# Four general-purpose inference procedures for Gen:
#   1) Metropolis-Hastings (random-walk)
#   2) Hamiltonian Monte Carlo (wrapper over Gen.hmc with robust call patterns)
#   3) Importance Sampling
#   4) Particle Filter / SMC via incremental conditioning with update() + resampling
#
# Designed to be model-agnostic and importable.

using Gen
using Random
using Statistics

export mh_inference,
       hmc_inference,
       is_inference,
       pf_inference,
       resample_systematic,
       normalize_logweights,
       traces_to_samples

# --------------------------
# Utilities
# --------------------------

"""
    normalize_logweights(logw::Vector{Float64})

Return normalized weights `w` from log-weights `logw`, using a stable log-sum-exp.
"""
function normalize_logweights(logw::Vector{Float64})
    m = maximum(logw)
    w = exp.(logw .- m)
    s = sum(w)
    return w ./ s
end

"""
    resample_systematic(rng, particles, weights)

Systematic resampling. Returns new particles (with replacement) and equal weights.
"""
function resample_systematic(rng::AbstractRNG, particles::Vector, weights::Vector{Float64})
    N = length(particles)
    cdf = cumsum(weights)
    out = Vector{typeof(particles[1])}(undef, N)

    u0 = rand(rng) / N
    j = 1
    for i in 1:N
        u = u0 + (i - 1) / N
        while u > cdf[j]
            j += 1
        end
        out[i] = particles[j]
    end
    return out
end

"""
    traces_to_samples(traces, addrs)

Convert a vector of traces to a vector of tuples with values at `addrs`.
Example: addrs = [:alpha, :beta] => samples[i] = (trace[:alpha], trace[:beta])
"""
function traces_to_samples(traces::Vector{<:Gen.Trace}, addrs::Vector)
    out = Vector{Tuple}(undef, length(traces))
    for i in 1:length(traces)
        tr = traces[i]
        out[i] = tuple((tr[a] for a in addrs)...)
    end
    return out
end

# --------------------------
# 1) Metropolis-Hastings
# --------------------------

# Random-walk proposal over a list of real-valued addresses.
# For non-real values, we fall back to prior-resampling MH using select(addr).
@gen function rw_proposal(trace::Gen.Trace, addrs::Vector, sigmas::Vector{Float64})
    for (k, addr) in enumerate(addrs)
        cur = trace[addr]
        # Assume Real for RW proposal:
        {addr} ~ normal(cur, sigmas[k])
    end
end

"""
    mh_inference(model, model_args; observations, latent_addrs, n_samples, burnin, thin,
                 init_constraints, rw_sigmas, rng)

Generic MH. If `rw_sigmas` provided, does joint RW MH on `latent_addrs`.
If some latent addresses are non-Real, it automatically falls back to per-address `select(addr)` MH for those.
Returns: vector of traces.
"""
function mh_inference(model::GenerativeFunction, model_args::Tuple;
                      observations::ChoiceMap=choicemap(),
                      latent_addrs::Vector=Any[],
                      n_samples::Int=2000,
                      burnin::Int=0,
                      thin::Int=1,
                      init_constraints::ChoiceMap=choicemap(),
                      rw_sigmas::Union{Nothing,Vector{Float64}}=nothing,
                      rng::AbstractRNG=Random.default_rng())

    # initialize trace from constraints + observations
    constraints = merge(observations, init_constraints)
    trace, _ = generate(model, model_args, constraints)

    traces = Gen.Trace[]
    total_iters = burnin + n_samples * thin

    # Determine which addrs can use RW (Real) vs fallback (select)
    rw_addrs = Any[]
    fb_addrs = Any[]

    if !isempty(latent_addrs)
        for a in latent_addrs
            v = trace[a]
            if v isa Real
                push!(rw_addrs, a)
            else
                push!(fb_addrs, a)
            end
        end
    end

    # Default sigmas
    sigmas = rw_sigmas === nothing ? fill(0.5, length(rw_addrs)) : rw_sigmas
    if length(sigmas) != length(rw_addrs)
        error("mh_inference: rw_sigmas must match number of Real-valued latent_addrs.")
    end

    # Main loop
    kept = 0
    for it in 1:total_iters
        # Joint RW move (if any)
        if !isempty(rw_addrs)
            trace, _ = metropolis_hastings(trace, rw_proposal, (rw_addrs, sigmas))
        end

        # Fallback per-address proposals for non-Real
        for a in fb_addrs
            trace, _ = metropolis_hastings(trace, select(a))
        end

        if it > burnin && (it - burnin) % thin == 0
            kept += 1
            push!(traces, trace)
        end
    end

    return traces
end

# --------------------------
# 2) Hamiltonian Monte Carlo (HMC)
# --------------------------

"""
    hmc_inference(model, model_args; observations, latent_addrs, n_samples, burnin, thin,
                  init_constraints, step_size, n_steps, rng)

Generic HMC wrapper. Only valid for Real-valued latent addresses.
We attempt multiple Gen.hmc calling conventions for compatibility across Gen versions.
Returns: vector of traces.
"""
function hmc_inference(model::GenerativeFunction, model_args::Tuple;
                       observations::ChoiceMap=choicemap(),
                       latent_addrs::Vector=Any[],
                       n_samples::Int=2000,
                       burnin::Int=0,
                       thin::Int=1,
                       init_constraints::ChoiceMap=choicemap(),
                       step_size::Float64=0.02,   # maps to eps
                       n_steps::Int=10,           # maps to L
                       rng::AbstractRNG=Random.default_rng())

    constraints = merge(observations, init_constraints)
    trace, _ = generate(model, model_args, constraints)

    if isempty(latent_addrs)
        error("hmc_inference: provide latent_addrs (Real-valued) for HMC.")
    end
    for a in latent_addrs
        v = trace[a]
        if !(v isa Real)
            error("hmc_inference: address $a has non-Real value; HMC requires Real latents.")
        end
    end

    sel = select(latent_addrs...)

    traces = Gen.Trace[]
    total_iters = burnin + n_samples * thin

    for it in 1:total_iters
        # Gen version you have: hmc(trace, selection; L, eps, ...)
        trace, _ = hmc(trace, sel; L=n_steps, eps=step_size)

        if it > burnin && (it - burnin) % thin == 0
            push!(traces, trace)
        end
    end

    return traces
end


# --------------------------
# 3) Importance Sampling (IS)
# --------------------------

"""
    is_inference(model, model_args; observations, n_particles, rng, resample)

Generic importance sampling using `generate(model, args, observations)` as proposal (prior conditioned on obs).
Returns: traces, normalized_weights
If resample=true, returns resampled traces and uniform weights.
"""
function is_inference(model::GenerativeFunction, model_args::Tuple;
                      observations::ChoiceMap=choicemap(),
                      n_particles::Int=2000,
                      rng::AbstractRNG=Random.default_rng(),
                      resample::Bool=false)

    traces = Vector{Gen.Trace}(undef, n_particles)
    logw   = Vector{Float64}(undef, n_particles)

    for i in 1:n_particles
        tr, w = generate(model, model_args, observations)
        traces[i] = tr
        logw[i] = w
    end

    wnorm = normalize_logweights(logw)

    if resample
        traces_rs = resample_systematic(rng, traces, wnorm)
        return traces_rs, fill(1.0 / n_particles, n_particles)
    else
        return traces, wnorm
    end
end

# --------------------------
# 4) Particle Filter (PF / SMC) via incremental conditioning
# --------------------------

"""
    pf_inference(model, model_args; observations, obs_order, n_particles, rng, ess_threshold)

A generic SMC / PF for *any* Gen model, implemented by conditioning incrementally on observations
in a user-supplied order `obs_order` (vector of addresses).

This uses:
  - Prior sampling (generate with no observations)
  - update() to incorporate one observation at a time
  - resampling when ESS drops below threshold

Arguments:
- observations: ChoiceMap containing all observed addresses & values
- obs_order: Vector of addresses specifying incremental conditioning order
- ess_threshold: resample when ESS < ess_threshold * N (default 0.5)

Returns: particles (traces), normalized_weights
"""
function pf_inference(model::GenerativeFunction, model_args::Tuple;
                      observations::ChoiceMap,
                      obs_order::Vector,
                      n_particles::Int=2000,
                      rng::AbstractRNG=Random.default_rng(),
                      ess_threshold::Float64=0.5)

    # Initialize particles from prior (no obs)
    particles = Vector{Gen.Trace}(undef, n_particles)
    logw = zeros(Float64, n_particles)

    for i in 1:n_particles
        tr, _ = generate(model, model_args)  # prior sample
        particles[i] = tr
        logw[i] = 0.0
    end

    # Sequentially incorporate observations
    for addr in obs_order
        if !has_value(observations, addr)
            error("pf_inference: observations has no value at address $addr (but it's in obs_order).")
        end

        # Condition on one observation at a time
        cm = choicemap()
        cm[addr] = observations[addr]

        for i in 1:n_particles
            tr = particles[i]
            # No argument changes:
            new_tr, w, _ = update(tr, model_args, (NoChange(),), cm)
            particles[i] = new_tr
            logw[i] += w
        end

        # Normalize and possibly resample
        wnorm = normalize_logweights(logw)
        ess = 1.0 / sum(wnorm .^ 2)

        if ess < ess_threshold * n_particles
            particles = resample_systematic(rng, particles, wnorm)
            logw .= 0.0  # after resampling, reset incremental weights
        end
    end

    wnorm = normalize_logweights(logw)
    return particles, wnorm
end