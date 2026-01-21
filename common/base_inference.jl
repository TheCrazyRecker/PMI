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

#Helper methods
function normalize_logweights(logw::Vector{Float64})
    m = maximum(logw)
    w = exp.(logw .- m)
    s = sum(w)
    return w ./ s
end


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

function traces_to_samples(traces::Vector{<:Gen.Trace}, addrs::Vector)
    out = Vector{Tuple}(undef, length(traces))
    for i in 1:length(traces)
        tr = traces[i]
        out[i] = tuple((tr[a] for a in addrs)...)
    end
    return out
end


#Random walk proposal for MH
@gen function rw_proposal(trace::Gen.Trace, addrs::Vector, sigmas::Vector{Float64})
    for (k, addr) in enumerate(addrs)
        cur = trace[addr]
        # Assume Real for RW proposal:
        {addr} ~ normal(cur, sigmas[k])
    end
end

#Next 4 methods are the base infernce methods reused by challenger and text model
#MH
function mh_inference(model::GenerativeFunction, model_args::Tuple;
                      observations::ChoiceMap=choicemap(),
                      latent_addrs::Vector=Any[],
                      n_samples::Int=2000,
                      burnin::Int=0,
                      thin::Int=1,
                      init_constraints::ChoiceMap=choicemap(),
                      rw_sigmas::Union{Nothing,Vector{Float64}}=nothing,
                      rng::AbstractRNG=Random.default_rng())

    constraints = merge(observations, init_constraints)
    trace, _ = generate(model, model_args, constraints)

    traces = Gen.Trace[]
    total_iters = burnin + n_samples * thin

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

    sigmas = rw_sigmas === nothing ? fill(0.5, length(rw_addrs)) : rw_sigmas

    kept = 0
    for it in 1:total_iters
        if !isempty(rw_addrs)
            trace, _ = metropolis_hastings(trace, rw_proposal, (rw_addrs, sigmas))
        end

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

#HMC
function hmc_inference(model::GenerativeFunction, model_args::Tuple;
                       observations::ChoiceMap=choicemap(),
                       latent_addrs::Vector=Any[],
                       n_samples::Int=2000,
                       burnin::Int=0,
                       thin::Int=1,
                       init_constraints::ChoiceMap=choicemap(),
                       step_size::Float64=0.02,  
                       n_steps::Int=10,           
                       rng::AbstractRNG=Random.default_rng())

    constraints = merge(observations, init_constraints)
    trace, _ = generate(model, model_args, constraints)

    sel = select(latent_addrs...)

    traces = Gen.Trace[]
    total_iters = burnin + n_samples * thin

    for it in 1:total_iters
        if it % 100 == 0
            println(it)
        end
        trace, _ = hmc(trace, sel; L=n_steps, eps=step_size)

        if it > burnin && (it - burnin) % thin == 0
            push!(traces, trace)
        end
    end

    return traces
end

#IS
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

#PF
function pf_inference(model::GenerativeFunction, model_args::Tuple;
                      observations::ChoiceMap,
                      obs_order::Vector,
                      n_particles::Int=2000,
                      rng::AbstractRNG=Random.default_rng(),
                      ess_threshold::Float64=0.5)

    particles = Vector{Gen.Trace}(undef, n_particles)
    logw = zeros(Float64, n_particles)

    for i in 1:n_particles
        tr, _ = generate(model, model_args) 
        particles[i] = tr
        logw[i] = 0.0
    end

    for addr in obs_order
        cm = choicemap()
        cm[addr] = observations[addr]

        for i in 1:n_particles
            tr = particles[i]
            new_tr, w, _ = update(tr, model_args, (NoChange(),), cm)
            particles[i] = new_tr
            logw[i] += w
        end

        wnorm = normalize_logweights(logw)
        ess = 1.0 / sum(wnorm .^ 2)

        if ess < ess_threshold * n_particles
            particles = resample_systematic(rng, particles, wnorm)
            logw .= 0.0 
        end
    end

    wnorm = normalize_logweights(logw)
    return particles, wnorm
end