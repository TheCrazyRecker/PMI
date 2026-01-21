include("text_inference_v2.jl")
include("../../common/utils.jl")

true_tau = 40
true_l1 = 10
true_l2 = 40

n_iter = 10000

is_traces = run_is_text(n_iter)
pf_traces = run_pf_text(n_iter)

is_tau = extract_param(is_traces, :tau)
is_l1  = extract_param(is_traces, :lambda1)
is_l2  = extract_param(is_traces, :lambda2)

pf_tau = extract_param(pf_traces, :tau)
pf_l1  = extract_param(pf_traces, :lambda1)
pf_l2  = extract_param(pf_traces, :lambda2)

#Helpers for nicer plots

xl1 = shared_xlim(is_l1, pf_l1)
xl2 = shared_xlim(is_l2, pf_l2)
# ---------- Plots ----------

p_is_tau = tau_pmf_plot(is_tau, N; title="IS: τ PMF")
p_is_l1  = lambda_hist_plot(is_l1; title="IS: λ1 posterior", xlabel="lambda1", truth=true_l1, xlims=xl1)
p_is_l2  = lambda_hist_plot(is_l2; title="IS: λ2 posterior", xlabel="lambda2", truth=true_l2, xlims=xl2)

p_pf_tau = tau_pmf_plot(pf_tau, N; title="PF: τ PMF")
p_pf_l1  = lambda_hist_plot(pf_l1; title="PF: λ1 posterior", xlabel="lambda1", truth=true_l1, xlims=xl1)
p_pf_l2  = lambda_hist_plot(pf_l2; title="PF: λ2 posterior", xlabel="lambda2", truth=true_l2, xlims=xl2)

combined = plot(p_is_tau, p_is_l1, p_is_l2,
                p_pf_tau, p_pf_l1, p_pf_l2,
                layout=(2,3), size=(1100, 550), margin=5Plots.mm)

display(combined)