import collections
import torch
import torch.nn.functional as F
import torch.distributions as tdist

VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns", 
    [
        "vs", 
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
    ],
)

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")


def action_log_probs(policy_logits, actions):
    return -F.nll_loss(
        F.log_softmax(torch.flatten(policy_logits, 0, -2), dim=-1),
        torch.flatten(actions),
        reduction="none", # 如果reduction="mean"，则求均值
    ).view_as(actions)

def action_log_probs_continuous(policy_logits, actions):
    actions = torch.flatten(actions)
    unroll_length = policy_logits.shape[0]
    batch_size = policy_logits.shape[1]
    policy_logits_flatten = torch.flatten(policy_logits, 0, -2) 
    log_probs = torch.zeros(unroll_length * batch_size, 1, dtype=torch.float32).to("cuda")
    dist = tdist.normal.Normal(policy_logits_flatten[:, 0], policy_logits_flatten[:, 1])
    log_probs = dist.log_prob(actions)
    return log_probs.view([unroll_length, batch_size])


def from_logits(
    behavior_policy_logits,
    target_policy_logits,
    actions,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    target_action_log_probs = action_log_probs_continuous(target_policy_logits, actions) # shape of target_action_log_probs = [unroll_length, batch_size]
    behavior_action_log_probs = action_log_probs_continuous(behavior_policy_logits, actions)
    log_rhos = target_action_log_probs - behavior_action_log_probs
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


@torch.no_grad()
def from_importance_weights(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace from log importance weights."""
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = torch.clamp(rhos, max=1.0) # cs.shape = [unroll_length, batch_size]
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        ) # values_t_plus_1.shape = [unroll_length, batch_size]
        # deltas 是公式(1)中的 δtV
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values) # deltas.shape = [unroll_length, batch_size]

        acc = torch.zeros_like(bootstrap_value) # acc.shape = [batch_size]
        result = [] 
        # 因为 vs-1 - V(X_s-1) 可以由 vs - V(x_s) 递推得到，所以逆序遍历一个transition
        for t in range(discounts.shape[0] - 1, -1, -1):  
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse() # result.length = unroll_length; result[0].shape = [batch_size]
        vs_minus_v_xs = torch.stack(result) # vs_minus_v_xs.shape = [unroll_length, batch_size]

        # vs - V(x_s) + V(x_s) = v_s
        vs = torch.add(vs_minus_v_xs, values) # vs.shape = [unroll_length, batch_size]

        # Advantage for policy gradient.
        broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value # broadcasted_bootstrap_values.shape = [batch_size]
        vs_t_plus_1 = torch.cat( # vs_t_plus_1.shape = [unroll_length, batch_size]
            [vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0 # vs[1:].shape = [unroll_length - 1, batch_size]; 
        ) # broadcasted_bootstrap_values.unsqueeze(0).shape = [1, batch_size]
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold) # clipped_pg_rhos.shape = [unroll_length, batch_size]
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values) # pg_advantages = [unroll_length, batch_size]

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)
