

# --- the verb folded into the expression -------------------------------------------------------

def _space():
    from govsim.core.action import ActionSpace, VerbSpec
    return ActionSpace(verbs=[VerbSpec(name="set_lockdown"), VerbSpec(name="set_vaccination")],
                       context_vars=[])


def test_a_verb_folded_into_the_expr_is_accepted():
    """MEASURED: 19 of 22 no-action decisions on qwen3.5:0.8b were exactly this shape.

    Discarding it is not neutral — the decision becomes a silent no-op that leaves the previous law
    in force — and the slip rises with prompt length, so it is correlated with the treatment.
    """
    from govsim.core.llm.client import LLMResponse
    from govsim.regents.llm_regent import parse_action_requests
    reqs = parse_action_requests(
        LLMResponse(text='{"expr": "set_lockdown:0.5"}'), _space(), "regent:0")
    assert len(reqs) == 1
    assert reqs[0].verb == "set_lockdown"
    assert reqs[0].payload["expr"] == "0.5"


def test_whitespace_around_the_folded_verb_is_tolerated():
    from govsim.core.llm.client import LLMResponse
    from govsim.regents.llm_regent import parse_action_requests
    reqs = parse_action_requests(
        LLMResponse(text='{"expr": "set_vaccination : 0.9 if I > 0.02 else 0.0"}'),
        _space(), "regent:0")
    assert reqs[0].verb == "set_vaccination"
    assert reqs[0].payload["expr"] == "0.9 if I > 0.02 else 0.0"


def test_a_colon_inside_a_real_expression_is_not_mistaken_for_a_verb():
    """Only a KNOWN verb before the colon triggers the split. A dict literal or a slice must not."""
    from govsim.core.llm.client import LLMResponse
    from govsim.regents.llm_regent import parse_action_requests
    reqs = parse_action_requests(
        LLMResponse(text='{"verb": "set_lockdown", "expr": "{1: 2}[1]"}'), _space(), "regent:0")
    assert len(reqs) == 1
    assert reqs[0].payload["expr"] == "{1: 2}[1]", "the expression must survive untouched"


def test_an_unknown_verb_before_a_colon_is_still_rejected():
    from govsim.core.llm.client import LLMResponse
    from govsim.regents.llm_regent import parse_action_requests
    assert parse_action_requests(
        LLMResponse(text='{"expr": "set_interest_rate:0.5"}'), _space(), "regent:0") == []


def test_an_empty_tail_is_rejected_rather_than_installed_as_a_blank_law():
    from govsim.core.llm.client import LLMResponse
    from govsim.regents.llm_regent import parse_action_requests
    assert parse_action_requests(
        LLMResponse(text='{"expr": "set_lockdown:   "}'), _space(), "regent:0") == []


def test_an_explicit_verb_still_wins_over_the_folded_form():
    from govsim.core.llm.client import LLMResponse
    from govsim.regents.llm_regent import parse_action_requests
    reqs = parse_action_requests(
        LLMResponse(text='{"verb": "set_vaccination", "expr": "0.3"}'), _space(), "regent:0")
    assert reqs[0].verb == "set_vaccination" and reqs[0].payload["expr"] == "0.3"
