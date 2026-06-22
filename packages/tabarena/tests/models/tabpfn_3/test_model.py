from tabarena.models.tabpfn_3.model import TabPFN3Model


def _model_with_problem_type(problem_type: str) -> TabPFN3Model:
    """A bare instance with only ``problem_type`` set, bypassing AutoGluon's __init__."""
    model = TabPFN3Model.__new__(TabPFN3Model)
    model.problem_type = problem_type
    return model


def test__resolve_hyperparameters_for_problem_type__exact_key__wins_over_umbrella():
    model = _model_with_problem_type("multiclass")
    out = model._resolve_hyperparameters_for_problem_type(
        {"classification": {"a": 1}, "multiclass": {"a": 2, "b": 3}}
    )
    assert out == {"a": 2, "b": 3}


def test__resolve_hyperparameters_for_problem_type__classification_umbrella__used_for_binary():
    model = _model_with_problem_type("binary")
    out = model._resolve_hyperparameters_for_problem_type({"classification": {"a": 1}})
    assert out == {"a": 1}


def test__resolve_hyperparameters_for_problem_type__regression__no_umbrella_fallback():
    model = _model_with_problem_type("regression")
    assert model._resolve_hyperparameters_for_problem_type({"classification": {"a": 1}}) == {}


def test__resolve_hyperparameters_for_problem_type__none__returns_empty():
    model = _model_with_problem_type("regression")
    assert model._resolve_hyperparameters_for_problem_type(None) == {}
