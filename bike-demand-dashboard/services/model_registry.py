from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    kind: str  # "int" | "float" | "bool" | "select"
    default: Any
    help: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[Any]] = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: str  # "regression" | "classification"
    family: str
    formula: str
    notes: str
    requires: Tuple[str, ...]
    params: Tuple[ParamSpec, ...]
    factory: Callable[[Dict[str, Any]], Any]


def _has(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _missing(requires: Tuple[str, ...]) -> List[str]:
    return [m for m in requires if not _has(m)]


def _clamp_int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        x = int(v)
    except Exception:
        return int(default)
    return int(max(lo, min(hi, x)))


def _clamp_float(v: Any, default: float, lo: float, hi: float) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    return float(max(lo, min(hi, x)))


def _bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _regression_specs() -> List[ModelSpec]:
    # Keep names stable because they are stored in DB and referenced by UI.
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, PoissonRegressor, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    specs: List[ModelSpec] = []

    specs.append(
        ModelSpec(
            name="Linear Regression",
            task="regression",
            family="Linear",
            formula="ŷ = β₀ + Σᵢ βᵢ xᵢ",
            notes="Fast baseline; assumes mostly linear relationships.",
            requires=(),
            params=(),
            factory=lambda p: LinearRegression(),
        )
    )

    specs.append(
        ModelSpec(
            name="Ridge Regression",
            task="regression",
            family="Linear",
            formula="minβ ||y − Xβ||² + α||β||²",
            notes="Linear baseline with L2 regularization (helps multicollinearity).",
            requires=(),
            params=(
                ParamSpec("alpha", "α (regularization)", "float", 1.0, min=0.0, max=1000.0, step=0.1),
            ),
            factory=lambda p: Ridge(alpha=_clamp_float(p.get("alpha"), 1.0, 0.0, 1e6), random_state=42),
        )
    )

    specs.append(
        ModelSpec(
            name="Decision Tree Regressor",
            task="regression",
            family="Tree",
            formula="ŷ = average(y in leaf(x))",
            notes="Non-linear; can overfit; good for quick interpretability.",
            requires=(),
            params=(
                ParamSpec("max_depth", "Max depth", "int", 0, help="0 = unlimited", min=0, max=64, step=1),
                ParamSpec("min_samples_leaf", "Min samples/leaf", "int", 1, min=1, max=50, step=1),
            ),
            factory=lambda p: DecisionTreeRegressor(
                random_state=42,
                max_depth=None if _clamp_int(p.get("max_depth"), 0, 0, 10_000) <= 0 else _clamp_int(p.get("max_depth"), 0, 1, 10_000),
                min_samples_leaf=_clamp_int(p.get("min_samples_leaf"), 1, 1, 10_000),
            ),
        )
    )

    specs.append(
        ModelSpec(
            name="Random Forest Regressor",
            task="regression",
            family="Ensemble (Bagging)",
            formula="ŷ = (1/T) Σₜ fₜ(x)",
            notes="Strong general model; robust for mixed features.",
            requires=(),
            params=(
                ParamSpec("n_estimators", "Trees", "int", 250, min=50, max=1200, step=50),
                ParamSpec("max_depth", "Max depth", "int", 0, help="0 = unlimited", min=0, max=64, step=1),
            ),
            factory=lambda p: RandomForestRegressor(
                n_estimators=_clamp_int(p.get("n_estimators"), 250, 10, 5000),
                random_state=42,
                n_jobs=-1,
                max_depth=None if _clamp_int(p.get("max_depth"), 0, 0, 10_000) <= 0 else _clamp_int(p.get("max_depth"), 0, 1, 10_000),
            ),
        )
    )

    specs.append(
        ModelSpec(
            name="Gradient Boosting Regressor",
            task="regression",
            family="Ensemble (Boosting)",
            formula="fₘ(x)=fₘ₋₁(x)+η·hₘ(x)  (fit hₘ to residuals)",
            notes="High accuracy; may need tuning; handles non-linearities well.",
            requires=(),
            params=(
                ParamSpec("n_estimators", "Stages", "int", 300, min=50, max=2000, step=50),
                ParamSpec("learning_rate", "Learning rate", "float", 0.05, min=0.001, max=1.0, step=0.01),
                ParamSpec("max_depth", "Max depth", "int", 3, min=1, max=10, step=1),
            ),
            factory=lambda p: GradientBoostingRegressor(
                random_state=42,
                n_estimators=_clamp_int(p.get("n_estimators"), 300, 10, 5000),
                learning_rate=_clamp_float(p.get("learning_rate"), 0.05, 1e-4, 10.0),
                max_depth=_clamp_int(p.get("max_depth"), 3, 1, 32),
            ),
        )
    )

    specs.append(
        ModelSpec(
            name="KNN Regressor",
            task="regression",
            family="Instance-based",
            formula="ŷ = Σᵢ wᵢ yᵢ / Σᵢ wᵢ  (neighbors of x)",
            notes="Can be slow on large datasets; sensitive to scaling.",
            requires=(),
            params=(
                ParamSpec("n_neighbors", "Neighbors (k)", "int", 12, min=1, max=100, step=1),
            ),
            factory=lambda p: KNeighborsRegressor(
                n_neighbors=_clamp_int(p.get("n_neighbors"), 12, 1, 5000),
                weights="distance",
            ),
        )
    )

    specs.append(
        ModelSpec(
            name="SVR (RBF)",
            task="regression",
            family="Kernel method",
            formula="min (1/2)||w||² + CΣ(ξᵢ+ξᵢ*)  s.t. |y−f(x)|≤ε",
            notes="Powerful for non-linear patterns; can be slow on large data.",
            requires=(),
            params=(
                ParamSpec("C", "C", "float", 10.0, min=0.01, max=500.0, step=0.5),
                ParamSpec("epsilon", "ε", "float", 0.1, min=0.0, max=5.0, step=0.05),
                ParamSpec("gamma", "Gamma", "select", "scale", options=["scale", "auto"]),
            ),
            factory=lambda p: SVR(
                C=_clamp_float(p.get("C"), 10.0, 1e-4, 1e9),
                epsilon=_clamp_float(p.get("epsilon"), 0.1, 0.0, 1e9),
                gamma=(p.get("gamma") if p.get("gamma") in {"scale", "auto"} else "scale"),
            ),
        )
    )

    # Count regression baseline (commonly used when "demand" is a count)
    specs.append(
        ModelSpec(
            name="Poisson Regressor (Count)",
            task="regression",
            family="GLM (Count)",
            formula="log(E[y|x]) = β₀ + Σᵢ βᵢ xᵢ",
            notes="Good for non-negative count demand. If your target is not a count, prefer other regressors.",
            requires=(),
            params=(
                ParamSpec("alpha", "α (L2)", "float", 0.0, min=0.0, max=10.0, step=0.1),
                ParamSpec("max_iter", "Max iterations", "int", 300, min=50, max=3000, step=50),
            ),
            factory=lambda p: PoissonRegressor(
                alpha=_clamp_float(p.get("alpha"), 0.0, 0.0, 1e6),
                max_iter=_clamp_int(p.get("max_iter"), 300, 50, 100_000),
            ),
        )
    )

    # Optional: statsmodels Negative Binomial (sklearn-like wrapper)
    if _has("statsmodels"):
        import numpy as np
        import statsmodels.api as sm

        class _NBRegressor:
            def __init__(self, alpha: float = 1.0, max_iter: int = 100):
                self.alpha = float(alpha)
                self.max_iter = int(max_iter)
                self._result = None

            def fit(self, X, y):
                X2 = sm.add_constant(np.asarray(X), has_constant="add")
                y2 = np.asarray(y, dtype=float)
                model = sm.GLM(y2, X2, family=sm.families.NegativeBinomial(alpha=self.alpha))
                self._result = model.fit(maxiter=self.max_iter, disp=0)
                return self

            def predict(self, X):
                if self._result is None:
                    raise RuntimeError("Model is not fitted.")
                X2 = sm.add_constant(np.asarray(X), has_constant="add")
                return self._result.predict(X2)

        specs.append(
            ModelSpec(
                name="Negative Binomial Regressor",
                task="regression",
                family="GLM (Count)",
                formula="Var(y|x) = μ + αμ²  (over-dispersed counts)",
                notes="Useful for over-dispersed count demand. Requires statsmodels.",
                requires=("statsmodels",),
                params=(
                    ParamSpec("alpha", "α (dispersion)", "float", 1.0, min=0.0, max=50.0, step=0.1),
                    ParamSpec("max_iter", "Max iterations", "int", 100, min=20, max=1000, step=20),
                ),
                factory=lambda p: _NBRegressor(
                    alpha=_clamp_float(p.get("alpha"), 1.0, 0.0, 1e6),
                    max_iter=_clamp_int(p.get("max_iter"), 100, 20, 100_000),
                ),
            )
        )
    else:
        specs.append(
            ModelSpec(
                name="Negative Binomial Regressor",
                task="regression",
                family="GLM (Count)",
                formula="Var(y|x) = μ + αμ²  (over-dispersed counts)",
                notes="Requires statsmodels (optional).",
                requires=("statsmodels",),
                params=(),
                factory=lambda p: None,
            )
        )

    # Optional: XGBoost
    if _has("xgboost"):
        from xgboost import XGBRegressor

        specs.append(
            ModelSpec(
                name="XGBoost Regressor",
                task="regression",
                family="Boosting (XGBoost)",
                formula="ŷ = Σₘ η·fₘ(x)  (gradient-boosted trees)",
                notes="High accuracy; fast on large data; requires xgboost.",
                requires=("xgboost",),
                params=(
                    ParamSpec("n_estimators", "Trees", "int", 600, min=50, max=4000, step=50),
                    ParamSpec("learning_rate", "Learning rate", "float", 0.05, min=0.001, max=1.0, step=0.01),
                    ParamSpec("max_depth", "Max depth", "int", 6, min=2, max=16, step=1),
                    ParamSpec("subsample", "Subsample", "float", 0.9, min=0.3, max=1.0, step=0.05),
                ),
                factory=lambda p: XGBRegressor(
                    n_estimators=_clamp_int(p.get("n_estimators"), 600, 10, 50_000),
                    learning_rate=_clamp_float(p.get("learning_rate"), 0.05, 1e-4, 10.0),
                    max_depth=_clamp_int(p.get("max_depth"), 6, 1, 64),
                    subsample=_clamp_float(p.get("subsample"), 0.9, 0.05, 1.0),
                    colsample_bytree=1.0,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    objective="reg:squarederror",
                ),
            )
        )
    else:
        specs.append(
            ModelSpec(
                name="XGBoost Regressor",
                task="regression",
                family="Boosting (XGBoost)",
                formula="ŷ = Σₘ η·fₘ(x)  (gradient-boosted trees)",
                notes="Requires xgboost (optional).",
                requires=("xgboost",),
                params=(),
                factory=lambda p: None,
            )
        )

    # Optional: LightGBM
    if _has("lightgbm"):
        from lightgbm import LGBMRegressor

        specs.append(
            ModelSpec(
                name="LightGBM Regressor",
                task="regression",
                family="Boosting (LightGBM)",
                formula="ŷ = Σₘ η·fₘ(x)  (leaf-wise gradient boosting)",
                notes="High accuracy; fast; requires lightgbm.",
                requires=("lightgbm",),
                params=(
                    ParamSpec("n_estimators", "Trees", "int", 800, min=50, max=8000, step=50),
                    ParamSpec("learning_rate", "Learning rate", "float", 0.05, min=0.001, max=1.0, step=0.01),
                    ParamSpec("num_leaves", "Num leaves", "int", 63, min=7, max=255, step=1),
                ),
                factory=lambda p: LGBMRegressor(
                    n_estimators=_clamp_int(p.get("n_estimators"), 800, 10, 50_000),
                    learning_rate=_clamp_float(p.get("learning_rate"), 0.05, 1e-4, 10.0),
                    num_leaves=_clamp_int(p.get("num_leaves"), 63, 2, 2048),
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        )
    else:
        specs.append(
            ModelSpec(
                name="LightGBM Regressor",
                task="regression",
                family="Boosting (LightGBM)",
                formula="ŷ = Σₘ η·fₘ(x)  (leaf-wise gradient boosting)",
                notes="Requires lightgbm (optional).",
                requires=("lightgbm",),
                params=(),
                factory=lambda p: None,
            )
        )

    # Stacking Regressor (scikit-learn)
    from sklearn.ensemble import StackingRegressor

    def _stacking_factory(p):
        # Keep this simple and stable: a small ensemble of strong, complementary models.
        base = [
            ("ridge", Ridge(alpha=1.0, random_state=42)),
            ("rf", RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)),
            ("gbr", GradientBoostingRegressor(random_state=42)),
        ]
        final = Ridge(alpha=1.0, random_state=42)
        return StackingRegressor(estimators=base, final_estimator=final, passthrough=_bool(p.get("passthrough"), False), n_jobs=None)

    specs.append(
        ModelSpec(
            name="Stacking Regressor",
            task="regression",
            family="Ensemble (Stacking)",
            formula="ŷ = g(f₁(x), …, fₖ(x))",
            notes="Combines multiple regressors; strong but slower to train.",
            requires=(),
            params=(
                ParamSpec("passthrough", "Passthrough raw features", "bool", False),
            ),
            factory=_stacking_factory,
        )
    )

    return specs


def _classification_specs() -> List[ModelSpec]:
    # Classification support is optional (auto-selected only when target is categorical).
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC

    specs: List[ModelSpec] = []
    specs.append(
        ModelSpec(
            name="Logistic Regression (Classifier)",
            task="classification",
            family="Linear (Classifier)",
            formula="P(y=1|x)=σ(β₀+Σᵢβᵢxᵢ)",
            notes="Strong baseline for categorical targets; fast.",
            requires=(),
            params=(
                ParamSpec("C", "C", "float", 1.0, min=0.01, max=100.0, step=0.1),
            ),
            factory=lambda p: LogisticRegression(
                C=_clamp_float(p.get("C"), 1.0, 1e-6, 1e9),
                max_iter=2000,
                n_jobs=None,
            ),
        )
    )
    specs.append(
        ModelSpec(
            name="Naive Bayes (Gaussian)",
            task="classification",
            family="Naive Bayes",
            formula="P(y|x) ∝ P(y)∏ᵢP(xᵢ|y)",
            notes="Very fast; works well for some categorical targets.",
            requires=(),
            params=(),
            factory=lambda p: GaussianNB(),
        )
    )
    specs.append(
        ModelSpec(
            name="SVM (RBF) Classifier",
            task="classification",
            family="Kernel method",
            formula="max margin in kernel space (RBF)",
            notes="Powerful; can be slow on large datasets.",
            requires=(),
            params=(
                ParamSpec("C", "C", "float", 2.0, min=0.01, max=200.0, step=0.5),
                ParamSpec("gamma", "Gamma", "select", "scale", options=["scale", "auto"]),
            ),
            factory=lambda p: SVC(
                C=_clamp_float(p.get("C"), 2.0, 1e-6, 1e9),
                gamma=(p.get("gamma") if p.get("gamma") in {"scale", "auto"} else "scale"),
                probability=True,
                random_state=42,
            ),
        )
    )
    specs.append(
        ModelSpec(
            name="Random Forest (Classifier)",
            task="classification",
            family="Ensemble (Bagging)",
            formula="majority vote across trees",
            notes="Robust classifier for mixed features.",
            requires=(),
            params=(
                ParamSpec("n_estimators", "Trees", "int", 400, min=50, max=2000, step=50),
            ),
            factory=lambda p: RandomForestClassifier(
                n_estimators=_clamp_int(p.get("n_estimators"), 400, 10, 10_000),
                random_state=42,
                n_jobs=-1,
            ),
        )
    )

    # Optional: stacking classifier
    from sklearn.ensemble import StackingClassifier

    specs.append(
        ModelSpec(
            name="Stacking Classifier",
            task="classification",
            family="Ensemble (Stacking)",
            formula="ŷ = g(f₁(x), …, fₖ(x))",
            notes="Combines multiple classifiers; strong but slower.",
            requires=(),
            params=(
                ParamSpec("passthrough", "Passthrough raw features", "bool", False),
            ),
            factory=lambda p: StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=2000)),
                    ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
                ],
                final_estimator=LogisticRegression(max_iter=2000),
                passthrough=_bool(p.get("passthrough"), False),
                n_jobs=None,
            ),
        )
    )

    # Optional: XGBoost / LightGBM classifiers
    if _has("xgboost"):
        from xgboost import XGBClassifier

        specs.append(
            ModelSpec(
                name="XGBoost Classifier",
                task="classification",
                family="Boosting (XGBoost)",
                formula="ŷ = Σₘ η·fₘ(x)  (gradient-boosted trees)",
                notes="High accuracy; requires xgboost.",
                requires=("xgboost",),
                params=(
                    ParamSpec("n_estimators", "Trees", "int", 800, min=50, max=5000, step=50),
                    ParamSpec("learning_rate", "Learning rate", "float", 0.05, min=0.001, max=1.0, step=0.01),
                    ParamSpec("max_depth", "Max depth", "int", 6, min=2, max=16, step=1),
                ),
                factory=lambda p: XGBClassifier(
                    n_estimators=_clamp_int(p.get("n_estimators"), 800, 10, 50_000),
                    learning_rate=_clamp_float(p.get("learning_rate"), 0.05, 1e-4, 10.0),
                    max_depth=_clamp_int(p.get("max_depth"), 6, 1, 64),
                    subsample=0.9,
                    colsample_bytree=1.0,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="logloss",
                    use_label_encoder=False,
                ),
            )
        )
    else:
        specs.append(
            ModelSpec(
                name="XGBoost Classifier",
                task="classification",
                family="Boosting (XGBoost)",
                formula="ŷ = Σₘ η·fₘ(x)  (gradient-boosted trees)",
                notes="Requires xgboost (optional).",
                requires=("xgboost",),
                params=(),
                factory=lambda p: None,
            )
        )

    if _has("lightgbm"):
        from lightgbm import LGBMClassifier

        specs.append(
            ModelSpec(
                name="LightGBM Classifier",
                task="classification",
                family="Boosting (LightGBM)",
                formula="ŷ = Σₘ η·fₘ(x)  (leaf-wise gradient boosting)",
                notes="High accuracy; requires lightgbm.",
                requires=("lightgbm",),
                params=(
                    ParamSpec("n_estimators", "Trees", "int", 1200, min=50, max=10_000, step=50),
                    ParamSpec("learning_rate", "Learning rate", "float", 0.05, min=0.001, max=1.0, step=0.01),
                ),
                factory=lambda p: LGBMClassifier(
                    n_estimators=_clamp_int(p.get("n_estimators"), 1200, 10, 50_000),
                    learning_rate=_clamp_float(p.get("learning_rate"), 0.05, 1e-4, 10.0),
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        )
    else:
        specs.append(
            ModelSpec(
                name="LightGBM Classifier",
                task="classification",
                family="Boosting (LightGBM)",
                formula="ŷ = Σₘ η·fₘ(x)  (leaf-wise gradient boosting)",
                notes="Requires lightgbm (optional).",
                requires=("lightgbm",),
                params=(),
                factory=lambda p: None,
            )
        )

    return specs


def list_model_specs() -> List[ModelSpec]:
    return [*_regression_specs(), *_classification_specs()]


def get_catalog() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in list_model_specs():
        missing = _missing(s.requires)
        out.append(
            {
                "name": s.name,
                "task": s.task,
                "family": s.family,
                "formula": s.formula,
                "notes": s.notes,
                "available": len(missing) == 0,
                "missing_deps": missing,
                "params": [
                    {
                        "key": p.key,
                        "label": p.label,
                        "kind": p.kind,
                        "default": p.default,
                        "help": p.help,
                        "min": p.min,
                        "max": p.max,
                        "step": p.step,
                        "options": p.options,
                    }
                    for p in s.params
                ],
            }
        )
    return out


def available_models(task: Optional[str] = None) -> Dict[str, str]:
    """
    Returns installed/usable models only.
    - task=None: all available (regression + classification)
    - task="regression" or "classification": filters by task
    """
    out: Dict[str, str] = {}
    for s in list_model_specs():
        if task and s.task != task:
            continue
        if _missing(s.requires):
            continue
        out[s.name] = s.name
    return out


def get_estimator(model_name: str, params: Optional[Dict[str, Any]] = None, task: Optional[str] = None):
    name = (model_name or "").strip()
    params = params or {}
    for s in list_model_specs():
        if s.name != name:
            continue
        if task and s.task != task:
            raise ValueError(f"Model '{name}' is for {s.task}, not {task}.")
        missing = _missing(s.requires)
        if missing:
            raise ValueError(f"Model '{name}' is not available. Missing dependency: {', '.join(missing)}.")
        est = s.factory(params or {})
        if est is None:
            raise ValueError(f"Model '{name}' is not available in this environment.")
        return est
    raise ValueError("Unknown model name.")


def get_model_spec(model_name: str) -> ModelSpec:
    name = (model_name or "").strip()
    for s in list_model_specs():
        if s.name == name:
            return s
    raise ValueError("Unknown model name.")


def get_model_task(model_name: str) -> str:
    return get_model_spec(model_name).task
