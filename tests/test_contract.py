"""The /v1/systemone contract, with a fake backend. No weights."""
import json
import pathlib
import sys
import threading
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from jevtoo.backends import common_prefix_len
from jevtoo.contract import ContractError, evaluate
from jevtoo.decide import DecisionModel
from jevtoo.readout import Distribution
from jevtoo.serve import create_server


class FakeBackend:
    def __init__(self):
        self.prompts = []

    def distribution(self, prompt, labels):
        self.prompts.append(prompt)
        n = len(labels)
        raw = [0.05] * n
        raw[-1] = 0.4
        return Distribution(
            labels=list(labels), probs=list(raw), raw=list(raw),
            latency_s=0.012, n_tokens=len(prompt.split()), missing=0,
        )

    def coverage(self):
        return 1.0

    stats = {"head_mode": "full"}


def _model():
    return DecisionModel(backend=FakeBackend())


def _body(**overrides):
    body = {
        "model": "jev-latest",
        "state": "Customer: I was charged twice.",
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": "What is the issue about?",
                "criteria": {"billing": "money problems", "bug": "broken product"},
            },
            "urgent": {"type": "noul", "instructions": "Escalate to a human now?"},
            "heat": {
                "type": "score",
                "instructions": "How frustrated?",
                "criteria": ["Calm", "Annoyed", "Furious"],
            },
        },
    }
    body.update(overrides)
    return body


def test_common_prefix_len_stops_at_the_first_difference():
    assert common_prefix_len([[1, 2, 3, 4], [1, 2, 9]]) == 2
    assert common_prefix_len([[1], [1, 2]]) == 1
    assert common_prefix_len([[7, 7], [8, 7]]) == 0
    assert common_prefix_len([]) == 0


def test_response_matches_the_published_shape():
    model = _model()
    payload, obs = evaluate(model, _body(), model_id="org/model", aliases={"jev-latest"}, observe=False)
    assert "observability" not in payload
    assert payload["model"] == "org/model"
    assert payload["usage"]["output_tokens"] == 3
    topic = payload["answers"]["topic"]
    assert topic["type"] == "choice"
    assert topic["choice"] == "bug"  # last label carries the mass
    assert abs(sum(topic["probabilities"].values()) - 1.0) < 1e-6
    # shares are 0.05/0.45 and 0.40/0.45. peakedness for n=2: (2*peak - 1)
    assert abs(topic["confidence"] - ((2 * (0.4 / 0.45)) - 1)) < 1e-5
    urgent = payload["answers"]["urgent"]
    assert urgent == {"type": "noul", "noul": round(0.4 / 0.45, 6)}
    heat = payload["answers"]["heat"]
    assert heat["legend"] == {"0": "Calm", "1": "Annoyed", "2": "Furious"}
    assert abs(heat["score"] - (0 * 0.1 + 1 * 0.1 + 2 * 0.8)) < 1e-5
    assert obs["questions"]["urgent"]["binary_position_prior"] is True
    assert obs["questions"]["topic"]["binary_position_prior"] is True  # two options
    assert obs["questions"]["urgent"]["option_mass"] == round(0.45, 6)
    assert "binary_position_prior" not in obs["questions"]["heat"]


def test_observe_flag_attaches_the_same_object_the_server_logs():
    payload, obs = evaluate(
        _model(), _body(), model_id="org/model", aliases={"jev-latest"}, observe=True,
    )
    assert payload["observability"] is obs
    assert "winner_probability" in obs["questions"]["topic"]
    assert "peakedness" in obs["questions"]["topic"]


def test_state_object_is_serialized_into_every_question():
    model = _model()
    evaluate(
        model, _body(state={"b": 1, "a": 2}),
        model_id="org/model", aliases={"jev-latest"}, observe=False,
    )
    assert len(model.backend.prompts) == 3
    for prompt in model.backend.prompts:
        assert prompt.startswith('{"a": 2, "b": 1}')


def test_unknown_model_and_missing_fields_are_422():
    model = _model()
    try:
        evaluate(model, _body(model="other"), model_id="org/model", aliases={"jev-latest"}, observe=False)
        raise AssertionError("expected ContractError")
    except ContractError as exc:
        assert exc.field == "model"
    try:
        evaluate(model, {"model": "jev-latest", "questions": {}}, model_id="m", aliases={"jev-latest"}, observe=False)
        raise AssertionError("expected ContractError")
    except ContractError as exc:
        assert exc.field == "state"


def test_choice_over_26_options_is_rejected():
    criteria = {f"o{i}": "x" for i in range(27)}
    body = _body(questions={"wide": {"type": "choice", "instructions": "Which?", "criteria": criteria}})
    try:
        evaluate(_model(), body, model_id="m", aliases={"jev-latest"}, observe=False)
        raise AssertionError("expected ContractError")
    except ContractError as exc:
        assert "26" in exc.message


def test_gguf_missing_label_retries_once_with_a_wider_window():
    class Partial:
        def __init__(self):
            self.n_probs = 200
            self.calls = 0
            self.stats = {}

        def distribution(self, prompt, labels):
            self.calls += 1
            missing = 0 if self.n_probs >= 1000 else 1
            raw = [0.1, 0.9] if missing == 0 else [0.0, 0.9]
            return Distribution(
                labels=list(labels), probs=raw, raw=list(raw),
                latency_s=0.001, n_tokens=4, missing=missing,
            )

        def coverage(self):
            return 1.0

    backend = Partial()
    body = {"model": "m", "state": "s", "questions": {
        "urgent": {"type": "noul", "instructions": "Now?"},
    }}
    payload, obs = evaluate(
        DecisionModel(backend=backend), body, model_id="m", aliases=set(), observe=True,
    )
    assert backend.calls == 2
    assert backend.n_probs == 200  # restored
    assert payload["answers"]["urgent"]["noul"] == round(0.9 / 1.0, 6)
    assert obs["questions"]["urgent"]["missing_labels"] == 0


def _post(port, body, headers=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), json.loads(exc.read().decode())


def test_server_auth_observe_and_metrics():
    httpd = create_server(
        _model(), "org/model", host="127.0.0.1", port=0, api_key="secret", backend_name="fake",
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        status, _, payload = _post(port, _body())
        assert status == 401, payload

        status, headers, payload = _post(port, _body(), {
            "Authorization": "Bearer secret",
            "X-Jev-Observe": "1",
        })
        assert status == 200, payload
        assert payload["model"] == "org/model"
        assert payload["observability"]["queue_ms"] >= 0
        assert "X-Request-Id" in {k.title() if False else k for k in headers} or any(
            k.lower() == "x-request-id" for k in headers
        )
        assert any(k.lower() == "server-timing" for k in headers)

        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5)
            raise AssertionError("metrics without a key should be rejected")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        metrics_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/metrics",
            headers={"Authorization": "Bearer secret"},
        )
        with urllib.request.urlopen(metrics_req, timeout=5) as resp:
            text = resp.read().decode()
        assert "jevtoo_requests_total" in text
        assert "jevtoo_questions_total 3" in text

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
            health = json.loads(resp.read().decode())
        assert health["model"] == "org/model"
        assert health["endpoint"] == "/v1/systemone"
    finally:
        httpd.shutdown()
        httpd.server_close()
