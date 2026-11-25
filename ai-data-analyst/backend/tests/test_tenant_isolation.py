"""One user must never reach another user's models, files or the platform itself.

Each test pins an attack that worked before this file existed:
- the model registry was one shared folder: any user could list, load, score with and promote
  anyone's model, and a column called "../../x" chose where a model was written;
- dataset SQL ran where DuckDB could read any file on the server (other users' uploads, .env);
- a saved connection could open the platform's own database or reach internal hosts;
- maintenance endpoints that act on every tenant's files were open to any analyst;
- API keys could be minted with admin rights and revoked by anyone.
"""

from __future__ import annotations

import json
import threading
import time
from uuid import UUID, uuid4

import httpx
import numpy as np
import pandas as pd
import pytest
import pytest_asyncio

from app.core.config import AuthMode, settings
from app.ml.model_registry import (
    ModelRegistry,
    ValidationException,
    get_model_registry,
    migrate_shared_registry,
    model_dir_name,
)

A = UUID("00000000-0000-0000-0000-00000000000a")
B = UUID("00000000-0000-0000-0000-00000000000b")


@pytest.fixture()
def registry_root(tmp_path, monkeypatch):
    root = tmp_path / "registry"
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(root))
    return root


def _model():
    from sklearn.dummy import DummyClassifier

    return DummyClassifier(strategy="most_frequent").fit(pd.DataFrame({"x": [1, 2, 3]}), ["a", "b", "b"])


# ---------------------------------------------------------------------------
# Registry: one folder per owner
# ---------------------------------------------------------------------------

def test_one_owners_models_do_not_exist_for_another(registry_root):
    get_model_registry(A).register_model("churn", _model(), tags={"dataset_id": str(uuid4())})
    assert [m["name"] for m in get_model_registry(A).list_models()] == ["churn"]
    other = get_model_registry(B)
    assert other.list_models() == []
    assert other.get_model_versions("churn") == []
    with pytest.raises(ValidationException):
        other.load_model("churn")
    with pytest.raises(ValidationException):
        get_model_registry(None)
    with pytest.raises(ValidationException):
        get_model_registry("../../etc")


def test_model_names_from_column_headers_cannot_escape_the_registry(registry_root):
    reg = get_model_registry(A)
    for hostile in ["../../escape", "..\\..\\escape", "CON", "a/b", "ünïcode name", "x" * 150]:
        version = reg.register_model(hostile, _model())
        assert (reg.storage_path / version.model_path).resolve().is_relative_to(reg.storage_path.resolve())
        assert reg.load_model(hostile).predict(pd.DataFrame({"x": [1]}))[0] == "b"
    assert not (registry_root.parent / "escape").exists()
    assert model_dir_name("../../escape") != model_dir_name("__escape")  # no two names share a folder
    with pytest.raises(ValidationException):
        reg.register_model("bad\x00name", _model())


def test_a_tampered_registry_entry_cannot_point_outside_it(registry_root, tmp_path):
    reg = get_model_registry(A)
    reg.register_model("m", _model())
    outside = tmp_path / "planted.pkl"
    outside.write_bytes(b"not a model")
    index = reg.storage_path / "registry.json"
    data = json.loads(index.read_text(encoding="utf-8"))
    data["models"][0]["versions"][0]["model_path"] = str(outside)
    index.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValidationException, match="outside"):
        ModelRegistry(reg.storage_path).load_model("m")


def test_concurrent_trainings_never_lose_or_overwrite_a_version(registry_root):
    errors: list[Exception] = []

    def train(i: int) -> None:
        try:
            # A fresh instance per thread behaves like separate worker processes.
            ModelRegistry(get_model_registry(A).storage_path).register_model("shared", {"run": i})
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=train, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    reg = get_model_registry(A)
    versions = [v["version"] for v in reg.get_model_versions("shared")]
    assert versions == list(range(1, 9))
    assert sorted(reg.load_model("shared", version=v)["run"] for v in versions) == list(range(8))


def test_shared_registry_models_move_to_their_owner(registry_root):
    ds_a, gone = str(uuid4()), str(uuid4())
    legacy = ModelRegistry(registry_root)
    legacy.register_model("ds_a_churn", _model(), tags={"dataset_id": ds_a})
    legacy.register_model("ds_a_churn", _model(), tags={"dataset_id": ds_a})
    legacy.register_model("orphan", _model(), tags={"dataset_id": gone})

    report = migrate_shared_registry({ds_a: A}.get)
    assert report == {"moved": ["ds_a_churn"], "unclaimed": ["orphan"], "conflicts": []}
    mine = get_model_registry(A)
    assert [v["version"] for v in mine.get_model_versions("ds_a_churn")] == [1, 2]
    assert mine.load_model("ds_a_churn", version=2).predict(pd.DataFrame({"x": [1]}))[0] == "b"
    assert get_model_registry(B).list_models() == []
    assert [m["name"] for m in ModelRegistry(registry_root).list_models()] == ["orphan"]
    assert migrate_shared_registry({ds_a: A}.get)["moved"] == []  # idempotent


def test_deleting_a_dataset_deletes_the_models_trained_on_it(registry_root):
    ds, other = str(uuid4()), str(uuid4())
    reg = get_model_registry(A)
    reg.register_model("on_ds", _model(), tags={"dataset_id": ds})
    reg.register_model("on_other", _model(), tags={"dataset_id": other})
    folder = reg.storage_path / model_dir_name("on_ds")
    assert folder.is_dir()
    assert reg.delete_models_for_dataset(ds) == ["on_ds"]
    assert not folder.exists()
    assert [m["name"] for m in get_model_registry(A).list_models()] == ["on_other"]


def test_version_comparison_recommends_on_cross_validated_score_only(registry_root):
    reg = get_model_registry(A)
    reg.register_model("m", _model(), metrics={"cv_neg_root_mean_squared_error": -12.0, "rmse": 12.0, "r2": 0.5})
    reg.register_model("m", _model(), metrics={"cv_neg_root_mean_squared_error": -9.0, "rmse": 9.0, "r2": 0.7})
    out = reg.compare_versions("m", 1, 2)
    assert out["recommendation"] == "v2" and out["recommendation_basis"] == "cv_neg_root_mean_squared_error"
    reg.register_model("n", _model(), metrics={"rmse": 9.0})
    reg.register_model("n", _model(), metrics={"rmse": 12.0})
    assert reg.compare_versions("n", 1, 2)["recommendation"] is None


# ---------------------------------------------------------------------------
# Dataset SQL: the query cannot reach any file but its own dataset
# ---------------------------------------------------------------------------

def test_dataset_sql_cannot_read_other_files_even_past_the_text_check(tmp_path, monkeypatch):
    from app.services import dataset_sql

    monkeypatch.setattr(dataset_sql, "_cache_dir", lambda: (tmp_path / "cache").mkdir(exist_ok=True) or tmp_path / "cache")
    data = tmp_path / "mine.csv"
    data.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    secret = tmp_path / "someone_else.csv"
    secret.write_text("password\nhunter2\n", encoding="utf-8")

    df, _ = dataset_sql.run_dataset_sql(file_path=str(data), file_format="csv", sql="SELECT sum(a) AS s FROM dataset")
    assert int(df["s"].iloc[0]) == 4
    # These skip sql_safety on purpose: the sandbox must hold on its own.
    for attack in [
        f"SELECT * FROM read_text('{secret.as_posix()}')",
        f"SELECT * FROM read_csv_auto('{secret.as_posix()}')",
        f"SELECT * FROM glob('{tmp_path.as_posix()}/*')",
        f"SELECT * FROM read_blob('{secret.as_posix()}')",
        "SET enable_external_access = true",
    ]:
        with pytest.raises(Exception) as err:
            dataset_sql.run_dataset_sql(file_path=str(data), file_format="csv", sql=attack)
        assert "hunter2" not in str(err.value)


def test_text_check_blocks_known_file_functions_and_comment_tricks():
    from app.services.sql_safety import UnsafeSQLError, validate_dataset_sql

    for q in ["SELECT * FROM glob('*')", "SELECT * FROM read_text('.env')", "SELECT * FROM/**/'x.csv'",
              "SELECT * FROM read_ndjson_auto('x')", "SELECT getenv('HOME')"]:
        with pytest.raises(UnsafeSQLError):
            validate_dataset_sql(q)
    assert validate_dataset_sql("SELECT load, region FROM dataset")  # ordinary column names still work


def test_purge_removes_the_sql_copy_of_a_deleted_dataset(tmp_path, monkeypatch):
    from app.services import dataset_sql

    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(dataset_sql, "_cache_dir", lambda: cache)
    data = tmp_path / "d.csv"
    data.write_text("a\n1\n", encoding="utf-8")
    dataset_sql.run_dataset_sql(file_path=str(data), file_format="csv", sql="SELECT * FROM dataset")
    assert list(cache.glob("*.duckdb"))
    assert dataset_sql.prune_orphans() == 0  # source still there
    data.unlink()
    assert dataset_sql.prune_orphans() == 1
    assert not list(cache.glob("*"))


# ---------------------------------------------------------------------------
# Connections: never the platform database, never internal hosts with sign-in on
# ---------------------------------------------------------------------------

@pytest.fixture()
def multi_user(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "auth_mode", AuthMode.JWT)
    monkeypatch.setattr(settings.connections, "allow_private_hosts", False)
    monkeypatch.setattr(settings.connections, "sqlite_root", None)
    platform = tmp_path / "platform.db"
    monkeypatch.setattr(settings.database, "url", f"sqlite+aiosqlite:///{platform.as_posix()}")
    return tmp_path


def test_connections_cannot_reach_the_platform_or_internal_hosts(multi_user, monkeypatch):
    from app.services.connection_policy import ConnectionNotAllowed, assert_connection_allowed

    refused = [
        ("sqlite", "", 0, str(multi_user / "platform.db")),       # the platform's own database
        ("sqlite", "", 0, str(multi_user / "anything.db")),       # SQLite files off by default
        ("sqlite", "", 0, "file:x.db?mode=ro"),
        ("postgresql", "127.0.0.1", 5432, "postgres"),
        ("postgresql", "localhost", 5432, "postgres"),
        ("postgresql", "169.254.169.254", 80, "x"),              # cloud metadata
        ("postgresql", "10.0.0.5", 5432, "x"),
        ("mysql", "[::1]", 3306, "x"),
        ("mysql", "::ffff:127.0.0.1", 3306, "x"),
        ("oracle", "8.8.8.8", 1521, "x"),
    ]
    for case in refused:
        with pytest.raises(ConnectionNotAllowed):
            assert_connection_allowed(*case)
    assert_connection_allowed("postgresql", "8.8.8.8", 5432, "warehouse")  # public host is fine
    assert_connection_allowed("sqlite", "", 0, ":memory:")

    monkeypatch.setattr(settings.connections, "sqlite_root", multi_user / "allowed")
    (multi_user / "allowed").mkdir()
    assert_connection_allowed("sqlite", "", 0, str(multi_user / "allowed" / "mine.db"))
    with pytest.raises(ConnectionNotAllowed):
        assert_connection_allowed("sqlite", "", 0, str(multi_user / "allowed" / ".." / "platform.db"))

    monkeypatch.setattr(settings.connections, "allow_private_hosts", True)
    assert_connection_allowed("postgresql", "10.0.0.5", 5432, "x")  # self-hosted team network


def test_single_user_mode_allows_local_databases_but_not_the_platform(multi_user, monkeypatch):
    from app.services.connection_policy import ConnectionNotAllowed, assert_connection_allowed

    monkeypatch.setattr(settings, "auth_mode", AuthMode.LOCAL)
    assert_connection_allowed("sqlite", "", 0, str(multi_user / "mine.db"))
    assert_connection_allowed("postgresql", "127.0.0.1", 6543, "x")
    with pytest.raises(ConnectionNotAllowed):
        assert_connection_allowed("sqlite", "", 0, str(multi_user / "platform.db"))


def test_a_public_or_saas_server_refuses_to_start_without_sign_in(monkeypatch):
    from app.core.config import DeploymentMode, Environment
    from app.main import validate_startup_security

    monkeypatch.setattr(settings, "auth_mode", AuthMode.LOCAL)
    monkeypatch.setattr(settings, "environment", Environment.PRODUCTION)
    with pytest.raises(RuntimeError, match="AUTH_MODE=jwt"):
        validate_startup_security()
    monkeypatch.setattr(settings, "environment", Environment.DEVELOPMENT)
    monkeypatch.setattr(settings, "deployment_mode", DeploymentMode.SAAS)
    with pytest.raises(RuntimeError, match="AUTH_MODE=jwt"):
        validate_startup_security()
    monkeypatch.setattr(settings, "deployment_mode", DeploymentMode.SELF_HOST)
    validate_startup_security()  # a local development server is fine


def test_billing_rejects_replayed_webhooks_and_foreign_return_urls():
    import hashlib
    import hmac

    from app.api.routes.billing import _same_site, _verify_stripe_signature

    payload, secret = b'{"id": "evt_1"}', "whsec_test"

    def header(ts: int) -> str:
        sig = hmac.new(secret.encode(), f"{ts}.{payload.decode()}".encode(), hashlib.sha256).hexdigest()
        return f"t={ts},v1={sig}"

    assert _verify_stripe_signature(payload, header(int(time.time())), secret)
    assert not _verify_stripe_signature(payload, header(int(time.time()) - 3600), secret)
    assert _same_site("https://app.example.com/done", "https://app.example.com")
    assert not _same_site("https://evil.example/phish", "https://app.example.com")
    assert not _same_site("javascript:alert(1)", "https://app.example.com")


# ---------------------------------------------------------------------------
# End to end: two signed-in users on one server
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture()
async def two_users(tmp_path, monkeypatch):
    from app.main import create_application
    from app.services.database import db_manager

    (tmp_path / "uploads").mkdir()
    (tmp_path / "artifacts").mkdir()
    monkeypatch.setattr(settings.database, "url", f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setattr(settings, "upload_directory", tmp_path / "uploads")
    monkeypatch.setattr(settings, "artifact_directory", tmp_path / "artifacts")
    monkeypatch.setattr(settings, "auth_mode", AuthMode.JWT)
    monkeypatch.setattr(settings.security, "rate_limit_requests", 10_000)
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(tmp_path / "registry"))
    await db_manager.close()
    app = create_application()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                users = []
                for name in ("alice", "bob"):
                    creds = {"email": f"{name}@example.com", "password": "Str0ngPassw0rd!"}
                    await client.post("/api/v1/auth/register", json={**creds, "role": "analyst", "full_name": name})
                    login = await client.post("/api/v1/auth/login", json=creds)
                    assert login.status_code == 200, login.text
                    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
                    me = await client.get("/api/v1/auth/me", headers=headers)
                    users.append((headers, UUID(me.json()["user_id"])))
                yield client, users
    finally:
        await db_manager.close()
        try:
            from app.main import RateLimitMiddleware

            RateLimitMiddleware._requests.clear()
        except Exception:
            pass


async def _upload(client, headers) -> str:
    from app.services.dataset_processing import DatasetProcessingService

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": rng.integers(0, 100, 80), "y": rng.choice(["a", "b"], 80)})
    up = await client.post("/api/v1/datasets/upload", headers=headers,
                           files={"file": ("d.csv", frame.to_csv(index=False).encode(), "text/csv")},
                           data={"name": "D"})
    assert up.status_code == 201, up.text
    data = up.json()["data"]
    await DatasetProcessingService().process_dataset(UUID(data["dataset_id"]), UUID(data["job_id"]))
    return data["dataset_id"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_another_users_model_is_invisible_through_every_model_route(two_users):
    client, [(alice, alice_id), (bob, _)] = two_users
    get_model_registry(alice_id).register_model("alice_model", _model())
    row = {"model_name": "alice_model", "data": [{"x": 1}]}

    mine = await client.post("/api/v1/ml/predict", headers=alice, json=row)
    assert mine.status_code == 200, mine.text
    assert (await client.get("/api/v1/ml/models", headers=alice)).json()["total"] == 1

    assert (await client.get("/api/v1/ml/models", headers=bob)).json() == {"models": [], "total": 0}
    for method, url, body in [
        ("get", "/api/v1/ml/models/alice_model", None),
        ("post", "/api/v1/ml/predict", row),
        ("post", "/api/v1/ml/explain", row),
        ("post", "/api/v1/ml/models/alice_model/versions/1/promote", None),
        ("post", "/api/v1/ml/models/alice_model/compare?version1=1&version2=1", None),
    ]:
        res = await getattr(client, method)(url, headers=bob, **({"json": body} if body else {}))
        assert res.status_code in {403, 404}, (url, res.status_code, res.text)
    # Bob's promote attempt must not have changed Alice's model.
    versions = get_model_registry(alice_id).get_model_versions("alice_model")
    assert versions[0]["stage"] == "development"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_training_on_another_users_dataset_is_refused_before_any_job(two_users):
    client, [(alice, _), (bob, bob_id)] = two_users
    dataset_id = await _upload(client, alice)
    for mode in ("baseline", "automl"):
        res = await client.post("/api/v1/ml/train", headers=bob,
                                json={"dataset_id": dataset_id, "target_column": "y", "mode": mode})
        assert res.status_code == 404, (mode, res.text)
    res = await client.post("/api/v1/ml/automl", headers=bob, json={"dataset_id": dataset_id, "target_column": "y"})
    assert res.status_code == 404, res.text
    jobs = await client.get("/api/v1/jobs", headers=bob)
    assert "ml_training" not in jobs.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dataset_sql_over_the_api_cannot_list_other_uploads(two_users):
    client, [(alice, _), (bob, _)] = two_users
    await _upload(client, alice)
    bob_ds = await _upload(client, bob)
    ok = await client.post(f"/api/v1/datasets/{bob_ds}/query", headers=bob, json={"query": "select count(*) as n from dataset"})
    assert ok.status_code == 200, ok.text
    for q in ["select * from glob('uploads/*')", "select * from read_text('.env')", "select * from/**/'dev.db'"]:
        res = await client.post(f"/api/v1/datasets/{bob_ds}/query", headers=bob, json={"query": q})
        assert res.status_code == 400, (q, res.text)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_connections_api_refuses_the_platform_database(two_users, tmp_path):
    client, [(_, _), (bob, _)] = two_users
    res = await client.post("/api/v1/connections", headers=bob, json={
        "name": "steal", "connector_type": "sqlite", "host": "", "port": 0,
        "database": str(tmp_path / "test.db"), "username": "", "password": "",
    })
    assert res.status_code == 400, res.text
    res = await client.post("/api/v1/connections", headers=bob, json={
        "name": "metadata", "connector_type": "postgresql", "host": "169.254.169.254", "port": 80,
        "database": "x", "username": "u", "password": "p",
    })
    assert res.status_code == 400, res.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_analysts_cannot_run_platform_maintenance_or_mint_admin_keys(two_users):
    client, [(alice, _), (bob, _)] = two_users
    gc = await client.post("/api/v1/maintenance/storage-gc", headers=bob, json={"dry_run": True, "min_age_days": 0})
    assert gc.status_code == 403, gc.text
    assert (await client.post("/api/v1/maintenance/cache-prune", headers=bob)).status_code == 403

    admin_key = await client.post("/api/v1/auth/api-keys", headers=bob, json={"name": "k", "permissions": ["admin:*"]})
    assert admin_key.status_code in {400, 403}, admin_key.text
    alice_key = await client.post("/api/v1/auth/api-keys", headers=alice, json={"name": "mine"})
    assert alice_key.status_code == 200, alice_key.text
    key_id = alice_key.json()["key_id"]
    assert (await client.delete(f"/api/v1/auth/api-keys/{key_id}", headers=bob)).status_code == 404
    assert (await client.delete(f"/api/v1/auth/api-keys/{key_id}", headers=alice)).status_code == 200

    metrics = await client.get("/api/v1/metrics", headers=bob)
    if metrics.status_code == 200:
        data = metrics.json()["data"]
        assert "operators" not in data and "http" not in data
