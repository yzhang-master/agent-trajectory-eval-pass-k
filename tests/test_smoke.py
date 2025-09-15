from trajectory_eval import healthcheck


def test_package_smoke() -> None:
    assert healthcheck()
