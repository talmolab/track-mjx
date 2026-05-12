"""Imports the train_dmpo module to catch syntax / import errors early."""


def test_import_train_dmpo():
    from track_mjx.agent.dmpo import train_dmpo  # noqa: F401
    assert hasattr(train_dmpo, "main")
