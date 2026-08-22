from maadpg_reproduction.schema import (
    OBSERVATION_ACTION_SCHEMA,
    observation_action_schema_sha256,
)


def test_observation_action_schema_is_stable_and_complete():
    assert OBSERVATION_ACTION_SCHEMA["pursuer_observation"]["shape"] == [3, 26]
    assert OBSERVATION_ACTION_SCHEMA["joint_action"]["shape"] == [3, 2]
    assert OBSERVATION_ACTION_SCHEMA["joint_critic_input"]["shape"] == [84]
    assert (
        OBSERVATION_ACTION_SCHEMA["joint_action"]["replay_value"]
        == "exact_environment_executed_action"
    )
    assert len(observation_action_schema_sha256()) == 64
    assert observation_action_schema_sha256() == observation_action_schema_sha256()
