from dualgoose.probes import last_write_replacement_probe, timestep_merge_routing_probe


def test_rank_replacement_separates_histories_diagonal_collides():
    result = last_write_replacement_probe()
    assert result["rank_accuracy"] == 1.0
    assert result["diagonal_accuracy"] == 0.5
    assert result["diagonal_collision"] is True
    assert result["rank_separates_histories"] is True


def test_timestep_merge_routes_identical_streams_by_timestep():
    result = timestep_merge_routing_probe(steps=200, d_model=4, batch_size=32, lr=0.05)
    assert result["film_beats_static"] is True
    assert result["film"]["final_mse"] < 0.05
    assert result["static"]["final_mse"] > 0.2
