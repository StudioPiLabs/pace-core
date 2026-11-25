from pace_core.breakdown.derive_lighting_state import derive


def shot(n, text, lighting=None):
    return {"shot_id": f"shot_{n:02d}", "events": {"actions": [{"description_en": text}]},
            **({"lighting": lighting} if lighting is not None else {})}


def test_blackout_runs_from_the_loss_until_the_light_returns():
    scene = {"shots": [shot(1, "Lucas says, \"Whatever.\""),
                       shot(2, "All the panels lose power, plunging the surroundings into pitch black"),
                       shot(3, "Lucas nervously asks what just happened"),
                       shot(4, "The power comes back and the screens wake"),
                       shot(5, "They drive on")]}
    assert derive(scene) == ["shot_02", "shot_03"]
    assert [(s.get("lighting") or {}).get("condition") for s in scene["shots"]] == [
        None, "blackout", "blackout", None, None]


def test_an_authored_condition_is_kept():
    scene = {"shots": [shot(1, "the car loses power"), shot(2, "a match is struck", {"condition": "candlelight"})]}
    assert derive(scene) == ["shot_01"]
    assert scene["shots"][1]["lighting"]["condition"] == "candlelight"


def test_a_lit_scene_is_untouched():
    scene = {"shots": [shot(1, "The screen shows a family on a beach"), shot(2, "The power steering hums")]}
    assert derive(scene) == []
    assert all("lighting" not in s for s in scene["shots"])
