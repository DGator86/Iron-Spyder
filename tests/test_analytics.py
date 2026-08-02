from analytics.metrics import call_wall, gamma_flip, gamma_profile, put_wall
from data.synthetic import generate_option_chain


def test_gamma_profile_and_flip() -> None:
    chain = generate_option_chain()
    profile = gamma_profile(chain, [530, 540, 550, 560, 570])
    assert len(profile) == 5
    _ = gamma_flip(profile)


def test_wall_detection() -> None:
    chain = generate_option_chain()
    assert call_wall(chain) > 0
    assert put_wall(chain) > 0
