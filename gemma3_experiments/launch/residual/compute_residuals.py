from simple_parsing import parse

from gemma3_experiments.residual_sae import ResidualCacheConfig, compute_residual_shards


def main() -> None:
    cfg = parse(ResidualCacheConfig)
    compute_residual_shards(cfg)


if __name__ == "__main__":
    main()
