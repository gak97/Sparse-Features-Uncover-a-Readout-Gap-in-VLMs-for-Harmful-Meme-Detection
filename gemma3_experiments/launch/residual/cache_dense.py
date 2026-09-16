import logging

from simple_parsing import parse

from gemma3_experiments.residual_sae import DenseActivationCacheConfig, run_dense_activation_cache


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse(DenseActivationCacheConfig)
    run_dense_activation_cache(cfg)


if __name__ == "__main__":
    main()
