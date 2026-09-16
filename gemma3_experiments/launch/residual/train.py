import logging

from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.residual_sae import ResidualSaeTrainConfig, train_residual_sae


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse(ResidualSaeTrainConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    train_residual_sae(cfg)


if __name__ == "__main__":
    main()
