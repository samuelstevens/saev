import logging
import tomllib
import typing

import beartype
import tyro

from . import config

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger("contrib.semseg")


@beartype.beartype
def train(
    cfg: typing.Annotated[config.Train, tyro.conf.arg(name="")],
    sweep: str | None = None,
):
    import submitit

    from . import training

    if sweep is not None:
        with open(sweep, "rb") as fd:
            cfgs, errs = config.grid(cfg, tomllib.load(fd))

        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
            return

    else:
        cfgs = [cfg]

    logger.info("Running %d training jobs.", len(cfgs))

    executor = submitit.DebugExecutor(folder=cfg.log_to)

    job = executor.submit(training.main, cfgs)
    job.result()


@beartype.beartype
def visuals(cfg: typing.Annotated[config.Visuals, tyro.conf.arg(name="")]):
    from . import visuals

    visuals.main(cfg)


@beartype.beartype
def validate(cfg: typing.Annotated[config.Validation, tyro.conf.arg(name="")]):
    from . import validation

    validation.main(cfg)


@beartype.beartype
def manipulate(cfg: typing.Annotated[config.Manipulation, tyro.conf.arg(name="")]):
    from . import manipulation

    manipulation.main(cfg)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "train": train,
        "visuals": visuals,
        "validate": validate,
        "manipulate": manipulate,
    })
