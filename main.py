import wandb
from utils import configurator


def main():
    config = configurator.load_config()
    config.wandb.run_id = wandb.util.generate_id() if config.wandb.run_id == 'None' else config.wandb.run_id
    wandb_name = f"{config.model}:{config.dataset}:{config.wandb.run_id}"
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=wandb_name,
        id=config.wandb.run_id,
        resume="allow",
        config=config
    )


if __name__ == '__main__':
    main()
