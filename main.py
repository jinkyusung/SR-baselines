import wandb
from utils import configurator


def main():
    config = configurator.load_config()
    wandb.init(entity=config.wandb.entity, project=config.wandb.project, config=config)


if __name__ == '__main__':
    main()
