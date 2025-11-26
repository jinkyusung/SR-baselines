import wandb
from utils import configurator
from loader import load_data_object


def main():
    config = configurator.load_config()
    config.wandb.run_id = wandb.util.generate_id() if config.wandb.run_id == 'None' else config.wandb.run_id
    wandb_name = f"{config.model.name}:{config.dataset.name}:{config.wandb.run_id}"
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=wandb_name,
        id=config.wandb.run_id,
        resume="allow",
        config=config
    )

    data_obj = load_data_object(dataset=config.dataset.name, k_core=config.dataset.k_core)
    print(data_obj)

if __name__ == '__main__':
    main()
