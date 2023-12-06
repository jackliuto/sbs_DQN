import wandb
api = wandb.Api()
run = api.run("/mulab/lowerbound_dqn/runs/yopjgzja")

print(run.history())