from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Replace this with your actual path
event_file = '/home/aubreygress/tensorboard_vit-experiment_run-beans-vit-v1-20250721-055303_events.out.tfevents.1753102938.cmle-training-workerpool0-b12a5e5737-0-cd5sv.230 (1).0'

# Load the event file
ea = EventAccumulator(event_file)
ea.Reload()

# Print all available tags
print("Available tags:", ea.Tags())

# Print scalar data
if 'scalars' in ea.Tags():
    for tag in ea.Tags()['scalars']:
        print(f"\n== Scalar tag: {tag} ==")
        for event in ea.Scalars(tag):
            print(f"Step: {event.step}, Value: {event.value}")
else:
    print("No scalar data found.")
