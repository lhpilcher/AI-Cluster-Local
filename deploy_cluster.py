import subprocess

# Names of your nodes (containers)
nodes = ["node1", "node2"]

# Step 1: Build Docker image
print("Building Docker image...")
subprocess.run(["docker", "build", "-t", "ai_node", "."])

# Step 2: Create and start containers
for node in nodes:
    # Check if container already exists
    result = subprocess.run(["docker", "ps", "-a", "-q", "-f", f"name={node}"], capture_output=True, text=True)
    if result.stdout.strip():
        print(f"Container {node} already exists. Removing it...")
        subprocess.run(["docker", "rm", "-f", node])

    print(f"Starting container {node}...")
    subprocess.run([
        "docker", "run", "-dit",
        "--name", node,
        "ai_node",
        "bash"
    ])

# Step 3: Run training inside each container
for node in nodes:
    print(f"Running training on {node}...")
    subprocess.run(["docker", "exec", node, "python", "/app/train_model.py"])
