import subprocess

nodes = ["node1", "node2"]

for node in nodes:
    print(f"--- Monitoring {node} ---")
    subprocess.run(["docker", "exec", node, "top", "-b", "-n", "1"])
