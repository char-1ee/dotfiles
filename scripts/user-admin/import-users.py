import argparse
import json
import subprocess


def import_users(user_dict: dict):
    for name, user in user_dict.items():
        options = [
            f"-d {user['home']}",
            "-M",
            f"-s {user['shell']}",
            f"-u {user['uid']}",
            "-N",
            f"-g {user['gid']}",
        ]
        if len(user["full_name"]) > 0:
            options.append(f"-c {user['full_name']}")
        if user["passwd"] != "x":
            options.append(f"-p {user['passwd']}")
        subprocess.run(
            f"useradd {' '.join(options)} {name}",
            shell=True,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("user_json")
    args = parser.parse_args()
    with open(args.user_json) as f:
        import_users(json.load(f))
