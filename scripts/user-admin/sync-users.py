import argparse
import json
import os
import subprocess
import tempfile

DIRNAME = os.path.dirname(os.path.realpath(__file__))
EXPORT_SCRIPT = os.path.join(DIRNAME, "export-users.py")
IMPORT_SCRIPT = os.path.join(DIRNAME, "import-users.py")


def get_master_users(uid_start: int, uid_end: int):
    r = subprocess.run(
        f"python3 {EXPORT_SCRIPT} --uid-start {uid_start} --uid-end {uid_end}",
        shell=True,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(r.stdout)


def get_worker_users(host: str, uid_start: int, uid_end: int):
    r = subprocess.run(
        f'ssh {host} "python3 {EXPORT_SCRIPT} --uid-start {uid_start} --uid-end {uid_end}"',
        shell=True,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(r.stdout)


def diff_users(master_users: dict, worker_users: dict) -> dict:
    master_keys = set(master_users.keys())
    worker_keys = set(worker_users.keys())

    common_keys = master_keys & worker_keys
    mising_keys = master_keys - worker_keys

    for key in common_keys:
        user1 = {**master_users[key]}
        user2 = {**worker_users[key]}
        del user1["shell"]
        del user2["shell"]
        assert user1 == user2, f'user {key} does not match'
    used_uids = set()
    for value in worker_users.values():
        uid = value["uid"]
        used_uids.add(uid)
    for key in mising_keys:
        uid = master_users[key]["uid"]
        assert uid not in used_uids, f'uid {uid} is used'
    return {key: master_users[key] for key in mising_keys}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hostfile", type=str)
    parser.add_argument("--uid-start", type=int, default=1000)
    parser.add_argument("--uid-end", type=int, default=2000)
    args = parser.parse_args()
    master_users = get_master_users(args.uid_start, args.uid_end)
    with open(args.hostfile) as f:
        for line in f.readlines():
            host = line.strip().split()[0]
            worker_users = get_worker_users(host, args.uid_start, args.uid_end)
            try:
                diff = diff_users(master_users, worker_users)
            except AssertionError as e:
                print(f"Error: {host} cannot be synced, skip. \n{e}")
                continue
            if len(diff) == 0:
                print(f"{host} does not need to be synced, skip.")
                continue
            print(f"Syncing {host} with {list(diff.keys())}")
            with tempfile.NamedTemporaryFile("w+", dir=DIRNAME) as f:
                json.dump(diff, f)
                f.flush()
                subprocess.run(
                    f'ssh {host} "python3 {IMPORT_SCRIPT} {os.path.join(DIRNAME, f.name)}"',
                    shell=True,
                    check=True,
                )
